"""Output guard: PII scrub, duplicate collapse, claim-grounding check, tone check.

Runs on the final agent response before it is returned to the customer. Only
two mutations are allowed — the PII scrub and collapsing a paragraph the model
emitted twice verbatim. Grounding and tone problems are recorded as guardrail
flags on the trace (the eval harness and admin metrics consume them) — the
substance of a support answer is never silently rewritten.

Grounding: verifiable claims (₹ amounts, ISO dates, day/hour windows,
percentages) in the response must appear somewhere in what the agent actually
observed this turn — tool results, retrieved chunks, or the customer's own
query. Anything else is flagged as a potential hallucination.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.guardrails.input_guard import mask_pii
from app.observability.trace import Trace

# Claim extractors: each match is a checkable factual assertion.
_CLAIM_PATTERNS = [
    re.compile(r"₹\s?[\d,]+(?:\.\d+)?"),                     # ₹1,299 / ₹ 500.50
    re.compile(r"\b(?:rs\.?|inr)\s?[\d,]+(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                     # ISO dates
    re.compile(r"\b\d+\s*(?:days?|hours?|weeks?)\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),                       # percentages
]

# Customer-facing answers are required to write dates as "23 Aug 2026", so an
# ISO-only claim check never sees them. These are canonicalised to ISO digits
# before grounding, letting "23 Aug 2026" match a "2026-08-23" tool result.
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_DAY_FIRST_DATE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b")
_MONTH_FIRST_DATE = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")

_EMPATHY_MARKERS = (
    "sorry", "apolog", "understand", "regret", "frustrat", "inconvenien",
    "appreciate your patience", "thank you for bearing",
)

# Only collapse repeats substantial enough to be a model stutter rather than a
# legitimately recurring short line (a bare "Thanks!", a repeated list marker).
_MIN_DUPLICATE_BLOCK_CHARS = 40


def has_verifiable_claim(text: str) -> bool:
    """True if the text asserts something checkable (amount, date, window, %)."""
    return any(pattern.search(text) for pattern in _CLAIM_PATTERNS) or any(
        _iter_month_name_dates(text)
    )


def _iter_month_name_dates(text: str):
    """Yield (as_written, YYYYMMDD) for every month-name date in the text."""
    for pattern, day_first in ((_DAY_FIRST_DATE, True), (_MONTH_FIRST_DATE, False)):
        for match in pattern.finditer(text):
            day, name = (
                (match.group(1), match.group(2)) if day_first
                else (match.group(2), match.group(1))
            )
            month = _MONTHS.get(name[:3].lower())
            if month is None:
                continue
            yield match.group(0), f"{match.group(3)}{month:02d}{int(day):02d}"


@dataclass
class OutputGuardResult:
    response: str
    flags: list[str] = field(default_factory=list)


def _normalise(text: str) -> str:
    """Lowercase and strip everything but alphanumerics so '₹1,299.00' in the
    response matches '1299.0' in a JSON tool result."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


class OutputGuard:
    def check(
        self, response: str, trace: Trace, sentiment: str = "neutral"
    ) -> OutputGuardResult:
        scrubbed, pii_flags = mask_pii(response)
        flags = [f"output_{f}" for f in pii_flags]

        deduped = self._collapse_duplicate_blocks(
            self._collapse_verbatim_repeat(scrubbed)
        )
        if deduped != scrubbed:
            flags.append("duplicate_block_collapsed")
            scrubbed = deduped

        flags.extend(self._unsupported_claims(response, trace))

        if sentiment == "angry" and not self._has_empathy(response):
            flags.append("tone_missing_empathy")

        return OutputGuardResult(response=scrubbed, flags=flags)

    # ------------------------------------------------------------------ #
    # Duplicate collapse                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _collapse_verbatim_repeat(response: str) -> str:
        """Drop the second copy when the model emitted the whole reply twice.

        The repeat is not reliably separated by a blank line, so every newline
        boundary is tested as a candidate midpoint.
        """
        stripped = response.strip()
        for separator in re.finditer(r"\n+", stripped):
            head = stripped[: separator.start()].strip()
            if len(_normalise(head)) < _MIN_DUPLICATE_BLOCK_CHARS:
                continue
            tail = stripped[separator.end():].strip()
            if _normalise(head) == _normalise(tail):
                return head
        return response

    @staticmethod
    def _collapse_duplicate_blocks(response: str) -> str:
        blocks = re.split(r"\n\s*\n", response)
        if len(blocks) < 2:
            return response

        kept: list[str] = []
        seen: set[str] = set()
        for block in blocks:
            key = _normalise(block)
            if len(key) >= _MIN_DUPLICATE_BLOCK_CHARS and key in seen:
                continue
            seen.add(key)
            kept.append(block)
        return "\n\n".join(kept) if len(kept) != len(blocks) else response

    # ------------------------------------------------------------------ #
    # Claim grounding                                                      #
    # ------------------------------------------------------------------ #

    def _unsupported_claims(self, response: str, trace: Trace) -> list[str]:
        context = self._observed_context(trace)
        context_norm = _normalise(context)
        context_digits = _digits(context)

        flags = []
        seen: set[str] = set()
        for pattern in _CLAIM_PATTERNS:
            for match in pattern.findall(response):
                claim = match.strip()
                if claim in seen:
                    continue
                seen.add(claim)
                claim_digits = _digits(claim)
                grounded = (
                    _normalise(claim) in context_norm
                    or (claim_digits and claim_digits in context_digits)
                )
                if not grounded:
                    flags.append(f"unsupported_claim:{claim}")

        for claim, iso_digits in _iter_month_name_dates(response):
            if claim in seen:
                continue
            seen.add(claim)
            grounded = (
                iso_digits in context_digits or _normalise(claim) in context_norm
            )
            if not grounded:
                flags.append(f"unsupported_claim:{claim}")
        return flags

    @staticmethod
    def _observed_context(trace: Trace) -> str:
        # Only real observations ground a claim — the model's own intermediate
        # thoughts don't count, or it could launder hallucinations through them.
        parts = [trace.query]
        for step in trace.steps:
            if step.get("type") == "tool_result" and step.get("data_preview"):
                parts.append(step["data_preview"])
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _has_empathy(response: str) -> bool:
        lowered = response.lower()
        return any(marker in lowered for marker in _EMPATHY_MARKERS)
