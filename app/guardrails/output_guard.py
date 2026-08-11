"""Output guard: PII scrub, claim-grounding check, tone check.

Runs on the final agent response before it is returned to the customer.
Only the PII scrub modifies the response; grounding and tone problems are
recorded as guardrail flags on the trace (the eval harness and admin metrics
consume them) — a support answer is never silently rewritten.

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

_EMPATHY_MARKERS = (
    "sorry", "apolog", "understand", "regret", "frustrat", "inconvenien",
    "appreciate your patience", "thank you for bearing",
)


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

        flags.extend(self._unsupported_claims(response, trace))

        if sentiment == "angry" and not self._has_empathy(response):
            flags.append("tone_missing_empathy")

        return OutputGuardResult(response=scrubbed, flags=flags)

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
