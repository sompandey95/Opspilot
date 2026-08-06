"""Input guard: length cap, Indian-PII masking, prompt-injection blocking.

Runs in the API middleware before the query reaches the agent. PII is masked
(the agent never needs a card/Aadhaar/PAN number to help), injection attempts
and oversized queries are blocked outright.

PII detection order matters: cards (16 digits) are masked before Aadhaar
(12 digits) so the Aadhaar pattern can't claim the first 12 digits of a card.

The naive UPI pattern would also match emails ("user@gmail"), so a handle is
only masked when the part after '@' has no dotted TLD — UPI VPAs like
"name@okhdfcbank" never contain a dot, emails practically always do.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import Settings

# (flag, pattern, replacement) — applied in order.
_PII_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    (
        "pii_card",
        re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
        "[CARD_MASKED]",
    ),
    (
        "pii_aadhaar",
        re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
        "[AADHAAR_MASKED]",
    ),
    (
        "pii_pan",
        re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
        "[PAN_MASKED]",
    ),
    (
        # '@' handle with no dotted domain ⇒ UPI VPA, not an email.
        "pii_upi",
        re.compile(r"\b[A-Za-z0-9][A-Za-z0-9._-]+@[A-Za-z]{2,}(?!\.[A-Za-z])\b"),
        "[UPI_MASKED]",
    ),
]

_INJECTION_SIGNALS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore the above",
    "disregard your instructions",
    "disregard previous",
    "system prompt",
    "your instructions are",
    "you are now",
    "act as if",
    "pretend you are",
    "developer mode",
    "jailbreak",
    "reveal your prompt",
    "repeat your instructions",
]

BLOCKED_INJECTION_DETAIL = (
    "This query looks like an attempt to manipulate the assistant and was blocked."
)


@dataclass
class InputGuardResult:
    allowed: bool
    query: str
    flags: list[str] = field(default_factory=list)
    reason: str | None = None


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Mask Indian PII (cards, Aadhaar, PAN, UPI VPAs). Shared with the output
    guard. Returns (masked_text, flags)."""
    flags: list[str] = []
    for flag, pattern, replacement in _PII_PATTERNS:
        text, count = pattern.subn(replacement, text)
        if count:
            flags.append(flag)
    return text, flags


class InputGuard:
    def __init__(self, settings: Settings) -> None:
        self._max_length = settings.INPUT_MAX_QUERY_LENGTH

    def check(self, query: str) -> InputGuardResult:
        if len(query) > self._max_length:
            return InputGuardResult(
                allowed=False,
                query=query,
                flags=["query_too_long"],
                reason=f"Query exceeds the {self._max_length}-character limit.",
            )

        lowered = query.lower()
        matched = [s for s in _INJECTION_SIGNALS if s in lowered]
        if matched:
            return InputGuardResult(
                allowed=False,
                query=query,
                flags=[f"injection:{matched[0]}"],
                reason=BLOCKED_INJECTION_DETAIL,
            )

        masked, flags = mask_pii(query)
        return InputGuardResult(allowed=True, query=masked, flags=flags)
