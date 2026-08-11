"""Hallucination detection — deterministic claim extraction + grounding.

No LLM: extracts verifiable claims from the answer and checks each against
what the agent actually observed (query + retrieved chunks + tool results).
Claim types: order IDs, refund IDs, ₹ amounts, ISO dates, day/hour windows,
percentages, phone numbers.

Severity rule from the spec: ANY fabricated order/refund ID ⇒ the scenario is
a hallucination fail outright, regardless of everything else. Other ungrounded
claims are reported and also fail the scenario (a support answer must not
invent numbers), but they're distinguished in the result for debugging.

`tool_call_args` (optional) also gets scanned for fabricated IDs: an agent
that invents an order ID, feeds it to a tool, and gets a failure back — then
recovers with an honest "I couldn't find that, please share the full ID" —
never repeats the fabricated ID in `answer`, so scanning `answer` alone misses
the fabrication entirely. The tool call itself is the hallucination.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_ID_PATTERNS = [
    re.compile(r"\bORD-\d{4}-\d{4,6}\b", re.IGNORECASE),
    re.compile(r"\bREF-[A-Za-z0-9-]+\b", re.IGNORECASE),
]

_FACT_PATTERNS = [
    re.compile(r"₹\s?[\d,]+(?:\.\d+)?"),
    re.compile(r"\b(?:rs\.?|inr)\s?[\d,]+(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d+\s*(?:days?|hours?|weeks?|din)\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    re.compile(r"\b(?:\+91[ -]?)?[6-9]\d{9}\b"),
]


@dataclass
class HallucinationResult:
    hallucinated: bool
    fabricated_ids: list[str] = field(default_factory=list)
    ungrounded_claims: list[str] = field(default_factory=list)
    total_claims: int = 0


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def check_hallucination(
    answer: str, context: str, tool_call_args: str = ""
) -> HallucinationResult:
    """`context` = everything the agent legitimately saw: the customer query,
    retrieved chunk contents, and tool-result payloads, concatenated.

    `tool_call_args` = every argument the agent passed to a tool this turn
    (regardless of whether the call succeeded), concatenated. Checked against
    the same `context` — an ID the agent supplied to a tool must already have
    come from the customer or an earlier tool result, not be invented fresh.
    """
    context_norm = _normalise(context)
    context_digits = _digits(context)

    fabricated_ids: list[str] = []
    ungrounded: list[str] = []
    total = 0
    seen: set[str] = set()

    for pattern in _ID_PATTERNS:
        for match in pattern.findall(answer) + pattern.findall(tool_call_args):
            claim = match.upper()
            if claim in seen:
                continue
            seen.add(claim)
            total += 1
            if _normalise(claim) not in context_norm:
                fabricated_ids.append(claim)

    for pattern in _FACT_PATTERNS:
        for match in pattern.findall(answer):
            claim = match.strip()
            if claim.lower() in seen:
                continue
            seen.add(claim.lower())
            total += 1
            claim_digits = _digits(claim)
            grounded = (
                _normalise(claim) in context_norm
                or (claim_digits and claim_digits in context_digits)
            )
            if not grounded:
                ungrounded.append(claim)

    return HallucinationResult(
        hallucinated=bool(fabricated_ids or ungrounded),
        fabricated_ids=fabricated_ids,
        ungrounded_claims=ungrounded,
        total_claims=total,
    )
