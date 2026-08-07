"""Summarise old conversation turns with GPT-5.4-mini.

The one hard requirement (eval-tested): order numbers and customer identifiers
must survive summarisation. The LLM is instructed to keep them, and a
deterministic post-check appends any ORD-/customer IDs the summary dropped —
so the guarantee holds even if the model ignores the instruction. If the LLM
is unavailable the fallback is a truncated transcript plus the extracted IDs:
degraded quality, same guarantee, never an exception.
"""
from __future__ import annotations

import logging
import re

from app.llm.client import LLMClient, ModelRole

logger = logging.getLogger(__name__)

_ORDER_ID_RE = re.compile(r"\bORD-\d{4}-\d{4,6}\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+91[ -]?)?[6-9]\d{9}\b")

_SUMMARY_TIMEOUT_S = 20.0
_FALLBACK_TRANSCRIPT_CHARS = 1200

_SYSTEM_PROMPT = """Summarise this ShopEasy customer-support conversation in 3-6 sentences for the agent handling the next turn.

MUST preserve exactly, verbatim: every order ID (ORD-...), customer email/phone, amount (₹), and date mentioned. Also capture: what the customer wants, what was already checked or done, and any pending action."""


def extract_identifiers(text: str) -> list[str]:
    ids: list[str] = []
    for pattern in (_ORDER_ID_RE, _EMAIL_RE, _PHONE_RE):
        for match in pattern.findall(text):
            normalised = match.upper() if pattern is _ORDER_ID_RE else match
            if normalised not in ids:
                ids.append(normalised)
    return ids


class SessionSummarizer:
    def __init__(self, llm: LLMClient | None) -> None:
        self._llm = llm

    async def summarize(self, messages: list[dict]) -> str:
        transcript = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content') or ''}" for m in messages
        )
        identifiers = extract_identifiers(transcript)

        summary = await self._llm_summary(transcript)
        if summary is None:
            summary = self._fallback_summary(transcript)

        # Deterministic guarantee: no identifier is lost to summarisation.
        dropped = [i for i in identifiers if i.lower() not in summary.lower()]
        if dropped:
            summary += f"\nIdentifiers from earlier turns: {', '.join(dropped)}"
        return summary

    async def _llm_summary(self, transcript: str) -> str | None:
        if self._llm is None:
            return None
        try:
            response = await self._llm.complete(
                role=ModelRole.SUMMARIZER,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                max_completion_tokens=400,
                timeout=_SUMMARY_TIMEOUT_S,
            )
        except Exception as exc:
            logger.error("Session summary LLM call failed: %s", exc)
            return None
        return response.content.strip() if response.content else None

    @staticmethod
    def _fallback_summary(transcript: str) -> str:
        return (
            "Earlier conversation (truncated, summariser unavailable): "
            + transcript[-_FALLBACK_TRANSCRIPT_CHARS:]
        )
