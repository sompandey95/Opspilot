"""Keep session history inside the context budget.

tiktoken count over the whole history; while it fits CONTEXT_MAX_TOKENS
(12k, inclusive — exactly-at-limit passes through untouched) nothing changes.
Over the limit, everything except the last CONTEXT_KEEP_LAST_EXCHANGES (3)
user→assistant exchanges is replaced by one system summary message produced by
SessionSummarizer (which guarantees ORD-/customer IDs survive).

tiktoken loads its encoding lazily and can fail offline (first use downloads
the BPE file) — in that case a ~4-chars-per-token estimate keeps the window
working; the compaction logic is identical either way.
"""
from __future__ import annotations

import logging

from app.config import Settings
from app.session.summarizer import SessionSummarizer

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = "Summary of earlier conversation: "

_encoding = None
_encoding_failed = False


def _count_tokens_default(text: str) -> int:
    global _encoding, _encoding_failed
    if _encoding is None and not _encoding_failed:
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("o200k_base")
        except Exception as exc:
            logger.error("tiktoken unavailable (%s) — using chars/4 estimate", exc)
            _encoding_failed = True
    if _encoding is not None:
        return len(_encoding.encode(text))
    return max(1, len(text) // 4)


class ContextWindow:
    def __init__(
        self,
        settings: Settings,
        summarizer: SessionSummarizer,
        token_counter=None,
    ) -> None:
        self._max_tokens = settings.CONTEXT_MAX_TOKENS
        self._keep_exchanges = settings.CONTEXT_KEEP_LAST_EXCHANGES
        self._summarizer = summarizer
        self._count = token_counter or _count_tokens_default

    def count_history_tokens(self, history: list[dict]) -> int:
        return sum(self._count(m.get("content") or "") for m in history)

    async def fit(self, history: list[dict]) -> list[dict]:
        if not history:
            return history
        if self.count_history_tokens(history) <= self._max_tokens:
            return history

        old, recent = self._split(history)
        if not old:
            # Pathological: the last exchanges alone bust the limit. Nothing
            # sane to summarise away — pass through and let the model cope.
            logger.warning("Recent exchanges alone exceed the context budget")
            return history

        try:
            summary = await self._summarizer.summarize(old)
        except Exception as exc:
            # Summariser has its own fallbacks; this is belt-and-braces.
            logger.error("Context summarisation failed (%s) — dropping old turns", exc)
            return recent

        return [{"role": "system", "content": SUMMARY_PREFIX + summary}] + recent

    def _split(self, history: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split into (old, recent) where recent = the last N user→assistant
        exchanges, kept verbatim. An exchange starts at a user message."""
        starts = [i for i, m in enumerate(history) if m.get("role") == "user"]
        if len(starts) <= self._keep_exchanges:
            return [], history
        cut = starts[-self._keep_exchanges]
        return history[:cut], history[cut:]
