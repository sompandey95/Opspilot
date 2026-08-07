"""Redis-backed conversation history: `session:{id}`, TTL 2h.

History is a JSON list of OpenAI-format messages ({"role", "content"}) — only
user/assistant turns plus at most one leading system summary written by the
context window. Tool call/observation messages are deliberately not persisted:
they're bulky, model-specific, and the final assistant answer already carries
the outcome.

Every read/write refreshes the TTL (a session dies 2h after its *last*
activity). All operations degrade gracefully — a dead Redis means an empty
history and a lost turn, never a failed customer reply.
"""
from __future__ import annotations

import json
import logging

from app.config import Settings

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self, settings: Settings, redis_client=None) -> None:
        self._ttl = settings.SESSION_TTL_SECONDS
        self._redis = redis_client

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}"

    def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            from app.db.redis import get_redis

            return get_redis()
        except RuntimeError:
            return None

    async def get_history(self, session_id: str) -> list[dict]:
        redis = self._get_redis()
        if redis is None:
            return []
        try:
            raw = await redis.get(self._key(session_id))
            if not raw:
                return []
            await redis.expire(self._key(session_id), self._ttl)
            history = json.loads(raw)
            return history if isinstance(history, list) else []
        except Exception as exc:
            logger.error("Session history read failed for %s: %s", session_id, exc)
            return []

    async def append_exchange(
        self, session_id: str, user_message: str, assistant_message: str
    ) -> None:
        redis = self._get_redis()
        if redis is None:
            return
        history = await self.get_history(session_id)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})
        await self.save_history(session_id, history)

    async def save_history(self, session_id: str, history: list[dict]) -> None:
        """Overwrite the stored history (used after context-window compaction)."""
        redis = self._get_redis()
        if redis is None:
            return
        try:
            await redis.setex(self._key(session_id), self._ttl, json.dumps(history))
        except Exception as exc:
            logger.error("Session history write failed for %s: %s", session_id, exc)
