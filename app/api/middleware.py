"""API middleware: request-ID header, API-key auth, Redis sliding-window rate
limit, and the input guard on /chat.

Implemented as pure ASGI (not BaseHTTPMiddleware) because the input guard must
rewrite the request body (PII masking) before it reaches the route — reliable
body replacement needs control of `receive`.

Ordering in create_app puts CORSMiddleware outside this one so preflight
OPTIONS requests are answered before auth can reject them (we also skip
OPTIONS here as a belt-and-braces measure).

Failure posture: auth and the input guard fail closed; the rate limiter fails
open (a dead Redis must not take the API down with it).
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.guardrails.input_guard import InputGuard

logger = logging.getLogger(__name__)

_RATE_LIMIT_WINDOW_SECONDS = 60

# No auth / rate limiting / guarding: health must stay probeable, docs are dev.
_EXEMPT_PATHS = {"/", "/docs", "/openapi.json", "/api/v1/health"}

_CHAT_PATH = "/api/v1/chat"

_OPERATOR_PATH_PREFIXES = ("/api/v1/admin", "/api/v1/hitl")


class APIMiddleware:
    def __init__(
        self,
        app,
        settings: Settings | None = None,
        input_guard: InputGuard | None = None,
        redis_client=None,
    ) -> None:
        self.app = app
        self._settings = settings or get_settings()
        self._guard = input_guard or InputGuard(self._settings)
        self._redis = redis_client

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["method"] == "OPTIONS":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        send = self._with_request_id(send, request_id)
        path = scope["path"]

        if path not in _EXEMPT_PATHS:
            headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                       for k, v in scope["headers"]}

            if not self._authorized(headers):
                await self._reply(scope, receive, send, 401, "Invalid or missing API key")
                return

            client_host = scope["client"][0] if scope.get("client") else "unknown"
            client = headers.get("x-api-key") or client_host
            if await self._rate_limited(f"{self._bucket(path)}:{client}"):
                await self._reply(scope, receive, send, 429, "Rate limit exceeded — try again in a minute")
                return

        if path == _CHAT_PATH and scope["method"] == "POST":
            receive, blocked_detail = await self._guard_chat_body(scope, receive)
            if blocked_detail is not None:
                await self._reply(scope, receive, send, 400, blocked_detail)
                return

        await self.app(scope, receive, send)

    # ------------------------------------------------------------------ #
    # Auth + rate limit                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bucket(path: str) -> str:
        """Rate-limit bucket for a path.

        Operator surfaces (admin dashboard, approval queue) are polled on a
        timer and must not spend the quota that customer chat needs — one
        shared bucket lets an open console 429 real conversations.
        """
        return "operator" if path.startswith(_OPERATOR_PATH_PREFIXES) else "customer"

    def _authorized(self, headers: dict[str, str]) -> bool:
        expected = self._settings.OPSPILOT_API_KEY
        if not expected:  # auth disabled (local dev)
            return True
        return headers.get("x-api-key") == expected

    def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            from app.db.redis import get_redis

            return get_redis()
        except RuntimeError:
            return None

    async def _rate_limited(self, client: str) -> bool:
        redis = self._get_redis()
        if redis is None:
            return False

        key = f"ratelimit:{client}"
        now = time.time()
        try:
            pipe = redis.pipeline()
            pipe.zremrangebyscore(key, 0, now - _RATE_LIMIT_WINDOW_SECONDS)
            pipe.zadd(key, {f"{now}:{uuid.uuid4().hex[:8]}": now})
            pipe.zcard(key)
            pipe.expire(key, _RATE_LIMIT_WINDOW_SECONDS)
            results = await pipe.execute()
        except Exception as exc:
            logger.error("Rate limiter Redis failure (%s) — allowing request", exc)
            return False

        return results[2] > self._settings.RATE_LIMIT_PER_MINUTE

    # ------------------------------------------------------------------ #
    # Input guard on /chat                                                 #
    # ------------------------------------------------------------------ #

    async def _guard_chat_body(self, scope, receive):
        """Run the input guard over the chat body. Returns (receive, None) with
        a replayable — possibly PII-masked — body, or (receive, detail) when
        the request must be blocked."""
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break

        try:
            payload = json.loads(body)
            query = payload.get("query", "")
            assert isinstance(query, str)
        except (ValueError, AssertionError):
            # Malformed JSON is the route's 422 to give, not ours.
            return self._replay(body), None

        result = self._guard.check(query)
        if not result.allowed:
            logger.warning("Input guard blocked query (%s)", result.flags)
            return self._replay(body), result.reason

        scope.setdefault("state", {})["guardrail_flags"] = result.flags
        if result.query != query:
            payload["query"] = result.query
            body = json.dumps(payload).encode()
            self._set_content_length(scope, len(body))
        return self._replay(body), None

    @staticmethod
    def _replay(body: bytes):
        sent = False

        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        return receive

    @staticmethod
    def _set_content_length(scope, length: int) -> None:
        scope["headers"] = [
            (k, v) for k, v in scope["headers"] if k.lower() != b"content-length"
        ] + [(b"content-length", str(length).encode("latin-1"))]

    # ------------------------------------------------------------------ #
    # Responses                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _with_request_id(send, request_id: str):
        async def wrapped(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        return wrapped

    @staticmethod
    async def _reply(scope, receive, send, status: int, detail: str) -> None:
        response = JSONResponse(status_code=status, content={"detail": detail})
        await response(scope, receive, send)
