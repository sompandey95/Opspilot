"""Per-org token budgets in Redis: daily + monthly counters, 429 when spent.

Keys: `budget:{org}:day:{YYYYMMDD}` (TTL 2 days) and
`budget:{org}:month:{YYYYMM}` (TTL 40 days) — natural expiry, no cron reset.
A limit of 0 disables that window. Consistent with the rate limiter, the
budget fails open on Redis errors: a dead Redis must not take chat down.

check() is called before the agent runs (with the tokens the request *will*
spend still unknown), record() after — so a burst can overshoot by one
request's worth per org. Acceptable: this is a cost brake, not an invariant.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from app.config import Settings

logger = logging.getLogger(__name__)

_DAY_TTL_SECONDS = 2 * 24 * 3600
_MONTH_TTL_SECONDS = 40 * 24 * 3600


@dataclass
class BudgetStatus:
    allowed: bool
    reason: str | None = None
    daily_used: int = 0
    monthly_used: int = 0


class TokenBudget:
    def __init__(self, settings: Settings, redis_client=None) -> None:
        self._daily_limit = settings.BUDGET_DAILY_TOKENS
        self._monthly_limit = settings.BUDGET_MONTHLY_TOKENS
        self._redis = redis_client

    def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            from app.db.redis import get_redis

            return get_redis()
        except RuntimeError:
            return None

    @staticmethod
    def _keys(org: str) -> tuple[str, str]:
        now = datetime.now(timezone.utc)
        return (
            f"budget:{org}:day:{now.strftime('%Y%m%d')}",
            f"budget:{org}:month:{now.strftime('%Y%m')}",
        )

    async def check(self, org: str) -> BudgetStatus:
        redis = self._get_redis()
        if redis is None:
            return BudgetStatus(allowed=True)

        day_key, month_key = self._keys(org)
        try:
            daily_used = int(await redis.get(day_key) or 0)
            monthly_used = int(await redis.get(month_key) or 0)
        except Exception as exc:
            logger.error("Budget check failed for org '%s' (%s) — allowing", org, exc)
            return BudgetStatus(allowed=True)

        if self._daily_limit and daily_used >= self._daily_limit:
            return BudgetStatus(
                allowed=False,
                reason=f"daily token budget exhausted ({daily_used}/{self._daily_limit})",
                daily_used=daily_used,
                monthly_used=monthly_used,
            )
        if self._monthly_limit and monthly_used >= self._monthly_limit:
            return BudgetStatus(
                allowed=False,
                reason=f"monthly token budget exhausted ({monthly_used}/{self._monthly_limit})",
                daily_used=daily_used,
                monthly_used=monthly_used,
            )
        return BudgetStatus(allowed=True, daily_used=daily_used, monthly_used=monthly_used)

    async def record(self, org: str, tokens: int) -> None:
        if tokens <= 0:
            return
        redis = self._get_redis()
        if redis is None:
            return

        day_key, month_key = self._keys(org)
        try:
            await redis.incrby(day_key, tokens)
            await redis.expire(day_key, _DAY_TTL_SECONDS)
            await redis.incrby(month_key, tokens)
            await redis.expire(month_key, _MONTH_TTL_SECONDS)
        except Exception as exc:
            logger.error("Budget record failed for org '%s': %s", org, exc)
