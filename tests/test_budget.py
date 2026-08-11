"""Budget tests: ₹ cost calculation (pricing lookup incl. deployment-name
matching, per-trace attribution) and per-org token budgets (daily/monthly
limits, disabled limits, fail-open on Redis errors)."""
import pytest

from app.budget.cost_calculator import PRICING_INR_PER_1M, CostCalculator
from app.budget.token_budget import TokenBudget
from app.config import Settings
from app.observability.trace import Trace


# --------------------------------------------------------------------- #
# Pricing lookup                                                          #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "deployment,expected_key",
    [
        ("gpt-5.4-mini", "gpt-5.4-mini"),
        ("gpt54-mini-prod", "gpt-5.4-mini"),   # mini must beat the gpt-5.4 prefix
        ("my-GPT-5.4-eu", "gpt-5.4"),
        ("gpt54", "gpt-5.4"),
        ("gpt-4o-judge", "gpt-4o"),
        ("text-embedding-3-large", "text-embedding-3-large"),
    ],
)
def test_pricing_lookup_matches_deployment_names(deployment, expected_key):
    assert CostCalculator.pricing_for(deployment) == PRICING_INR_PER_1M[expected_key]


def test_unknown_model_priced_at_most_expensive_tier():
    assert CostCalculator.pricing_for("mystery-model") == max(PRICING_INR_PER_1M.values())


def test_cost_math_per_million_tokens():
    # gpt-5.4-mini: ₹13.2 in / ₹52.8 out per 1M
    assert CostCalculator.cost_inr("gpt-5.4-mini", 1_000_000, 0) == pytest.approx(13.2)
    assert CostCalculator.cost_inr("gpt-5.4-mini", 0, 1_000_000) == pytest.approx(52.8)
    assert CostCalculator.cost_inr("gpt-5.4", 10_000, 2_000) == pytest.approx(
        (10_000 * 220.0 + 2_000 * 880.0) / 1_000_000
    )


def test_cost_for_trace_prices_each_step_by_its_model():
    trace = Trace(query="q")
    trace.steps.append(
        {"type": "llm", "model": "gpt-5.4", "input_tokens": 1000, "output_tokens": 500}
    )
    trace.steps.append(
        {"type": "llm", "model": "gpt-5.4-mini", "input_tokens": 2000, "output_tokens": 100}
    )
    trace.steps.append({"type": "tool_result", "tool": "x"})  # ignored
    # Totals include 300/50 classifier tokens beyond the steps → mini rate
    trace.input_tokens = 1000 + 2000 + 300
    trace.output_tokens = 500 + 100 + 50

    expected = (
        CostCalculator.cost_inr("gpt-5.4", 1000, 500)
        + CostCalculator.cost_inr("gpt-5.4-mini", 2000, 100)
        + CostCalculator.cost_inr("gpt-5.4-mini", 300, 50)
    )
    assert CostCalculator.cost_for_trace(trace) == pytest.approx(expected, abs=1e-6)


def test_cost_for_empty_trace_is_zero():
    assert CostCalculator.cost_for_trace(Trace(query="q")) == 0.0


# --------------------------------------------------------------------- #
# Token budget                                                            #
# --------------------------------------------------------------------- #

class FakeRedis:
    def __init__(self):
        self.store: dict[str, int] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key):
        value = self.store.get(key)
        return None if value is None else str(value)

    async def incrby(self, key, amount):
        self.store[key] = self.store.get(key, 0) + amount
        return self.store[key]

    async def expire(self, key, ttl):
        self.ttls[key] = ttl
        return True


class BrokenRedis:
    def __getattr__(self, name):
        async def boom(*args, **kwargs):
            raise ConnectionError("redis down")

        return boom


def budget(daily=1000, monthly=5000, redis=None) -> TokenBudget:
    settings = Settings(
        _env_file=None, BUDGET_DAILY_TOKENS=daily, BUDGET_MONTHLY_TOKENS=monthly
    )
    return TokenBudget(settings, redis_client=redis if redis is not None else FakeRedis())


async def test_fresh_org_is_allowed():
    status = await budget().check("acme")
    assert status.allowed
    assert status.daily_used == 0


async def test_daily_limit_blocks_at_boundary():
    redis = FakeRedis()
    b = budget(daily=1000, redis=redis)
    await b.record("acme", 999)
    assert (await b.check("acme")).allowed  # 999 < 1000

    await b.record("acme", 1)
    status = await b.check("acme")          # 1000 >= 1000
    assert not status.allowed
    assert "daily" in status.reason


async def test_monthly_limit_blocks_independently():
    redis = FakeRedis()
    b = budget(daily=0, monthly=500, redis=redis)  # daily disabled
    await b.record("acme", 600)
    status = await b.check("acme")
    assert not status.allowed
    assert "monthly" in status.reason


async def test_zero_limits_disable_budgeting():
    redis = FakeRedis()
    b = budget(daily=0, monthly=0, redis=redis)
    await b.record("acme", 10_000_000)
    assert (await b.check("acme")).allowed


async def test_orgs_are_isolated():
    redis = FakeRedis()
    b = budget(daily=100, redis=redis)
    await b.record("acme", 500)
    assert not (await b.check("acme")).allowed
    assert (await b.check("globex")).allowed


async def test_counters_get_ttls():
    redis = FakeRedis()
    b = budget(redis=redis)
    await b.record("acme", 10)
    day_key, month_key = b._keys("acme")
    assert redis.ttls[day_key] == 2 * 24 * 3600
    assert redis.ttls[month_key] == 40 * 24 * 3600


async def test_redis_down_fails_open():
    b = budget(daily=1, redis=BrokenRedis())
    assert (await b.check("acme")).allowed
    await b.record("acme", 100)  # must not raise


async def test_zero_token_record_is_noop():
    redis = FakeRedis()
    b = budget(redis=redis)
    await b.record("acme", 0)
    assert redis.store == {}
