"""Session layer tests: Redis history manager, context-window compaction
(empty / exactly-at-limit / over-limit), and the summariser's hard guarantee
that order numbers and customer IDs survive summarisation."""
import pytest

from app.config import Settings
from app.llm.client import LLMResponse, Usage
from app.session.context_window import SUMMARY_PREFIX, ContextWindow
from app.session.manager import SessionManager
from app.session.summarizer import SessionSummarizer, extract_identifiers


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.store[key] = value
        self.ttls[key] = ttl

    async def expire(self, key, ttl):
        self.ttls[key] = ttl
        return True


class BrokenRedis:
    def __getattr__(self, name):
        async def boom(*args, **kwargs):
            raise ConnectionError("redis down")

        return boom


class FakeLLM:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[dict] = []

    async def complete(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages, **kwargs})
        if self.error:
            raise self.error
        return LLMResponse(content=self.content, usage=Usage(100, 50), model="fake")


class FakeSummarizer:
    def __init__(self, summary="the summary"):
        self.summary = summary
        self.received: list[list[dict]] = []

    async def summarize(self, messages):
        self.received.append(messages)
        return self.summary


@pytest.fixture
def settings():
    return Settings(_env_file=None)


def exchange(n: int) -> list[dict]:
    return [
        {"role": "user", "content": f"question {n}"},
        {"role": "assistant", "content": f"answer {n}"},
    ]


# --------------------------------------------------------------------- #
# SessionManager                                                          #
# --------------------------------------------------------------------- #

async def test_missing_session_returns_empty_history(settings):
    manager = SessionManager(settings, redis_client=FakeRedis())
    assert await manager.get_history("nope") == []


async def test_append_and_get_roundtrip_refreshes_ttl(settings):
    redis = FakeRedis()
    manager = SessionManager(settings, redis_client=redis)

    await manager.append_exchange("s1", "hi", "hello!")
    await manager.append_exchange("s1", "where is ORD-2024-55001?", "it's delayed")

    history = await manager.get_history("s1")
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert history[2]["content"] == "where is ORD-2024-55001?"
    assert redis.ttls["session:s1"] == settings.SESSION_TTL_SECONDS  # 2h refresh


async def test_save_history_overwrites(settings):
    redis = FakeRedis()
    manager = SessionManager(settings, redis_client=redis)
    await manager.append_exchange("s1", "a", "b")
    await manager.save_history("s1", [{"role": "system", "content": "summary"}])
    assert await manager.get_history("s1") == [{"role": "system", "content": "summary"}]


async def test_dead_redis_degrades_to_stateless(settings):
    manager = SessionManager(settings, redis_client=BrokenRedis())
    assert await manager.get_history("s1") == []
    await manager.append_exchange("s1", "a", "b")  # must not raise


async def test_corrupt_history_returns_empty(settings):
    redis = FakeRedis()
    redis.store["session:s1"] = "not json {"
    manager = SessionManager(settings, redis_client=redis)
    assert await manager.get_history("s1") == []


# --------------------------------------------------------------------- #
# ContextWindow — edge cases from the spec                                #
# --------------------------------------------------------------------- #

def make_window(summarizer=None, max_tokens=100, keep=3):
    cfg = Settings(
        _env_file=None, CONTEXT_MAX_TOKENS=max_tokens, CONTEXT_KEEP_LAST_EXCHANGES=keep
    )
    # 1 token per word: deterministic, no tiktoken download in tests
    return ContextWindow(
        cfg, summarizer or FakeSummarizer(), token_counter=lambda t: len(t.split())
    )


async def test_empty_history_untouched():
    summarizer = FakeSummarizer()
    window = make_window(summarizer)
    assert await window.fit([]) == []
    assert summarizer.received == []


async def test_exactly_at_limit_untouched():
    summarizer = FakeSummarizer()
    history = [{"role": "user", "content": "one two three four five"}]  # 5 tokens
    window = make_window(summarizer, max_tokens=5)
    assert await window.fit(history) is history
    assert summarizer.received == []  # no summary at the boundary


async def test_over_limit_summarises_old_keeps_last_three_exchanges():
    summarizer = FakeSummarizer(summary="customer asked about ORD-2024-55001")
    history = []
    for n in range(6):
        history.extend(exchange(n))  # 6 exchanges, 2 words per message = 24 tokens

    window = make_window(summarizer, max_tokens=10, keep=3)
    fitted = await window.fit(history)

    # Head is one system summary, then the last 3 exchanges verbatim
    assert fitted[0]["role"] == "system"
    assert fitted[0]["content"] == SUMMARY_PREFIX + "customer asked about ORD-2024-55001"
    assert fitted[1:] == history[-6:]
    # Summariser saw exactly the old turns
    assert summarizer.received == [history[:-6]]


async def test_over_limit_with_few_exchanges_passes_through():
    """Recent exchanges alone bust the budget — nothing to summarise away."""
    summarizer = FakeSummarizer()
    history = exchange(1) + exchange(2)  # only 2 exchanges
    window = make_window(summarizer, max_tokens=1, keep=3)
    assert await window.fit(history) is history
    assert summarizer.received == []


# --------------------------------------------------------------------- #
# Summariser — order/customer IDs must survive                            #
# --------------------------------------------------------------------- #

HISTORY_WITH_IDS = [
    {"role": "user", "content": "My order ORD-2024-55001 is late, email rahul@gmail.com"},
    {"role": "assistant", "content": "Checked — it's delayed, ₹1,499 refund possible."},
    {"role": "user", "content": "Also check ORD-2024-78432 please"},
]


def test_extract_identifiers_finds_orders_emails_phones():
    ids = extract_identifiers(
        "ORD-2024-55001 and ord-2024-78432, mail rahul@gmail.com, phone 9876543210"
    )
    assert "ORD-2024-55001" in ids
    assert "ORD-2024-78432" in ids  # case-normalised
    assert "rahul@gmail.com" in ids
    assert "9876543210" in ids


async def test_llm_summary_used_when_it_keeps_ids():
    llm = FakeLLM(content="Customer's ORD-2024-55001 and ORD-2024-78432 are delayed; contact rahul@gmail.com.")
    summarizer = SessionSummarizer(llm)
    summary = await summarizer.summarize(HISTORY_WITH_IDS)
    assert summary.startswith("Customer's ORD-2024-55001")
    assert "Identifiers from earlier turns" not in summary  # nothing was dropped


async def test_ids_dropped_by_llm_are_appended():
    llm = FakeLLM(content="Customer has two delayed orders and wants refunds.")
    summarizer = SessionSummarizer(llm)
    summary = await summarizer.summarize(HISTORY_WITH_IDS)
    assert "ORD-2024-55001" in summary
    assert "ORD-2024-78432" in summary
    assert "rahul@gmail.com" in summary


async def test_llm_failure_falls_back_and_keeps_ids():
    llm = FakeLLM(error=TimeoutError("azure down"))
    summarizer = SessionSummarizer(llm)
    summary = await summarizer.summarize(HISTORY_WITH_IDS)
    assert "ORD-2024-55001" in summary
    assert "ORD-2024-78432" in summary


async def test_no_llm_at_all_still_summarises():
    summarizer = SessionSummarizer(None)
    summary = await summarizer.summarize(HISTORY_WITH_IDS)
    assert "ORD-2024-55001" in summary
