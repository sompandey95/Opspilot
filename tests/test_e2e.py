"""End-to-end lifecycle through the real HTTP app:

    POST /chat → middleware (auth, rate limit, input guard/PII mask)
      → intent classifier (fake LLM) → ReAct agent (scripted LLM, real tools
      against the mock order service, real HITL gate) → output guard
      → trace "persisted" with cost + prompt_version → budget recorded
      → session history stored.

Postgres is replaced by capturing `app.db.postgres.execute` (the suite runs
without live services); Redis by an in-memory fake. Everything else is the
real wiring produced by create_app().
"""
import json
import uuid

import httpx
import pytest

import app.db.postgres as postgres_module
from app.agent.intent_classifier import IntentClassifier
from app.agent.react_agent import ReActAgent
from app.budget.token_budget import TokenBudget
from app.config import Settings
from app.guardrails.output_guard import OutputGuard
from app.guardrails.schemas import SchemaValidator
from app.hitl.gate import HITLApprovalGate
from app.hitl.notifier import SlackNotifier
from app.hitl.queue import HITLQueue
from app.llm.client import LLMResponse, Usage
from app.main import create_app
from app.observability.tracer import Tracer
from app.session.context_window import ContextWindow
from app.session.manager import SessionManager
from app.session.summarizer import SessionSummarizer
from app.tools.registry import build_default_registry
from mock_services.order_service import main as svc
from mock_services.order_service.seed import build_store
from tests.test_react_agent import FixedConfidence, ScriptedLLM, answer, tool_call

# Trace insert arg positions (see _INSERT_SQL in app/observability/trace.py)
ARG_QUERY, ARG_PROMPT_VERSION, ARG_FLAGS, ARG_COST = 2, 14, 15, 16


class StaticJSONLLM:
    """Classifier LLM stub: always returns the same JSON classification."""

    def __init__(self, payload: dict):
        self._content = json.dumps(payload)

    async def complete(self, role, messages, **kwargs):
        return LLMResponse(content=self._content, usage=Usage(50, 30), model="fake-mini")


class FakeRedis:
    def __init__(self):
        self.store: dict = {}
        self.ttls: dict[str, int] = {}

    async def get(self, key):
        value = self.store.get(key)
        return None if value is None else str(value)

    async def setex(self, key, ttl, value):
        self.store[key] = value
        self.ttls[key] = ttl

    async def expire(self, key, ttl):
        self.ttls[key] = ttl
        return True

    async def incrby(self, key, amount):
        self.store[key] = int(self.store.get(key, 0)) + amount
        return self.store[key]


CLASSIFICATION = {
    "intent": "action_simple",
    "extracted_order_id": "ORD-2024-55001",
    "extracted_customer_id": None,
    "sentiment": "neutral",
    "language": "en",
    "reasoning": "order status lookup",
}


@pytest.fixture(autouse=True)
def fresh_store():
    customers, orders = build_store()
    svc.CUSTOMERS.clear()
    svc.CUSTOMERS.update(customers)
    svc.ORDERS.clear()
    svc.ORDERS.update(orders)
    yield


@pytest.fixture
def captured_inserts(monkeypatch):
    captured: list[tuple] = []

    async def fake_execute(query, *args):
        captured.append((query, args))
        return "INSERT 0 1"

    monkeypatch.setattr(postgres_module, "execute", fake_execute)
    return captured


def trace_inserts(captured: list[tuple]) -> list[tuple]:
    return [args for sql, args in captured if "INSERT INTO traces" in sql]


@pytest.fixture
async def order_client():
    transport = httpx.ASGITransport(app=svc.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://order-svc") as c:
        yield c


def build_e2e_app(
    agent_llm, order_client, session_redis, budget_redis, daily_limit=0, confidence=0.9
):
    settings = Settings(_env_file=None, BUDGET_DAILY_TOKENS=daily_limit, BUDGET_MONTHLY_TOKENS=0)

    app = create_app()
    queue = HITLQueue(settings)
    gate = HITLApprovalGate(queue, SlackNotifier(settings), settings)
    # hitl_queue wired into the registry too, same as _init_tools in app.main —
    # escalate_to_manager needs it to actually create a hitl_pending row.
    registry = build_default_registry(
        settings, retriever=None, order_client=order_client, hitl_queue=queue
    )

    app.state.retriever = None
    app.state.tool_registry = registry
    app.state.schema_validator = SchemaValidator(registry)
    app.state.hitl_queue = queue
    app.state.hitl_gate = gate
    app.state.intent_classifier = IntentClassifier(StaticJSONLLM(CLASSIFICATION))
    app.state.react_agent = ReActAgent(
        llm=agent_llm,
        tool_registry=registry,
        schema_validator=SchemaValidator(registry),
        settings=settings,
        hitl_gate=gate,
        confidence_scorer=FixedConfidence(confidence),
        redis_client=session_redis,
    )
    app.state.output_guard = OutputGuard()
    app.state.session_manager = SessionManager(settings, redis_client=session_redis)
    app.state.context_window = ContextWindow(
        settings, SessionSummarizer(None), token_counter=lambda t: 0
    )
    app.state.token_budget = TokenBudget(settings, redis_client=budget_redis)
    app.state.tracer = Tracer()
    return app


def client_for(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# --------------------------------------------------------------------- #
# The full lifecycle                                                      #
# --------------------------------------------------------------------- #

async def test_full_lifecycle_chat_to_trace_row(order_client, captured_inserts):
    llm = ScriptedLLM([
        tool_call("check_order_status", {"order_id": "ORD-2024-55001"}),
        answer("Your order ORD-2024-55001 is delayed — sorry about that!"),
    ])
    session_redis, budget_redis = FakeRedis(), FakeRedis()
    app = build_e2e_app(llm, order_client, session_redis, budget_redis)
    session_id = str(uuid.uuid4())

    async with client_for(app) as client:
        resp = await client.post(
            "/api/v1/chat",
            json={
                "query": "My PAN is ABCDE1234F. Where is my order ORD-2024-55001?",
                "session_id": session_id,
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "ORD-2024-55001" in body["response"]
    assert body["intent"] == "action_simple"
    assert body["confidence"] == 0.9
    assert not body["escalated"] and not body["pending_approval"]
    assert resp.headers.get("x-request-id")

    # The agent saw the masked query, never the PAN
    agent_user_msg = llm.calls[0]["messages"][-1]["content"]
    assert "[PAN_MASKED]" in agent_user_msg and "ABCDE1234F" not in agent_user_msg

    # Trace row: masked query, prompt version, guardrail flags, computed cost
    inserts = trace_inserts(captured_inserts)
    assert len(inserts) == 1
    args = inserts[0]
    assert "[PAN_MASKED]" in args[ARG_QUERY]
    assert args[ARG_PROMPT_VERSION] == "v3"
    assert "pii_pan" in json.loads(args[ARG_FLAGS])
    assert args[ARG_COST] and args[ARG_COST] > 0

    # Session stored this exchange (masked)
    history = json.loads(session_redis.store[f"session:{session_id}"])
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert "[PAN_MASKED]" in history[0]["content"]
    assert history[1]["content"] == body["response"]


async def test_low_confidence_escalation_actually_queues_for_a_human(order_client, captured_inserts):
    """Regression test for a real bug: the agent said "connecting you with a
    human agent" on low-confidence escalation, but nothing was ever queued or
    notified — only the classifier's explicit intent=="escalate" path did
    that. Any result.escalated must now reach the HITL queue."""
    llm = ScriptedLLM([answer("I'm not sure how to help with that.")])
    session_redis, budget_redis = FakeRedis(), FakeRedis()
    app = build_e2e_app(llm, order_client, session_redis, budget_redis, confidence=0.2)

    async with client_for(app) as client:
        resp = await client.post(
            "/api/v1/chat",
            json={"query": "Where is my order ORD-2024-55001?", "session_id": str(uuid.uuid4())},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["escalated"] is True
    assert body["confidence"] == 0.2

    hitl_inserts = [args for sql, args in captured_inserts if "INSERT INTO hitl_pending" in sql]
    assert len(hitl_inserts) == 1
    tool_name, tool_args = hitl_inserts[0][2], json.loads(hitl_inserts[0][3])
    assert tool_name == "escalate_to_manager"
    assert tool_args["reason"] == "low_confidence"
    assert "ORD-2024-55001" in tool_args["summary"]

    # Budget counters incremented for the default org
    assert any(k.startswith("budget:default:day:") for k in budget_redis.store)
    day_used = next(v for k, v in budget_redis.store.items() if ":day:" in k)
    assert day_used > 0


async def test_duplicate_escalation_message_notifies_once(order_client, captured_inserts, monkeypatch):
    """Regression test: escalate_to_manager is invoked directly from the route
    handler on low-confidence escalation, not through the agent's tool-call
    path — so it doesn't get react_agent's own idempotency wrapper for free.
    Without an explicit dedup key, a customer resending the identical message
    (impatient retry, slow response) would queue and Slack-notify a human once
    per resend instead of once per underlying issue."""
    dedup_redis = FakeRedis()
    monkeypatch.setattr("app.db.redis._redis", dedup_redis)

    llm = ScriptedLLM([
        answer("I'm not sure how to help with that."),
        answer("I'm not sure how to help with that."),
    ])
    session_redis, budget_redis = FakeRedis(), FakeRedis()
    app = build_e2e_app(llm, order_client, session_redis, budget_redis, confidence=0.2)
    session_id = str(uuid.uuid4())

    async with client_for(app) as client:
        for _ in range(2):
            resp = await client.post(
                "/api/v1/chat",
                json={"query": "Where is my order ORD-2024-55001?", "session_id": session_id},
            )
            assert resp.status_code == 200

    hitl_inserts = [args for sql, args in captured_inserts if "INSERT INTO hitl_pending" in sql]
    assert len(hitl_inserts) == 1


async def test_second_turn_sees_session_history(order_client, captured_inserts):
    llm = ScriptedLLM([
        tool_call("check_order_status", {"order_id": "ORD-2024-55001"}),
        answer("Your order ORD-2024-55001 is delayed."),
        answer("As I said, it's delayed — new ETA in 2 days."),
    ])
    session_redis, budget_redis = FakeRedis(), FakeRedis()
    app = build_e2e_app(llm, order_client, session_redis, budget_redis)
    session_id = str(uuid.uuid4())

    async with client_for(app) as client:
        first = await client.post(
            "/api/v1/chat",
            json={"query": "Where is my order ORD-2024-55001?", "session_id": session_id},
        )
        second = await client.post(
            "/api/v1/chat",
            json={"query": "So when will it arrive?", "session_id": session_id},
        )

    assert first.status_code == 200 and second.status_code == 200

    # Turn 2's LLM context contains turn 1 verbatim
    turn2_messages = llm.calls[-1]["messages"]
    contents = [m.get("content") for m in turn2_messages]
    assert "Where is my order ORD-2024-55001?" in contents
    assert "Your order ORD-2024-55001 is delayed." in contents

    # Both turns produced trace rows; session now holds two exchanges
    assert len(trace_inserts(captured_inserts)) == 2
    history = json.loads(session_redis.store[f"session:{session_id}"])
    assert len(history) == 4


async def test_budget_exhaustion_returns_429(order_client, captured_inserts):
    llm = ScriptedLLM([answer("The return window is 7 days.")])
    session_redis, budget_redis = FakeRedis(), FakeRedis()
    app = build_e2e_app(llm, order_client, session_redis, budget_redis, daily_limit=100)

    async with client_for(app) as client:
        first = await client.post("/api/v1/chat", json={"query": "what is the return window?"})
        second = await client.post("/api/v1/chat", json={"query": "and for electronics?"})

    assert first.status_code == 200
    assert second.status_code == 429
    assert "budget" in second.json()["detail"]
    # The blocked request never reached the agent or produced a trace
    assert len(trace_inserts(captured_inserts)) == 1
