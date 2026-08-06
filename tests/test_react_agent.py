"""ReAct agent tests — scripted fake LLM, real tools against the mock order app."""
import asyncio
import json

import httpx
import pytest

from app.agent.intent_classifier import IntentResult, IntentType
from app.agent.react_agent import (
    ESCALATION_MESSAGE,
    LOW_CONFIDENCE_MESSAGE,
    PENDING_APPROVAL_MESSAGE,
    ApprovalDecision,
    ApprovalStatus,
    ReActAgent,
)
from app.config import Settings
from app.guardrails.schemas import SchemaValidator
from app.llm.client import LLMResponse, ToolCallRequest, Usage
from app.tools.registry import build_default_registry
from mock_services.order_service import main as svc
from mock_services.order_service.seed import build_store


# --------------------------------------------------------------------- #
# Fakes                                                                   #
# --------------------------------------------------------------------- #

class ScriptedLLM:
    """Pops scripted responses; repeats the last one when exhausted."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": [dict(m) for m in messages], **kwargs})
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.store[key] = value


class FixedConfidence:
    def __init__(self, value: float):
        self.value = value

    def score(self, query, answer, trace):
        return self.value


def answer(text: str) -> LLMResponse:
    return LLMResponse(content=text, usage=Usage(100, 50), model="fake")


def tool_call(name: str, args: dict | None, call_id: str = "call_1",
              raw: str | None = None) -> LLMResponse:
    raw_args = raw if raw is not None else json.dumps(args)
    parsed = args
    parse_error = None
    if raw is not None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            parsed = None
            parse_error = f"arguments are not valid JSON: {exc}"
    return LLMResponse(
        content=None,
        tool_calls=[ToolCallRequest(id=call_id, name=name, arguments_raw=raw_args,
                                    arguments=parsed, parse_error=parse_error)],
        usage=Usage(100, 20),
        model="fake",
    )


def intent(kind=IntentType.ACTION_SIMPLE, order_id=None) -> IntentResult:
    return IntentResult(intent=kind, extracted_order_id=order_id, usage=Usage(50, 30))


# --------------------------------------------------------------------- #
# Fixtures                                                                #
# --------------------------------------------------------------------- #

@pytest.fixture
def settings():
    return Settings(_env_file=None)


@pytest.fixture(autouse=True)
def fresh_store():
    customers, orders = build_store()
    svc.CUSTOMERS.clear()
    svc.CUSTOMERS.update(customers)
    svc.ORDERS.clear()
    svc.ORDERS.update(orders)
    yield


@pytest.fixture
async def order_client():
    transport = httpx.ASGITransport(app=svc.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def registry(settings, order_client):
    return build_default_registry(settings, retriever=None, order_client=order_client)


def make_agent(llm, registry, settings, **overrides) -> ReActAgent:
    defaults = dict(
        llm=llm,
        tool_registry=registry,
        schema_validator=SchemaValidator(registry),
        settings=settings,
        confidence_scorer=FixedConfidence(0.9),
        redis_client=FakeRedis(),
    )
    defaults.update(overrides)
    return ReActAgent(**defaults)


# --------------------------------------------------------------------- #
# Happy path: tool call → observation → answer                            #
# --------------------------------------------------------------------- #

async def test_tool_call_then_answer(registry, settings):
    llm = ScriptedLLM([
        tool_call("check_order_status", {"order_id": "ORD-2024-55001"}),
        answer("Your order ORD-2024-55001 is delayed; new ETA is soon."),
    ])
    agent = make_agent(llm, registry, settings)

    result = await agent.run("Where is my order ORD-2024-55001?", intent())

    assert not result.escalated
    assert result.confidence == 0.9
    assert "ORD-2024-55001" in result.response

    # Message history is well-formed: assistant tool_calls + paired tool msg
    second_call_messages = llm.calls[1]["messages"]
    assistant = next(m for m in second_call_messages if m.get("tool_calls"))
    tool_msg = next(m for m in second_call_messages if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == assistant["tool_calls"][0]["id"]
    observation = json.loads(tool_msg["content"])
    assert observation["success"] is True
    assert observation["data"]["status"] == "delayed"

    # Trace recorded llm steps + tool result and aggregated tokens
    types = [s["type"] for s in result.trace.steps]
    assert types.count("llm") == 2
    assert "tool_result" in types
    assert result.trace.total_tokens == 100 + 50 + (100 + 20) + 80  # + classifier usage
    assert result.trace.prompt_version == "v1"


async def test_final_response_recorded_on_trace(registry, settings):
    llm = ScriptedLLM([answer("Our return window is 7 days for most items.")])
    agent = make_agent(llm, registry, settings)
    result = await agent.run("What is the return window?", intent(IntentType.FAQ))
    assert result.trace.response == result.response
    assert result.trace.confidence == 0.9


# --------------------------------------------------------------------- #
# Malformed tool calls: agent is told and retries                         #
# --------------------------------------------------------------------- #

async def test_invalid_args_fed_back_and_retried(registry, settings):
    llm = ScriptedLLM([
        tool_call("check_order_status", {"order_id": "12345"}),  # bad pattern
        tool_call("check_order_status", {"order_id": "ORD-2024-55001"}),
        answer("Found it — your order is delayed."),
    ])
    agent = make_agent(llm, registry, settings)

    result = await agent.run("where is my order", intent())

    assert not result.escalated
    error_msg = next(
        m for m in llm.calls[1]["messages"] if m["role"] == "tool"
    )
    assert error_msg["content"].startswith("Error:")
    assert any(s["type"] == "validation_error" for s in result.trace.steps)


async def test_unparseable_json_args_fed_back(registry, settings):
    llm = ScriptedLLM([
        tool_call("check_order_status", None, raw='{"order_id": broken'),
        answer("Sorry, could you share your order ID again?"),
    ])
    agent = make_agent(llm, registry, settings)

    result = await agent.run("where is my order", intent())

    assert not result.escalated
    error_msg = next(m for m in llm.calls[1]["messages"] if m["role"] == "tool")
    assert "not valid JSON" in error_msg["content"]


async def test_unknown_tool_fed_back(registry, settings):
    llm = ScriptedLLM([
        tool_call("teleport_order", {"order_id": "ORD-2024-55001"}),
        answer("Let me handle that differently."),
    ])
    agent = make_agent(llm, registry, settings)

    result = await agent.run("teleport my order", intent())
    error_msg = next(m for m in llm.calls[1]["messages"] if m["role"] == "tool")
    assert "Unknown tool" in error_msg["content"]
    assert not result.escalated


# --------------------------------------------------------------------- #
# Max steps                                                               #
# --------------------------------------------------------------------- #

async def test_max_steps_exhausted_escalates(registry, settings):
    looping = Settings(_env_file=None, MAX_AGENT_STEPS=3)
    llm = ScriptedLLM([tool_call("check_order_status", {"order_id": "ORD-2024-55001"})])
    agent = make_agent(llm, registry, looping)

    result = await agent.run("where is my order", intent())

    assert result.escalated
    assert result.response == ESCALATION_MESSAGE
    assert len(llm.calls) == 3
    assert any(
        s.get("reason") == "max_steps_reached"
        for s in result.trace.steps if s["type"] == "escalation"
    )


# --------------------------------------------------------------------- #
# Idempotency: no double refunds                                          #
# --------------------------------------------------------------------- #

async def test_idempotent_replay_of_state_changing_tool(registry, settings):
    shared_redis = FakeRedis()
    script = lambda: ScriptedLLM([
        tool_call("process_refund",
                  {"order_id": "ORD-2024-78432", "reason": "delivery_delayed"}),
        answer("Refund of ₹1,299 processed."),
    ])

    agent1 = make_agent(script(), registry, settings, redis_client=shared_redis)
    result1 = await agent1.run("refund my late order ORD-2024-78432", intent())
    step1 = next(s for s in result1.trace.steps if s["type"] == "tool_result")
    assert step1["success"] and not step1["from_cache"]

    # Order is now refunded — a real second execution would fail. The cached
    # result must be returned instead.
    agent2 = make_agent(script(), registry, settings, redis_client=shared_redis)
    result2 = await agent2.run("refund my late order ORD-2024-78432", intent())
    step2 = next(s for s in result2.trace.steps if s["type"] == "tool_result")
    assert step2["success"] and step2["from_cache"]
    assert len(shared_redis.store) == 1


async def test_failed_state_changing_call_not_cached(registry, settings):
    redis = FakeRedis()
    llm = ScriptedLLM([
        tool_call("process_refund",
                  {"order_id": "ORD-2024-54000", "reason": "delivery_delayed"}),
        answer("That order isn't eligible for a refund."),
    ])
    agent = make_agent(llm, registry, settings, redis_client=redis)

    result = await agent.run("refund ORD-2024-54000", intent())
    step = next(s for s in result.trace.steps if s["type"] == "tool_result")
    assert not step["success"]
    assert redis.store == {}  # failures stay retryable


async def test_read_only_tools_bypass_idempotency_cache(registry, settings):
    redis = FakeRedis()
    llm = ScriptedLLM([
        tool_call("check_order_status", {"order_id": "ORD-2024-55001"}),
        answer("Delayed."),
    ])
    agent = make_agent(llm, registry, settings, redis_client=redis)
    await agent.run("status?", intent())
    assert redis.store == {}


# --------------------------------------------------------------------- #
# HITL gate seam                                                          #
# --------------------------------------------------------------------- #

async def test_high_risk_tool_passes_through_stub_gate(registry, settings):
    llm = ScriptedLLM([
        tool_call("process_refund",
                  {"order_id": "ORD-2024-78432", "reason": "delivery_delayed"}),
        answer("Refund processed."),
    ])
    agent = make_agent(llm, registry, settings)  # default AutoApproveHITLGate

    result = await agent.run("refund please", intent())

    assert result.trace.hitl_triggered
    hitl_step = next(s for s in result.trace.steps if s["type"] == "hitl")
    assert hitl_step["status"] == "approved"
    assert hitl_step["reason"] == "phase4_stub_auto_approve"


class RejectingGate:
    async def request_approval(self, tool, args, trace):
        return ApprovalDecision(status=ApprovalStatus.REJECTED, reason="amount too high")


async def test_hitl_rejection_informs_agent_and_continues(registry, settings):
    llm = ScriptedLLM([
        tool_call("process_refund",
                  {"order_id": "ORD-2024-78432", "reason": "delivery_delayed"}),
        answer("A supervisor declined the automatic refund; I've noted your case."),
    ])
    agent = make_agent(llm, registry, settings, hitl_gate=RejectingGate())

    result = await agent.run("refund please", intent())

    assert not result.pending_approval
    rejection_msg = next(m for m in llm.calls[1]["messages"] if m["role"] == "tool")
    assert "rejected by a supervisor" in rejection_msg["content"]
    assert "amount too high" in rejection_msg["content"]
    # Order must NOT have been refunded
    order = await registry.get("check_order_status").execute(order_id="ORD-2024-78432")
    assert order.data["status"] != "refunded"


class TimeoutGate:
    async def request_approval(self, tool, args, trace):
        return ApprovalDecision(status=ApprovalStatus.TIMEOUT)


async def test_hitl_timeout_returns_pending(registry, settings):
    llm = ScriptedLLM([
        tool_call("process_refund",
                  {"order_id": "ORD-2024-78432", "reason": "delivery_delayed"}),
    ])
    agent = make_agent(llm, registry, settings, hitl_gate=TimeoutGate())

    result = await agent.run("refund please", intent())

    assert result.pending_approval
    assert result.response == PENDING_APPROVAL_MESSAGE
    assert not result.escalated


async def test_none_risk_tools_skip_the_gate(registry, settings):
    class ExplodingGate:
        async def request_approval(self, tool, args, trace):
            raise AssertionError("gate must not be called for zero-risk tools")

    llm = ScriptedLLM([
        tool_call("search_knowledge", {"query": "return window"}),
        answer("The return window is 7 days."),
    ])
    agent = make_agent(llm, registry, settings, hitl_gate=ExplodingGate())
    result = await agent.run("what's the return window?", intent(IntentType.FAQ))
    assert not result.trace.hitl_triggered


async def test_low_risk_auto_approval_not_recorded_as_hitl(registry, settings):
    """LOW-risk tools now pass through the gate (audit happens there), but an
    auto-approval must not mark the trace as HITL-triggered."""
    calls = []

    class RecordingGate:
        async def request_approval(self, tool, args, trace):
            calls.append(tool.name)
            return ApprovalDecision(status=ApprovalStatus.APPROVED, reason="auto")

    llm = ScriptedLLM([
        tool_call("check_order_status", {"order_id": "ORD-2024-55001"}),
        answer("Delayed."),
    ])
    agent = make_agent(llm, registry, settings, hitl_gate=RecordingGate())
    result = await agent.run("status?", intent())
    assert calls == ["check_order_status"]
    assert not result.trace.hitl_triggered


# --------------------------------------------------------------------- #
# Confidence + escalation paths                                           #
# --------------------------------------------------------------------- #

async def test_low_confidence_escalates(registry, settings):
    llm = ScriptedLLM([answer("Maybe it's fine?")])
    agent = make_agent(llm, registry, settings, confidence_scorer=FixedConfidence(0.2))

    result = await agent.run("weird question", intent(IntentType.FAQ))

    assert result.escalated
    assert result.response == LOW_CONFIDENCE_MESSAGE
    assert result.confidence == 0.2
    assert result.trace.confidence == 0.2


async def test_out_of_scope_decline_skips_confidence_gate(registry, settings):
    llm = ScriptedLLM([answer("I can only help with ShopEasy support questions.")])
    agent = make_agent(llm, registry, settings, confidence_scorer=FixedConfidence(0.5))

    result = await agent.run("write me a poem", intent(IntentType.OUT_OF_SCOPE))

    assert not result.escalated
    assert result.confidence == 0.5  # recorded, but not gated
    assert "ShopEasy" in result.response


async def test_escalate_intent_skips_llm_entirely(registry, settings):
    llm = ScriptedLLM([answer("should never be called")])
    agent = make_agent(llm, registry, settings)

    result = await agent.run("get me a human", intent(IntentType.ESCALATE))

    assert result.escalated
    assert result.response == ESCALATION_MESSAGE
    assert llm.calls == []


async def test_llm_timeout_escalates(registry, settings):
    llm = ScriptedLLM([asyncio.TimeoutError()])
    agent = make_agent(llm, registry, settings)

    result = await agent.run("anything", intent())

    assert result.escalated
    assert result.response == ESCALATION_MESSAGE
    assert any(
        s.get("reason") == "llm_error"
        for s in result.trace.steps if s["type"] == "escalation"
    )


# --------------------------------------------------------------------- #
# Message construction                                                    #
# --------------------------------------------------------------------- #

async def test_system_prompt_and_intent_context_in_messages(registry, settings):
    llm = ScriptedLLM([answer("ok, done")])
    agent = make_agent(llm, registry, settings)

    await agent.run(
        "Mera order late hai",
        IntentResult(
            intent=IntentType.ACTION_SIMPLE,
            extracted_order_id="ORD-2024-55001",
            sentiment="angry",
            language="mixed",
        ),
    )

    messages = llm.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "OpsPilot" in messages[0]["content"]
    context = messages[1]["content"]
    assert "ORD-2024-55001" in context
    assert "upset" in context
    assert "Hindi-English" in context
    assert messages[-1] == {"role": "user", "content": "Mera order late hai"}
    assert llm.calls[0]["tools"], "tool schemas must be passed to the LLM"
