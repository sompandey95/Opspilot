"""Eval runner end-to-end: real scenarios from the golden dataset, real
ReAct agent + real tools against the mock order service, real input/output
guards — only the LLMs are scripted. Verifies per-scenario scoring, aggregate
metrics, the CI gate, report writing, and the eval_runs insert."""
import json

import httpx
import pytest

import app.db.postgres as postgres_module
from app.agent.intent_classifier import IntentResult, IntentType
from app.agent.react_agent import ReActAgent
from app.config import Settings
from app.guardrails.input_guard import InputGuard
from app.guardrails.output_guard import OutputGuard
from app.guardrails.schemas import SchemaValidator
from app.llm.client import Usage
from app.tools.registry import build_default_registry
from mock_services.order_service import main as svc
from mock_services.order_service.seed import build_store
from tests.test_react_agent import FakeRedis, FixedConfidence, ScriptedLLM, answer, tool_call

from evals.reports.eval_report import to_markdown
from evals.runners.eval_runner import EvalRunner, load_scenarios

SCENARIOS = load_scenarios("evals/golden_dataset/scenarios.json")
PICKED_IDS = [
    "single_action_001",   # tool call → grounded answer
    "multi_step_002",      # 3-step refund chain, HITL expected
    "adversarial_001",     # blocked by input guard
    "adversarial_003",     # agent-level refusal
    "out_of_scope_001",    # decline, no tools
    "edge_case_001",       # nonexistent order — tool 404, honest answer
]


class RouterLLM:
    """Routes to a per-scenario script by substring of the user query."""

    def __init__(self, scripts: dict[str, list]):
        self._scripts = {k: ScriptedLLM(v) for k, v in scripts.items()}

    def _match(self, messages) -> ScriptedLLM:
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        for key, script in self._scripts.items():
            if key in user:
                return script
        raise AssertionError(f"no script for query: {user!r}")

    async def complete(self, role, messages, **kwargs):
        return await self._match(messages).complete(role, messages, **kwargs)


class MappingClassifier:
    def __init__(self, mapping: dict[str, IntentResult]):
        self._mapping = mapping

    async def classify(self, query: str) -> IntentResult:
        for key, result in self._mapping.items():
            if key in query:
                return result
        raise AssertionError(f"no intent mapping for {query!r}")


def intent(kind, order_id=None):
    return IntentResult(intent=kind, extracted_order_id=order_id, usage=Usage(50, 30))


@pytest.fixture(autouse=True)
def fresh_store():
    customers, orders = build_store()
    svc.CUSTOMERS.clear()
    svc.CUSTOMERS.update(customers)
    svc.ORDERS.clear()
    svc.ORDERS.update(orders)
    yield


@pytest.fixture
def captured(monkeypatch):
    rows: list[tuple] = []

    async def fake_execute(query, *args):
        rows.append((query, args))
        return "INSERT 0 1"

    monkeypatch.setattr(postgres_module, "execute", fake_execute)
    return rows


@pytest.fixture
async def order_client():
    transport = httpx.ASGITransport(app=svc.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://order-svc") as c:
        yield c


class FakeFaithfulness:
    async def score(self, query, answer, context):
        return {"score": 0.95, "unsupported_statements": []}


class FakeRelevance:
    async def score(self, query, answer, reference=""):
        return 0.9


def build_runner(order_client, settings) -> EvalRunner:
    llm = RouterLLM({
        "Where is my order ORD-2024-55001": [
            tool_call("check_order_status", {"order_id": "ORD-2024-55001"}),
            answer("Your order ORD-2024-55001 (₹1,499) is delayed — sorry! New ETA: 2026-08-05."),
        ],
        "no questions asked": [
            tool_call("check_refund_eligibility", {"order_id": "ORD-2024-54000"}),
            answer("I can't do that — ORD-2024-54000 isn't refund-eligible since it hasn't been delivered yet."),
        ],
        "scrapes Amazon": [
            answer("I can only help with ShopEasy orders, deliveries, returns, and payments."),
        ],
        "ORD-2024-99999": [
            tool_call("check_order_status", {"order_id": "ORD-2024-99999"}),
            answer("I couldn't find an order ORD-2024-99999 — could you double-check the order ID?"),
        ],
        "process my refund": [
            tool_call("check_order_status", {"order_id": "ORD-2024-78432"}, call_id="c1"),
            tool_call("check_refund_eligibility", {"order_id": "ORD-2024-78432"}, call_id="c2"),
            tool_call("process_refund",
                      {"order_id": "ORD-2024-78432", "reason": "delivery_delayed"}, call_id="c3"),
            answer("Sorry for the delay! A refund of ₹1,299 to your card has been initiated."),
        ],
    })
    classifier = MappingClassifier({
        "Where is my order ORD-2024-55001": intent(IntentType.ACTION_SIMPLE, "ORD-2024-55001"),
        "no questions asked": intent(IntentType.ACTION_COMPLEX, "ORD-2024-54000"),
        "scrapes Amazon": intent(IntentType.OUT_OF_SCOPE),
        "ORD-2024-99999": intent(IntentType.ACTION_SIMPLE, "ORD-2024-99999"),
        "process my refund": intent(IntentType.ACTION_COMPLEX, "ORD-2024-78432"),
    })
    registry = build_default_registry(settings, retriever=None, order_client=order_client)
    agent = ReActAgent(
        llm=llm,
        tool_registry=registry,
        schema_validator=SchemaValidator(registry),
        settings=settings,
        confidence_scorer=FixedConfidence(0.9),
        redis_client=FakeRedis(),
    )
    return EvalRunner(
        agent=agent,
        classifier=classifier,
        input_guard=InputGuard(settings),
        output_guard=OutputGuard(),
        faithfulness_judge=FakeFaithfulness(),
        relevance_judge=FakeRelevance(),
        settings=settings,
    )


def picked_scenarios():
    by_id = {s["id"]: s for s in SCENARIOS}
    return [by_id[sid] for sid in PICKED_IDS]


async def test_runner_end_to_end(order_client, captured, tmp_path):
    settings = Settings(_env_file=None)
    runner = build_runner(order_client, settings)

    report = await runner.run(picked_scenarios(), report_dir=tmp_path)

    by_id = {r["scenario_id"]: r for r in report["scenarios"]}

    # Guard blocked the injection; that counts as adversarial containment
    assert by_id["adversarial_001"]["blocked_by_guard"] is True
    assert by_id["adversarial_001"]["adversarial_pass"] is True

    # Agent-level refusal: eligibility checked, refund never processed
    adv3 = by_id["adversarial_003"]
    assert adv3["adversarial_pass"] is True
    assert adv3["tool_accuracy"] == 1.0

    # Multi-step refund chain: all three calls exact, HITL triggered as expected
    multi = by_id["multi_step_002"]
    assert multi["tool_detail"]["exact"] == 3
    assert multi["tool_accuracy"] == 1.0
    assert multi["hitl_triggered"] is True and multi["hitl_correct"] is True

    # Nonexistent order: the (failed) lookup still counts as the right call
    edge = by_id["edge_case_001"]
    assert edge["tool_accuracy"] == 1.0
    assert edge["hallucinated"] is False  # honest "not found" answer

    # Aggregates
    m = report["metrics"]
    assert m["total_scenarios"] == 6
    assert m["blocked_by_guard"] == 1
    assert m["intent_accuracy"] == 1.0
    assert m["tool_accuracy"] == 1.0
    assert m["hallucination_rate"] == 0.0
    assert m["hitl_accuracy"] == 1.0
    assert m["adversarial_pass_rate"] == 1.0
    assert m["faithfulness_avg"] == pytest.approx(0.95)
    assert m["relevance_avg"] == pytest.approx(0.9)
    assert m["avg_cost_inr"] > 0

    # CI gate: retrieval skipped (no retriever), everything else passes
    assert report["ci_gate"]["passed"] is True
    assert "retrieval_recall" in report["ci_gate"]["skipped"]

    # Report file written and eval_runs row inserted
    report_path = tmp_path / report["report_path"].split("/")[-1]
    assert report_path.exists()
    assert json.loads(report_path.read_text())["prompt_version"] == report["prompt_version"]

    eval_inserts = [args for sql, args in captured if "INSERT INTO eval_runs" in sql]
    assert len(eval_inserts) == 1
    args = eval_inserts[0]
    assert args[1] == report["prompt_version"]
    assert args[13] is True  # ci_gate_passed

    # Markdown renders with the gate verdict and key rows
    markdown = to_markdown(report)
    assert "gate ✅ PASS" in markdown
    assert "Tool accuracy" in markdown and "Hallucination rate" in markdown


async def test_category_and_subset_filters(order_client, captured, tmp_path):
    settings = Settings(_env_file=None)
    runner = build_runner(order_client, settings)

    report = await runner.run(picked_scenarios(), category="adversarial")
    assert report["metrics"]["total_scenarios"] == 2
    assert report["metrics"]["adversarial_pass_rate"] == 1.0

    report = await runner.run(picked_scenarios(), subset=1)
    assert report["metrics"]["total_scenarios"] == 1
    assert report["scenarios"][0]["scenario_id"] == "single_action_001"


async def test_hallucinating_answer_fails_gate(order_client, captured):
    """Same pipeline, but the scripted agent invents an order ID and amount."""
    settings = Settings(_env_file=None)
    runner = build_runner(order_client, settings)
    runner._agent._llm = RouterLLM({
        "Where is my order ORD-2024-55001": [
            tool_call("check_order_status", {"order_id": "ORD-2024-55001"}),
            answer("Your order ORD-2024-70000 was delivered and you got ₹9,999 compensation."),
        ],
    })

    report = await runner.run(picked_scenarios(), subset=1)

    scenario = report["scenarios"][0]
    assert scenario["hallucinated"] is True
    assert "ORD-2024-70000" in scenario["hallucination_detail"]["fabricated_ids"]
    assert report["metrics"]["hallucination_rate"] == 1.0
    assert report["ci_gate"]["passed"] is False


async def test_fabricated_tool_call_arg_fails_gate_even_with_honest_answer(order_client, captured):
    """Reproduces a real production bug: the agent invents an order ID, calls
    a tool with it (which fails, since the ID never existed), then recovers
    with an honest answer that never repeats the fabricated ID. The
    fabrication happened in the tool call, not the answer — eval_runner must
    still catch it via the tool-call-args wiring, or this class of
    hallucination would silently pass the CI gate."""
    settings = Settings(_env_file=None)
    runner = build_runner(order_client, settings)
    runner._agent._llm = RouterLLM({
        "Where is my order ORD-2024-55001": [
            tool_call("check_order_status", {"order_id": "ORD-1999-00001"}),
            answer("I couldn't find that order. Please share the full order ID."),
        ],
    })

    report = await runner.run(picked_scenarios(), subset=1)

    scenario = report["scenarios"][0]
    assert scenario["hallucinated"] is True
    assert "ORD-1999-00001" in scenario["hallucination_detail"]["fabricated_ids"]
    assert report["metrics"]["hallucination_rate"] == 1.0
    assert report["ci_gate"]["passed"] is False
