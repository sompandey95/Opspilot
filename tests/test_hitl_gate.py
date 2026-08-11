"""HITL gate tests — every risk-matrix row, the ₹500 refund boundary, the
low-confidence override, the escalation special case, and fail-closed behaviour
when the queue is down. Queue/notifier/audit are fakes; the real HITLQueue's
polling loop is exercised with a stubbed get()."""
import asyncio
import uuid

import pytest

from app.agent.react_agent import ApprovalStatus
from app.config import Settings
from app.hitl.gate import UNAVAILABLE_REASON, HITLApprovalGate
from app.hitl.queue import HITLQueue, HITLRequest
from app.observability.trace import Trace
from app.tools.registry import build_default_registry


# --------------------------------------------------------------------- #
# Fakes                                                                   #
# --------------------------------------------------------------------- #

class FakeQueue:
    def __init__(self, decision: HITLRequest | None = None, broken: bool = False):
        self.decision = decision
        self.broken = broken
        self.created: list[dict] = []

    async def create(self, **kwargs) -> str:
        if self.broken:
            raise ConnectionError("db down")
        self.created.append(kwargs)
        return str(uuid.uuid4())

    async def wait_for_decision(self, request_id, timeout_seconds):
        return self.decision


class FakeNotifier:
    def __init__(self):
        self.approval_requests: list[str] = []
        self.auto_approvals: list[str] = []

    async def notify_approval_request(self, request_id, tool_name, tool_args, agent_reasoning=None):
        self.approval_requests.append(tool_name)

    async def notify_auto_approved(self, tool_name, tool_args, risk_level, reason):
        self.auto_approvals.append(tool_name)


class AuditRecorder:
    def __init__(self):
        self.records: list[dict] = []

    async def __call__(self, **kwargs):
        self.records.append(kwargs)


def decided(status: str, decided_by="alice", reason=None) -> HITLRequest:
    return HITLRequest(
        id=str(uuid.uuid4()), tool_name="process_refund", tool_args={},
        risk_level="high", status=status, decided_by=decided_by,
        decision_reason=reason,
    )


# --------------------------------------------------------------------- #
# Fixtures                                                                #
# --------------------------------------------------------------------- #

@pytest.fixture
def settings():
    return Settings(_env_file=None, HITL_APPROVAL_TIMEOUT_MINUTES=1)


@pytest.fixture
def registry(settings):
    return build_default_registry(settings)


@pytest.fixture
def trace():
    return Trace(query="test query")


def make_gate(settings, queue=None, notifier=None, audit=None):
    queue = queue if queue is not None else FakeQueue()
    notifier = notifier or FakeNotifier()
    audit = audit or AuditRecorder()
    gate = HITLApprovalGate(queue, notifier, settings, audit=audit)
    return gate, queue, notifier, audit


# --------------------------------------------------------------------- #
# Matrix rows                                                             #
# --------------------------------------------------------------------- #

async def test_low_risk_auto_approved_with_audit_no_notify(settings, registry, trace):
    gate, queue, notifier, audit = make_gate(settings)
    decision = await gate.request_approval(
        registry.get("check_order_status"), {"order_id": "ORD-2024-55001"}, trace
    )
    assert decision.status == ApprovalStatus.APPROVED
    assert queue.created == []
    assert notifier.auto_approvals == [] and notifier.approval_requests == []
    assert len(audit.records) == 1
    assert audit.records[0]["decision"] == "auto_approved"
    assert audit.records[0]["decided_by"] == "system"


async def test_medium_risk_auto_approved_with_audit_and_notify(settings, registry, trace):
    gate, queue, notifier, audit = make_gate(settings)
    decision = await gate.request_approval(
        registry.get("create_jira_ticket"),
        {"summary": "Delayed order", "description": "..."},
        trace,
    )
    assert decision.status == ApprovalStatus.APPROVED
    assert queue.created == []
    assert notifier.auto_approvals == ["create_jira_ticket"]
    assert audit.records[0]["decision"] == "auto_approved"


async def test_high_risk_blocks_and_returns_human_approval(settings, registry, trace):
    gate, queue, notifier, _ = make_gate(
        settings, queue=FakeQueue(decision=decided("approved", decided_by="alice"))
    )
    decision = await gate.request_approval(
        registry.get("cancel_order"),
        {"order_id": "ORD-2024-54000", "reason": "changed my mind"},
        trace,
    )
    assert decision.status == ApprovalStatus.APPROVED
    assert "alice" in decision.reason
    assert len(queue.created) == 1
    assert queue.created[0]["tool_name"] == "cancel_order"
    assert notifier.approval_requests == ["cancel_order"]


async def test_high_risk_rejection_carries_reason(settings, registry, trace):
    gate, *_ = make_gate(
        settings,
        queue=FakeQueue(decision=decided("rejected", reason="customer is a fraud risk")),
    )
    decision = await gate.request_approval(
        registry.get("cancel_order"),
        {"order_id": "ORD-2024-54000", "reason": "cancel"},
        trace,
    )
    assert decision.status == ApprovalStatus.REJECTED
    assert decision.reason == "customer is a fraud risk"


async def test_high_risk_timeout_leaves_request_pending(settings, registry, trace):
    gate, queue, _, audit = make_gate(settings, queue=FakeQueue(decision=None))
    decision = await gate.request_approval(
        registry.get("cancel_order"),
        {"order_id": "ORD-2024-54000", "reason": "cancel"},
        trace,
    )
    assert decision.status == ApprovalStatus.TIMEOUT
    assert audit.records[-1]["decision"] == "timeout"


async def test_human_decisions_not_double_audited_by_gate(settings, registry, trace):
    """The gate audits only auto decisions/timeouts; queue.decide audits human
    ones — an approved HIGH request must not produce a gate-side audit row."""
    gate, _, _, audit = make_gate(settings, queue=FakeQueue(decision=decided("approved")))
    await gate.request_approval(
        registry.get("cancel_order"), {"order_id": "ORD-2024-54000", "reason": "x"}, trace
    )
    assert audit.records == []


# --------------------------------------------------------------------- #
# Refund limit override (₹500 boundary)                                   #
# --------------------------------------------------------------------- #

async def test_refund_at_limit_auto_approved(settings, registry, trace):
    gate, queue, notifier, audit = make_gate(settings)
    decision = await gate.request_approval(
        registry.get("process_refund"),
        {"order_id": "ORD-2024-55001", "reason": "delivery_delayed", "amount_inr": 500.00},
        trace,
    )
    assert decision.status == ApprovalStatus.APPROVED
    assert queue.created == []  # never reached the blocking path
    assert notifier.auto_approvals == ["process_refund"]  # treated as MEDIUM
    assert audit.records[0]["decision"] == "auto_approved"


async def test_refund_just_over_limit_blocks(settings, registry, trace):
    gate, queue, notifier, _ = make_gate(settings, queue=FakeQueue(decision=decided("approved")))
    decision = await gate.request_approval(
        registry.get("process_refund"),
        {"order_id": "ORD-2024-55001", "reason": "delivery_delayed", "amount_inr": 500.01},
        trace,
    )
    assert len(queue.created) == 1
    assert notifier.approval_requests == ["process_refund"]
    assert decision.status == ApprovalStatus.APPROVED  # via the human, not auto


async def test_refund_without_explicit_amount_blocks(settings, registry, trace):
    """No amount ⇒ full eligible amount, which the gate can't bound ⇒ HIGH."""
    gate, queue, *_ = make_gate(settings, queue=FakeQueue(decision=decided("approved")))
    await gate.request_approval(
        registry.get("process_refund"),
        {"order_id": "ORD-2024-55001", "reason": "delivery_delayed"},
        trace,
    )
    assert len(queue.created) == 1


# --------------------------------------------------------------------- #
# Confidence override + escalation special case                           #
# --------------------------------------------------------------------- #

async def test_low_confidence_forces_high_even_for_small_refund(settings, registry, trace):
    trace.set_confidence(0.4)
    gate, queue, *_ = make_gate(settings, queue=FakeQueue(decision=decided("approved")))
    await gate.request_approval(
        registry.get("process_refund"),
        {"order_id": "ORD-2024-55001", "reason": "delivery_delayed", "amount_inr": 100},
        trace,
    )
    assert len(queue.created) == 1  # blocked despite being within the limit


async def test_escalation_tool_is_never_gated(settings, registry, trace):
    """escalate_to_manager queues human review itself — blocking it on human
    approval would deadlock the escalation."""
    gate, queue, notifier, audit = make_gate(settings)
    decision = await gate.request_approval(
        registry.get("escalate_to_manager"),
        {"reason": "customer_requested_human", "summary": "wants a human now"},
        trace,
    )
    assert decision.status == ApprovalStatus.APPROVED
    assert queue.created == []
    assert len(audit.records) == 1


# --------------------------------------------------------------------- #
# Fail closed                                                             #
# --------------------------------------------------------------------- #

async def test_queue_unavailable_rejects_high_risk_action(settings, registry, trace):
    gate, _, _, audit = make_gate(settings, queue=FakeQueue(broken=True))
    decision = await gate.request_approval(
        registry.get("cancel_order"),
        {"order_id": "ORD-2024-54000", "reason": "cancel"},
        trace,
    )
    assert decision.status == ApprovalStatus.REJECTED
    assert decision.reason == UNAVAILABLE_REASON
    assert audit.records[-1]["decision"] == "rejected_unavailable"


# --------------------------------------------------------------------- #
# Real queue polling loop (DB stubbed at get())                           #
# --------------------------------------------------------------------- #

async def test_wait_for_decision_polls_until_decided(settings):
    class PollingQueue(HITLQueue):
        def __init__(self, settings, responses):
            super().__init__(settings)
            self.responses = responses

        async def get(self, request_id):
            return self.responses.pop(0)

    fast = Settings(_env_file=None, HITL_POLL_INTERVAL_SECONDS=0.11)
    pending = decided("pending")
    approved = decided("approved")
    queue = PollingQueue(fast, [pending, pending, approved])

    result = await asyncio.wait_for(
        queue.wait_for_decision("some-id", timeout_seconds=5), timeout=3
    )
    assert result is approved


async def test_wait_for_decision_times_out(settings):
    class AlwaysPending(HITLQueue):
        async def get(self, request_id):
            return decided("pending")

    fast = Settings(_env_file=None, HITL_POLL_INTERVAL_SECONDS=0.11)
    queue = AlwaysPending(fast)
    result = await queue.wait_for_decision("some-id", timeout_seconds=0.3)
    assert result is None
