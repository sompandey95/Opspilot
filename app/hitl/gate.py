"""Risk-matrix HITL gate — replaces the Phase-4 AutoApproveHITLGate stub.

Matrix (effective risk → behaviour):
    NONE    auto-approve
    LOW     auto-approve + audit
    MEDIUM  auto-approve + audit + Slack notification
    HIGH    block: queue a pending request, notify Slack, wait for a human

Overrides applied before the matrix:
- process_refund with amount ≤ HITL_REFUND_AUTO_APPROVE_LIMIT is downgraded to
  MEDIUM (auto + notify); above the limit — or with no explicit amount — it
  stays HIGH.
- confidence < CONFIDENCE_THRESHOLD (when already scored on the trace) forces
  HIGH regardless of tool risk.
- escalate_to_manager is auto-approved despite being HIGH: executing it *is*
  the act of queueing a human request — gating it would deadlock its purpose.

Every decision (auto or human) lands in hitl_audit_log: the gate audits
automatic decisions and timeouts; HITLQueue.decide audits human ones.
"""
from __future__ import annotations

import logging
import time

from app.agent.react_agent import ApprovalDecision, ApprovalStatus
from app.config import Settings
from app.hitl.notifier import SlackNotifier
from app.hitl.queue import HITLQueue, record_audit
from app.observability.trace import Trace
from app.tools.base import RiskLevel, Tool

logger = logging.getLogger(__name__)

UNAVAILABLE_REASON = "approval system unavailable — high-risk action refused"


class HITLApprovalGate:
    def __init__(
        self,
        queue: HITLQueue,
        notifier: SlackNotifier,
        settings: Settings,
        audit=record_audit,
    ) -> None:
        self._queue = queue
        self._notifier = notifier
        self._settings = settings
        self._audit = audit

    async def request_approval(
        self, tool: Tool, args: dict, trace: Trace
    ) -> ApprovalDecision:
        started = time.perf_counter()
        risk, override_reason = self._effective_risk(tool, args, trace)

        if risk == RiskLevel.NONE:
            return ApprovalDecision(status=ApprovalStatus.APPROVED, reason="auto_approved_none")

        if risk in (RiskLevel.LOW, RiskLevel.MEDIUM):
            reason = f"auto_approved_{risk.value}"
            if override_reason:
                reason = f"{reason}_{override_reason}"
            decision = ApprovalDecision(status=ApprovalStatus.APPROVED, reason=reason)
            if risk == RiskLevel.MEDIUM:
                await self._notifier.notify_auto_approved(tool.name, args, risk.value, reason)
            await self._audit(
                trace_id=trace.trace_id,
                tool_name=tool.name,
                tool_args=args,
                risk_level=risk.value,
                decision="auto_approved",
                decided_by="system",
                decision_time_ms=self._elapsed_ms(started),
            )
            return decision

        return await self._block_for_approval(tool, args, trace, override_reason, started)

    # ------------------------------------------------------------------ #
    # HIGH path                                                            #
    # ------------------------------------------------------------------ #

    async def _block_for_approval(
        self,
        tool: Tool,
        args: dict,
        trace: Trace,
        override_reason: str | None,
        started: float,
    ) -> ApprovalDecision:
        reasoning = override_reason or f"{tool.risk_level.value}-risk tool requires approval"
        try:
            request_id = await self._queue.create(
                tool_name=tool.name,
                tool_args=args,
                risk_level=RiskLevel.HIGH.value,
                trace_id=trace.trace_id,
                agent_reasoning=reasoning,
            )
        except Exception as exc:
            # Fail closed: a high-risk action without a working approval queue
            # must not execute.
            logger.error("HITL queue unavailable for '%s': %s", tool.name, exc)
            await self._audit(
                trace_id=trace.trace_id,
                tool_name=tool.name,
                tool_args=args,
                risk_level=RiskLevel.HIGH.value,
                decision="rejected_unavailable",
                decided_by="system",
                decision_time_ms=self._elapsed_ms(started),
            )
            return ApprovalDecision(status=ApprovalStatus.REJECTED, reason=UNAVAILABLE_REASON)

        await self._notifier.notify_approval_request(request_id, tool.name, args, reasoning)

        timeout_seconds = self._settings.HITL_APPROVAL_TIMEOUT_MINUTES * 60.0
        try:
            request = await self._queue.wait_for_decision(request_id, timeout_seconds)
        except Exception as exc:
            logger.error("HITL wait failed for request %s: %s", request_id, exc)
            request = None

        if request is None:
            await self._audit(
                trace_id=trace.trace_id,
                tool_name=tool.name,
                tool_args=args,
                risk_level=RiskLevel.HIGH.value,
                decision="timeout",
                decided_by="system",
                decision_time_ms=self._elapsed_ms(started),
            )
            return ApprovalDecision(status=ApprovalStatus.TIMEOUT, reason=f"request {request_id} still pending")

        # Human decisions are audited by HITLQueue.decide — don't double-write.
        if request.status == "approved":
            return ApprovalDecision(
                status=ApprovalStatus.APPROVED,
                reason=f"approved by {request.decided_by or 'unknown'}",
            )
        return ApprovalDecision(
            status=ApprovalStatus.REJECTED,
            reason=request.decision_reason or "rejected by supervisor",
        )

    # ------------------------------------------------------------------ #
    # Effective risk (overrides)                                           #
    # ------------------------------------------------------------------ #

    def _effective_risk(
        self, tool: Tool, args: dict, trace: Trace
    ) -> tuple[RiskLevel, str | None]:
        if tool.name == "escalate_to_manager":
            return RiskLevel.MEDIUM, "escalation_queues_human_review_itself"

        risk = tool.risk_level
        override: str | None = None

        if tool.name == "process_refund":
            limit = self._settings.HITL_REFUND_AUTO_APPROVE_LIMIT
            amount = args.get("amount_inr")
            if amount is not None and float(amount) <= limit:
                risk = RiskLevel.MEDIUM
                override = f"refund_within_auto_approve_limit_{limit}"
            else:
                risk = RiskLevel.HIGH
                override = (
                    f"refund amount exceeds auto-approve limit ₹{limit}"
                    if amount is not None
                    else "refund without explicit amount requires approval"
                )

        if (
            trace.confidence is not None
            and trace.confidence < self._settings.CONFIDENCE_THRESHOLD
        ):
            return RiskLevel.HIGH, f"low_confidence_{trace.confidence}"

        return risk, override

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)
