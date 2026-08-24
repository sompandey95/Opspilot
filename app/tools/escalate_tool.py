"""Escalation tool — hands the conversation to a human supervisor.

Escalations are tracked as pending rows in the HITL queue
(app/hitl/queue.py) so supervisors see them in GET /api/v1/hitl/pending, and
Slack is notified when configured. If the queue is unavailable the tool
degrades to Slack + logs only — an escalation must never fail outright.

Note: this tool is HIGH risk but the HITL gate deliberately auto-approves it —
executing it *is* the act of requesting human review; gating it would deadlock.
"""
from __future__ import annotations

import logging
import uuid

import httpx

from app.config import Settings
from app.tools.base import RiskLevel, Tool, ToolResult

logger = logging.getLogger(__name__)


class EscalateToManagerTool(Tool):
    name = "escalate_to_manager"
    description = (
        "Escalate the current conversation to a human supervisor. Use for "
        "explicit requests for a human, legal threats, safety issues, or when "
        "you cannot resolve the issue confidently."
    )
    risk_level = RiskLevel.HIGH
    is_state_changing = True
    parameters = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "enum": [
                    "customer_requested_human",
                    "legal_threat",
                    "safety_issue",
                    "low_confidence",
                    "policy_exception_needed",
                    "repeated_failure",
                ],
            },
            "summary": {
                "type": "string",
                "minLength": 10,
                "description": "Short summary of the issue and what was tried",
            },
            "customer_identifier": {
                "type": "string",
                "description": "Email, phone, or order ID if known",
            },
        },
        "required": ["reason", "summary"],
    }

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
        hitl_queue=None,
    ) -> None:
        self._webhook_url = settings.SLACK_WEBHOOK_URL
        self._client = client
        self._queue = hitl_queue

    async def execute(
        self,
        reason: str,
        summary: str,
        customer_identifier: str | None = None,
    ) -> ToolResult:
        escalation_id = f"ESC-{uuid.uuid4().hex[:8]}"
        queued_in_hitl = False

        if self._queue is not None:
            try:
                escalation_id = await self._queue.create(
                    tool_name=self.name,
                    tool_args={
                        "reason": reason,
                        "summary": summary,
                        "customer_identifier": customer_identifier,
                    },
                    risk_level=self.risk_level.value,
                    agent_reasoning=summary,
                )
                queued_in_hitl = True
            except Exception as exc:
                logger.error("HITL queue insert failed for escalation: %s", exc)

        logger.warning(
            "Escalation %s: reason=%s customer=%s summary=%s",
            escalation_id, reason, customer_identifier, summary,
        )

        if self._webhook_url:
            text = (
                f":rotating_light: *Escalation {escalation_id}* — {reason}\n"
                f"Customer: {customer_identifier or 'unknown'}\n{summary}"
            )
            try:
                if self._client is not None:
                    await self._client.post(self._webhook_url, json={"text": text})
                else:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        await client.post(self._webhook_url, json={"text": text})
            except httpx.HTTPError as exc:
                logger.error("Slack escalation notify failed: %s", exc)

        return ToolResult(
            success=True,
            data={
                "escalation_id": escalation_id,
                "reason": reason,
                "status": "queued",
                "in_hitl_queue": queued_in_hitl,
            },
        )
