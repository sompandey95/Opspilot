"""Slack notifications for HITL decisions (best-effort, never raises)."""
from __future__ import annotations

import json
import logging

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


class SlackNotifier:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._webhook_url = settings.SLACK_WEBHOOK_URL
        self._timeout_minutes = settings.HITL_APPROVAL_TIMEOUT_MINUTES
        self._client = client

    async def notify_approval_request(
        self,
        request_id: str,
        tool_name: str,
        tool_args: dict,
        agent_reasoning: str | None = None,
    ) -> None:
        text = (
            f":raised_hand: *Approval needed* — `{tool_name}` (request `{request_id}`)\n"
            f"Args: `{json.dumps(tool_args, default=str)}`\n"
            f"{('Context: ' + agent_reasoning) if agent_reasoning else ''}\n"
            f"Approve: `POST /api/v1/hitl/approve/{request_id}` · "
            f"Reject: `POST /api/v1/hitl/reject/{request_id}`\n"
            f"Auto-times-out for the customer after {self._timeout_minutes} min "
            f"(request stays open for a late decision)."
        )
        await self._post(text)

    async def notify_auto_approved(
        self, tool_name: str, tool_args: dict, risk_level: str, reason: str
    ) -> None:
        text = (
            f":white_check_mark: *Auto-approved* `{tool_name}` ({risk_level} risk, {reason})\n"
            f"Args: `{json.dumps(tool_args, default=str)}`"
        )
        await self._post(text)

    async def _post(self, text: str) -> None:
        if not self._webhook_url:
            return
        try:
            if self._client is not None:
                await self._client.post(self._webhook_url, json={"text": text})
            else:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(self._webhook_url, json={"text": text})
        except httpx.HTTPError as exc:
            logger.error("Slack HITL notification failed: %s", exc)
