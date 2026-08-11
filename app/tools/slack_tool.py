"""Slack notification tool (incoming webhook)."""
from __future__ import annotations

import httpx

from app.config import Settings
from app.tools.base import RiskLevel, Tool, ToolResult


class SendSlackSummaryTool(Tool):
    name = "send_slack_summary"
    description = (
        "Send a short summary message to the support team's Slack channel "
        "(e.g. after resolving a complex issue or spotting a pattern)."
    )
    risk_level = RiskLevel.LOW
    is_state_changing = True
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 5, "maxLength": 2000},
        },
        "required": ["text"],
    }

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._webhook_url = settings.SLACK_WEBHOOK_URL
        self._client = client

    async def execute(self, text: str) -> ToolResult:
        if not self._webhook_url:
            return ToolResult(success=False, error="Slack is not configured (SLACK_WEBHOOK_URL)")
        payload = {"text": text}
        try:
            if self._client is not None:
                response = await self._client.post(self._webhook_url, json=payload)
            else:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(self._webhook_url, json=payload)
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=f"Slack unreachable: {exc}")

        if response.status_code >= 400:
            return ToolResult(success=False, error=f"Slack {response.status_code}: {response.text[:200]}")
        return ToolResult(success=True, data={"sent": True})
