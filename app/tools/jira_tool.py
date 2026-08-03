"""Jira Cloud tools (REST API v3, API-token basic auth)."""
from __future__ import annotations

import httpx

from app.config import Settings
from app.tools.base import RiskLevel, Tool, ToolResult


class _JiraTool(Tool):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client

    @property
    def _configured(self) -> bool:
        s = self._settings
        return bool(s.JIRA_BASE_URL and s.JIRA_EMAIL and s.JIRA_API_TOKEN and s.JIRA_PROJECT_KEY)

    async def _request(self, method: str, path: str, payload: dict) -> ToolResult:
        if not self._configured:
            return ToolResult(
                success=False,
                error="Jira is not configured (JIRA_BASE_URL/JIRA_EMAIL/JIRA_API_TOKEN/JIRA_PROJECT_KEY)",
            )
        auth = (self._settings.JIRA_EMAIL, self._settings.JIRA_API_TOKEN)
        try:
            if self._client is not None:
                response = await self._client.request(method, path, json=payload, auth=auth)
            else:
                async with httpx.AsyncClient(
                    base_url=self._settings.JIRA_BASE_URL, timeout=15.0
                ) as client:
                    response = await client.request(method, path, json=payload, auth=auth)
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=f"Jira unreachable: {exc}")

        if response.status_code >= 400:
            return ToolResult(success=False, error=f"Jira {response.status_code}: {response.text[:500]}")

        data = response.json() if response.content else {}
        return ToolResult(success=True, data=data)

    @staticmethod
    def _adf(text: str) -> dict:
        """Wrap plain text in the Atlassian Document Format Jira v3 requires."""
        return {
            "type": "doc",
            "version": 1,
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": text}]}],
        }


class CreateJiraTicketTool(_JiraTool):
    name = "create_jira_ticket"
    description = (
        "Create a Jira issue in the support project for problems that need "
        "engineering or operations follow-up (payment stuck, courier issue, "
        "seller dispute). Returns the issue key."
    )
    risk_level = RiskLevel.MEDIUM
    is_state_changing = True
    parameters = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "minLength": 5, "maxLength": 255},
            "description": {"type": "string", "minLength": 10},
            "issue_type": {"type": "string", "enum": ["Task", "Bug"], "default": "Task"},
        },
        "required": ["summary", "description"],
    }

    async def execute(self, summary: str, description: str, issue_type: str = "Task") -> ToolResult:
        payload = {
            "fields": {
                "project": {"key": self._settings.JIRA_PROJECT_KEY},
                "summary": summary,
                "description": self._adf(description),
                "issuetype": {"name": issue_type},
            }
        }
        return await self._request("POST", "/rest/api/3/issue", payload)


class UpdateJiraTicketTool(_JiraTool):
    name = "update_jira_ticket"
    description = "Add a comment to an existing Jira issue by its key (e.g. SUP-123)."
    risk_level = RiskLevel.MEDIUM
    is_state_changing = True
    parameters = {
        "type": "object",
        "properties": {
            "issue_key": {"type": "string", "pattern": "^[A-Z][A-Z0-9]+-\\d+$"},
            "comment": {"type": "string", "minLength": 3},
        },
        "required": ["issue_key", "comment"],
    }

    async def execute(self, issue_key: str, comment: str) -> ToolResult:
        payload = {"body": self._adf(comment)}
        return await self._request("POST", f"/rest/api/3/issue/{issue_key}/comment", payload)
