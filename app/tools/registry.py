"""Tool registry — the agent discovers available tools here."""
from __future__ import annotations

import httpx

from app.config import Settings
from app.tools.base import Tool
from app.tools.customer_tool import SearchCustomerTool
from app.tools.escalate_tool import EscalateToManagerTool
from app.tools.jira_tool import CreateJiraTicketTool, UpdateJiraTicketTool
from app.tools.knowledge_tool import SearchKnowledgeTool
from app.tools.order_tool import (
    CancelOrderTool,
    CheckOrderStatusTool,
    CheckRefundEligibilityTool,
    GetDeliveryEtaTool,
    ProcessRefundTool,
)
from app.tools.slack_tool import SendSlackSummaryTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def get_tool_schemas(self) -> list[dict]:
        """OpenAI-format tool list to pass into chat completions."""
        return [tool.get_schema() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def build_default_registry(
    settings: Settings,
    retriever=None,
    order_client: httpx.AsyncClient | None = None,
    jira_client: httpx.AsyncClient | None = None,
    slack_client: httpx.AsyncClient | None = None,
    hitl_queue=None,
) -> ToolRegistry:
    """
    Register the full tool set. Unconfigured integrations (Jira, Slack) are
    still registered — their execute() returns a graceful error — so the tool
    schema surface presented to the LLM stays stable across environments.
    Optional clients exist for tests (httpx.ASGITransport against mock apps).
    """
    registry = ToolRegistry()

    registry.register(SearchKnowledgeTool(retriever))

    registry.register(CheckOrderStatusTool(settings, client=order_client))
    registry.register(GetDeliveryEtaTool(settings, client=order_client))
    registry.register(CheckRefundEligibilityTool(settings, client=order_client))
    registry.register(SearchCustomerTool(settings, client=order_client))
    registry.register(ProcessRefundTool(settings, client=order_client))
    registry.register(CancelOrderTool(settings, client=order_client))

    registry.register(CreateJiraTicketTool(settings, client=jira_client))
    registry.register(UpdateJiraTicketTool(settings, client=jira_client))
    registry.register(SendSlackSummaryTool(settings, client=slack_client))
    registry.register(EscalateToManagerTool(settings, client=slack_client, hitl_queue=hitl_queue))

    return registry
