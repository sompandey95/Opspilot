"""Customer lookup tool backed by the mock order service."""
from __future__ import annotations

from app.tools.base import RiskLevel, ToolResult
from app.tools.order_tool import OrderServiceTool


class SearchCustomerTool(OrderServiceTool):
    name = "search_customer"
    description = (
        "Find a customer (and their orders) by email, phone, or one of their "
        "order IDs. Provide at least one identifier."
    )
    risk_level = RiskLevel.LOW
    is_state_changing = False
    parameters = {
        "type": "object",
        "properties": {
            "email": {"type": "string", "format": "email"},
            "phone": {"type": "string", "description": "Phone number, e.g. +91-9876543210"},
            "order_id": {"type": "string", "pattern": "^ORD-\\d{4}-\\d{4,6}$"},
        },
        "anyOf": [
            {"required": ["email"]},
            {"required": ["phone"]},
            {"required": ["order_id"]},
        ],
    }

    async def execute(
        self,
        email: str | None = None,
        phone: str | None = None,
        order_id: str | None = None,
    ) -> ToolResult:
        params = {
            k: v
            for k, v in {"email": email, "phone": phone, "order_id": order_id}.items()
            if v
        }
        if not params:
            return ToolResult(success=False, error="Provide at least one of: email, phone, order_id")
        return await self._request("GET", "/customers/search", params=params)
