"""Order-system tools backed by the mock order service (HTTP).

Read-only tools are LOW risk; process_refund and cancel_order are HIGH risk,
state-changing, and will be intercepted by the HITL gate in Phase 5.
"""
from __future__ import annotations

import httpx

from app.config import Settings
from app.tools.base import RiskLevel, Tool, ToolResult

_ORDER_ID_SCHEMA = {
    "type": "string",
    "pattern": "^ORD-\\d{4}-\\d{4,6}$",
    "description": "ShopEasy order ID, e.g. ORD-2024-55001",
}


class OrderServiceTool(Tool):
    """Base for tools that call the order service over HTTP.

    An injected client (httpx.AsyncClient) is used when provided — tests pass
    an ASGITransport client bound to the mock app; production uses base_url.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._base_url = settings.ORDER_SERVICE_URL
        self._client = client

    async def _request(self, method: str, path: str, **kwargs) -> ToolResult:
        try:
            if self._client is not None:
                response = await self._client.request(method, path, **kwargs)
            else:
                async with httpx.AsyncClient(base_url=self._base_url, timeout=10.0) as client:
                    response = await client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            return ToolResult(success=False, error=f"Order service unreachable: {exc}")

        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            return ToolResult(success=False, error=f"{response.status_code}: {detail}")

        return ToolResult(success=True, data=response.json())


class CheckOrderStatusTool(OrderServiceTool):
    name = "check_order_status"
    description = (
        "Look up a ShopEasy order: status (processing/in_transit/delayed/"
        "delivered/cancelled/refunded), product, amount, payment method, dates."
    )
    risk_level = RiskLevel.LOW
    is_state_changing = False
    parameters = {
        "type": "object",
        "properties": {"order_id": _ORDER_ID_SCHEMA},
        "required": ["order_id"],
    }

    async def execute(self, order_id: str) -> ToolResult:
        return await self._request("GET", f"/orders/{order_id}")


class GetDeliveryEtaTool(OrderServiceTool):
    name = "get_delivery_eta"
    description = "Get the delivery ETA for an order and whether it is delayed."
    risk_level = RiskLevel.LOW
    is_state_changing = False
    parameters = {
        "type": "object",
        "properties": {"order_id": _ORDER_ID_SCHEMA},
        "required": ["order_id"],
    }

    async def execute(self, order_id: str) -> ToolResult:
        return await self._request("GET", f"/orders/{order_id}/eta")


class CheckRefundEligibilityTool(OrderServiceTool):
    name = "check_refund_eligibility"
    description = (
        "Check whether an order is eligible for a refund, the eligible amount "
        "in INR, and the reason (delivery_delayed, return_window, "
        "return_window_closed, already_refunded, not_yet_delivered)."
    )
    risk_level = RiskLevel.LOW
    is_state_changing = False
    parameters = {
        "type": "object",
        "properties": {"order_id": _ORDER_ID_SCHEMA},
        "required": ["order_id"],
    }

    async def execute(self, order_id: str) -> ToolResult:
        return await self._request("GET", f"/orders/{order_id}/refund-eligibility")


class ProcessRefundTool(OrderServiceTool):
    name = "process_refund"
    description = (
        "Process a refund for an eligible order. Requires human approval for "
        "amounts above the auto-approve limit. Always check eligibility first "
        "with check_refund_eligibility."
    )
    risk_level = RiskLevel.HIGH
    is_state_changing = True
    parameters = {
        "type": "object",
        "properties": {
            "order_id": _ORDER_ID_SCHEMA,
            "amount_inr": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Refund amount in INR; omit to refund the full eligible amount",
            },
            "reason": {
                "type": "string",
                "enum": [
                    "delivery_delayed",
                    "defective",
                    "wrong_item",
                    "not_as_described",
                    "change_of_mind",
                    "seller_cancellation",
                ],
            },
        },
        "required": ["order_id", "reason"],
    }

    async def execute(self, order_id: str, reason: str, amount_inr: float | None = None) -> ToolResult:
        return await self._request(
            "POST",
            f"/orders/{order_id}/refund",
            json={"amount_inr": amount_inr, "reason": reason},
        )


class CancelOrderTool(OrderServiceTool):
    name = "cancel_order"
    description = (
        "Cancel an order that has not yet been delivered. Prepaid orders are "
        "automatically refunded. Requires human approval."
    )
    risk_level = RiskLevel.HIGH
    is_state_changing = True
    parameters = {
        "type": "object",
        "properties": {
            "order_id": _ORDER_ID_SCHEMA,
            "reason": {"type": "string", "minLength": 3, "description": "Why the customer wants to cancel"},
        },
        "required": ["order_id", "reason"],
    }

    async def execute(self, order_id: str, reason: str) -> ToolResult:
        return await self._request("POST", f"/orders/{order_id}/cancel", json={"reason": reason})
