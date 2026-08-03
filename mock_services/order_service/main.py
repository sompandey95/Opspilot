"""Mock ShopEasy order/customer microservice (port 8001).

Deterministic in-memory store seeded at import time; a restart resets all
mutations. Dates are computed against seed.ANCHOR so behaviour is reproducible.
"""
from __future__ import annotations

from datetime import timedelta

from fastapi import FastAPI, HTTPException, Query

from mock_services.order_service.models import (
    CancelRequest,
    Customer,
    Order,
    OrderStatus,
    Refund,
    RefundEligibility,
    RefundRequest,
    RefundStatus,
)
from mock_services.order_service.seed import ANCHOR, build_store

RETURN_WINDOW_DAYS = 10

app = FastAPI(title="ShopEasy Mock Order Service", version="0.1.0")

CUSTOMERS, ORDERS = build_store()


def _get_order(order_id: str) -> Order:
    order = ORDERS.get(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    return order


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy", "orders": len(ORDERS), "customers": len(CUSTOMERS)}


@app.get("/orders/{order_id}", response_model=Order)
async def get_order(order_id: str) -> Order:
    return _get_order(order_id)


@app.get("/orders/{order_id}/eta")
async def get_delivery_eta(order_id: str) -> dict:
    order = _get_order(order_id)
    return {
        "order_id": order.order_id,
        "status": order.status,
        "delivery_eta": order.delivery_eta,
        "delivered_on": order.delivered_on,
        "delayed": order.status == OrderStatus.DELAYED,
    }


def _refund_eligibility(order: Order) -> RefundEligibility:
    if order.refund is not None or order.status == OrderStatus.REFUNDED:
        return RefundEligibility(
            order_id=order.order_id, eligible=False, amount_inr=0.0,
            reason="already_refunded",
        )
    if order.status == OrderStatus.CANCELLED:
        return RefundEligibility(
            order_id=order.order_id, eligible=False, amount_inr=0.0,
            reason="order_cancelled",
        )
    if order.status == OrderStatus.DELAYED:
        return RefundEligibility(
            order_id=order.order_id, eligible=True, amount_inr=order.amount_inr,
            reason="delivery_delayed",
        )
    if order.status == OrderStatus.DELIVERED:
        assert order.delivered_on is not None
        if ANCHOR - order.delivered_on <= timedelta(days=RETURN_WINDOW_DAYS):
            return RefundEligibility(
                order_id=order.order_id, eligible=True, amount_inr=order.amount_inr,
                reason="return_window",
            )
        return RefundEligibility(
            order_id=order.order_id, eligible=False, amount_inr=0.0,
            reason="return_window_closed",
        )
    return RefundEligibility(
        order_id=order.order_id, eligible=False, amount_inr=0.0,
        reason="not_yet_delivered",
    )


@app.get("/orders/{order_id}/refund-eligibility", response_model=RefundEligibility)
async def check_refund_eligibility(order_id: str) -> RefundEligibility:
    return _refund_eligibility(_get_order(order_id))


@app.post("/orders/{order_id}/refund", response_model=Refund)
async def process_refund(order_id: str, body: RefundRequest) -> Refund:
    order = _get_order(order_id)
    eligibility = _refund_eligibility(order)

    if not eligibility.eligible:
        status_code = 409 if eligibility.reason in ("already_refunded", "order_cancelled") else 400
        raise HTTPException(
            status_code=status_code,
            detail=f"Order {order_id} is not refund-eligible: {eligibility.reason}",
        )

    amount = body.amount_inr if body.amount_inr is not None else eligibility.amount_inr
    if amount <= 0 or amount > order.amount_inr:
        raise HTTPException(
            status_code=400,
            detail=f"Refund amount must be between 0 and {order.amount_inr}",
        )

    refund = Refund(
        refund_id=f"REF-{order_id.removeprefix('ORD-')}",
        order_id=order_id,
        amount_inr=amount,
        reason=body.reason,
        status=RefundStatus.PROCESSING,
        initiated_on=ANCHOR,
    )
    order.refund = refund
    order.status = OrderStatus.REFUNDED
    return refund


@app.post("/orders/{order_id}/cancel", response_model=Order)
async def cancel_order(order_id: str, body: CancelRequest) -> Order:
    order = _get_order(order_id)

    if order.status in (OrderStatus.DELIVERED, OrderStatus.REFUNDED):
        raise HTTPException(
            status_code=409,
            detail=f"Order {order_id} is {order.status.value} and can no longer be cancelled",
        )
    if order.status == OrderStatus.CANCELLED:
        raise HTTPException(status_code=409, detail=f"Order {order_id} is already cancelled")

    order.status = OrderStatus.CANCELLED
    # Prepaid orders get an automatic refund on cancellation
    if order.payment_method.value != "cod" and order.refund is None:
        order.refund = Refund(
            refund_id=f"REF-{order_id.removeprefix('ORD-')}",
            order_id=order_id,
            amount_inr=order.amount_inr,
            reason=f"cancelled: {body.reason}",
            status=RefundStatus.PROCESSING,
            initiated_on=ANCHOR,
        )
    return order


@app.get("/customers/search")
async def search_customer(
    email: str | None = Query(default=None),
    phone: str | None = Query(default=None),
    order_id: str | None = Query(default=None),
) -> dict:
    if not any([email, phone, order_id]):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: email, phone, order_id",
        )

    customer: Customer | None = None
    if order_id:
        order = ORDERS.get(order_id)
        if order:
            customer = CUSTOMERS.get(order.customer_id)
    elif email:
        customer = next((c for c in CUSTOMERS.values() if c.email == email.lower()), None)
    elif phone:
        normalised = phone.replace(" ", "").replace("-", "")
        customer = next(
            (c for c in CUSTOMERS.values() if c.phone.replace("-", "") == normalised),
            None,
        )

    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")

    customer_orders = [
        {"order_id": o.order_id, "status": o.status, "amount_inr": o.amount_inr}
        for o in ORDERS.values()
        if o.customer_id == customer.customer_id
    ]
    return {"customer": customer, "orders": customer_orders}
