"""Tests for the mock order service: determinism, endpoints, state transitions."""
import httpx
import pytest

from mock_services.order_service import main as svc
from mock_services.order_service.seed import build_store


@pytest.fixture(autouse=True)
def fresh_store():
    """Re-seed the in-memory store so mutation tests don't leak between tests."""
    customers, orders = build_store()
    svc.CUSTOMERS.clear()
    svc.CUSTOMERS.update(customers)
    svc.ORDERS.clear()
    svc.ORDERS.update(orders)
    yield


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=svc.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def test_seed_is_deterministic():
    customers_a, orders_a = build_store()
    customers_b, orders_b = build_store()
    assert list(orders_a.keys()) == list(orders_b.keys())
    assert len(orders_a) == 500
    sample = next(iter(orders_a))
    assert orders_a[sample].amount_inr == orders_b[sample].amount_inr


def test_pinned_orders_present():
    _, orders = build_store()
    for oid in (
        "ORD-2024-55001", "ORD-2024-78432", "ORD-2024-51234",
        "ORD-2024-52000", "ORD-2024-53000", "ORD-2024-54000",
    ):
        assert oid in orders


async def test_get_order_and_404(client):
    resp = await client.get("/orders/ORD-2024-55001")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "delayed"
    assert body["amount_inr"] == 1499.00

    resp = await client.get("/orders/ORD-9999-00000")
    assert resp.status_code == 404


async def test_eta_flags_delay(client):
    resp = await client.get("/orders/ORD-2024-55001/eta")
    assert resp.status_code == 200
    assert resp.json()["delayed"] is True


async def test_refund_eligibility_cases(client):
    delayed = (await client.get("/orders/ORD-2024-55001/refund-eligibility")).json()
    assert delayed == {
        "order_id": "ORD-2024-55001",
        "eligible": True,
        "amount_inr": 1499.00,
        "reason": "delivery_delayed",
    }

    in_window = (await client.get("/orders/ORD-2024-51234/refund-eligibility")).json()
    assert in_window["eligible"] is True and in_window["reason"] == "return_window"

    closed = (await client.get("/orders/ORD-2024-52000/refund-eligibility")).json()
    assert closed["eligible"] is False and closed["reason"] == "return_window_closed"

    refunded = (await client.get("/orders/ORD-2024-53000/refund-eligibility")).json()
    assert refunded["eligible"] is False

    in_transit = (await client.get("/orders/ORD-2024-54000/refund-eligibility")).json()
    assert in_transit["eligible"] is False and in_transit["reason"] == "not_yet_delivered"


async def test_refund_flow_and_double_refund_conflict(client):
    resp = await client.post(
        "/orders/ORD-2024-55001/refund",
        json={"reason": "delivery_delayed"},
    )
    assert resp.status_code == 200
    refund = resp.json()
    assert refund["amount_inr"] == 1499.00
    assert refund["status"] == "processing"

    # Order is now refunded; second attempt conflicts
    resp = await client.post(
        "/orders/ORD-2024-55001/refund",
        json={"reason": "delivery_delayed"},
    )
    assert resp.status_code == 409


async def test_refund_rejects_excess_amount(client):
    resp = await client.post(
        "/orders/ORD-2024-78432/refund",
        json={"amount_inr": 99999.0, "reason": "delivery_delayed"},
    )
    assert resp.status_code == 400


async def test_cancel_rules(client):
    # In-transit COD order can be cancelled; COD gets no auto-refund
    resp = await client.post("/orders/ORD-2024-54000/cancel", json={"reason": "changed my mind"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "cancelled"
    assert body["refund"] is None

    # Delivered order cannot be cancelled
    resp = await client.post("/orders/ORD-2024-51234/cancel", json={"reason": "too late"})
    assert resp.status_code == 409

    # Prepaid delayed order gets auto-refund on cancel
    resp = await client.post("/orders/ORD-2024-78432/cancel", json={"reason": "taking too long"})
    assert resp.status_code == 200
    assert resp.json()["refund"]["amount_inr"] == 1299.00


async def test_customer_search(client):
    by_order = await client.get("/customers/search", params={"order_id": "ORD-2024-55001"})
    assert by_order.status_code == 200
    payload = by_order.json()
    assert any(o["order_id"] == "ORD-2024-55001" for o in payload["orders"])

    email = payload["customer"]["email"]
    by_email = await client.get("/customers/search", params={"email": email})
    assert by_email.status_code == 200
    assert by_email.json()["customer"]["email"] == email

    no_params = await client.get("/customers/search")
    assert no_params.status_code == 400

    missing = await client.get("/customers/search", params={"email": "nobody@nowhere.dev"})
    assert missing.status_code == 404
