"""Admin route tests — metrics module monkeypatched (SQL readers are exercised
against real Postgres only in deployment), window parsing, error → 503."""
import asyncio
import uuid

import httpx
import pytest
from fastapi import FastAPI

from app.api import admin_routes
from app.api.admin_routes import _parse_window, router
from app.config import Settings
from mock_services.order_service import main as svc
from mock_services.order_service.seed import build_store


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(router)
    app.state.hitl_queue = None
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        c.app = app
        yield c


@pytest.fixture
def fresh_store():
    customers, orders = build_store()
    svc.CUSTOMERS.clear()
    svc.CUSTOMERS.update(customers)
    svc.ORDERS.clear()
    svc.ORDERS.update(orders)
    yield


def _order_service_asgi_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=svc.app), base_url="http://order-service"
    )


@pytest.fixture
def order_service(fresh_store, monkeypatch):
    monkeypatch.setattr(admin_routes, "_order_service_client", _order_service_asgi_client)


# --------------------------------------------------------------------- #
# Window parsing                                                          #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw,hours",
    [("24h", 24.0), ("7d", 168.0), ("1.5h", 1.5), ("48", 48.0), ("2D", 48.0)],
)
def test_parse_window_valid(raw, hours):
    assert _parse_window(raw) == hours


@pytest.mark.parametrize("raw", ["", "yesterday", "-4h", "0h", "99999d"])
def test_parse_window_invalid(raw):
    assert _parse_window(raw) is None


# --------------------------------------------------------------------- #
# Endpoints                                                               #
# --------------------------------------------------------------------- #

async def test_traces_list(client, monkeypatch):
    async def fake_recent(hours, limit):
        assert hours == 48.0 and limit == 5
        return [{"id": "t1", "intent": "faq"}]

    monkeypatch.setattr(admin_routes.metrics, "recent_traces", fake_recent)
    resp = await client.get("/api/v1/admin/traces?last=2d&limit=5")
    assert resp.status_code == 200
    assert resp.json()["count"] == 1


async def test_traces_bad_window_400(client):
    resp = await client.get("/api/v1/admin/traces?last=fortnight")
    assert resp.status_code == 400


async def test_trace_detail_invalid_id_400(client):
    resp = await client.get("/api/v1/admin/traces/not-a-uuid")
    assert resp.status_code == 400


async def test_trace_detail_not_found(client, monkeypatch):
    async def fake_fetch_one(sql, *args):
        return None

    monkeypatch.setattr(admin_routes, "fetch_one", fake_fetch_one)
    resp = await client.get(f"/api/v1/admin/traces/{uuid.uuid4()}")
    assert resp.status_code == 404


async def test_trace_detail_parses_json_columns(client, monkeypatch):
    tid = uuid.uuid4()

    async def fake_fetch_one(sql, *args):
        return {
            "id": str(tid),
            "steps": '[{"step": 0, "type": "llm"}]',
            "guardrail_flags": '["pii_pan"]',
            "eval_scores": None,
        }

    monkeypatch.setattr(admin_routes, "fetch_one", fake_fetch_one)
    resp = await client.get(f"/api/v1/admin/traces/{tid}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["steps"] == [{"step": 0, "type": "llm"}]
    assert body["guardrail_flags"] == ["pii_pan"]


async def test_metrics_summary(client, monkeypatch):
    async def fake_summary(hours):
        return {"total_traces": 10, "p99_latency_ms": 900}

    async def fake_intents(hours):
        return [{"intent": "faq", "count": 7}]

    monkeypatch.setattr(admin_routes.metrics, "summary", fake_summary)
    monkeypatch.setattr(admin_routes.metrics, "intent_distribution", fake_intents)
    resp = await client.get("/api/v1/admin/metrics/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"]["total_traces"] == 10
    assert body["intent_distribution"][0]["intent"] == "faq"


async def test_metrics_summary_db_down_503(client, monkeypatch):
    async def boom(hours):
        raise ConnectionError("db down")

    monkeypatch.setattr(admin_routes.metrics, "summary", boom)
    resp = await client.get("/api/v1/admin/metrics/summary")
    assert resp.status_code == 503


async def test_cost_breakdown(client, monkeypatch):
    async def fake_breakdown(hours):
        return [{"model": "gpt-5.4", "cost_inr": 12.5}]

    monkeypatch.setattr(admin_routes.metrics, "cost_breakdown", fake_breakdown)
    resp = await client.get("/api/v1/admin/metrics/cost-breakdown")
    assert resp.status_code == 200
    assert resp.json()["by_model"][0]["model"] == "gpt-5.4"


async def test_hitl_stats(client, monkeypatch):
    async def fake_stats(hours):
        return {"decisions": [{"decision": "approved", "count": 3}], "pending_count": 1}

    monkeypatch.setattr(admin_routes.metrics, "hitl_stats", fake_stats)
    resp = await client.get("/api/v1/admin/hitl/stats")
    assert resp.status_code == 200
    assert resp.json()["pending_count"] == 1


async def test_hitl_pending_503_without_queue(client):
    resp = await client.get("/api/v1/admin/hitl/pending")
    assert resp.status_code == 503


async def test_evals_latest_404_when_none(client, monkeypatch):
    async def fake_latest():
        return None

    monkeypatch.setattr(admin_routes.metrics, "latest_eval", fake_latest)
    resp = await client.get("/api/v1/admin/evals/latest")
    assert resp.status_code == 404


async def test_evals_trend_parses_versions(client, monkeypatch):
    async def fake_trend(versions):
        assert versions == ["v1", "v2"]
        return [{"prompt_version": "v1"}, {"prompt_version": "v2"}]

    monkeypatch.setattr(admin_routes.metrics, "eval_trend", fake_trend)
    resp = await client.get("/api/v1/admin/evals/trend?versions=v1,%20v2")
    assert resp.status_code == 200
    assert len(resp.json()["runs"]) == 2


async def test_evals_trend_requires_versions(client):
    resp = await client.get("/api/v1/admin/evals/trend?versions=,")
    assert resp.status_code == 400


async def test_evals_run_503_without_azure_creds(client, monkeypatch):
    monkeypatch.setattr(admin_routes, "get_settings", lambda: Settings(_env_file=None))
    resp = await client.post("/api/v1/admin/evals/run")
    assert resp.status_code == 503


async def test_evals_run_202_starts_background_job(client, monkeypatch):
    monkeypatch.setattr(
        admin_routes,
        "get_settings",
        lambda: Settings(
            _env_file=None, AZURE_OPENAI_API_KEY="k", AZURE_OPENAI_ENDPOINT="https://x"
        ),
    )
    calls = []

    async def fake_job(payload):
        calls.append(payload)

    monkeypatch.setattr(admin_routes, "_run_eval_job", fake_job)
    resp = await client.post(
        "/api/v1/admin/evals/run", json={"subset": 5, "category": "faq_en"}
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "started"

    await asyncio.sleep(0)  # let the fire-and-forget task run
    assert len(calls) == 1
    assert calls[0].subset == 5
    assert calls[0].category == "faq_en"


# --------------------------------------------------------------------- #
# Demo order fixtures                                                     #
# --------------------------------------------------------------------- #

async def test_demo_orders_reports_live_fixture_state(client, order_service):
    resp = await client.get("/api/v1/admin/demo-orders")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == len(admin_routes._DEMO_ORDERS)
    assert all(order["available"] for order in body["orders"])

    walkthrough = next(o for o in body["orders"] if o["order_id"] == "ORD-2024-78432")
    assert walkthrough["amount_inr"] == 1299.0
    assert walkthrough["status"] == "delayed"
    assert walkthrough["payment_method"] == "card"
    assert walkthrough["refund_eligible"] is True
    assert walkthrough["eligible_amount_inr"] == 1299.0


async def test_demo_orders_marks_closed_return_window_not_refundable(client, order_service):
    body = (await client.get("/api/v1/admin/demo-orders")).json()
    closed = next(o for o in body["orders"] if o["order_id"] == "ORD-2024-52000")
    assert closed["refund_eligible"] is False
    assert closed["refund_reason"] == "return_window_closed"


async def test_demo_orders_reflects_a_processed_refund(client, order_service):
    async with _order_service_asgi_client() as service:
        refund = await service.post(
            "/orders/ORD-2024-78432/refund", json={"reason": "delivery delayed"}
        )
    assert refund.status_code == 200

    body = (await client.get("/api/v1/admin/demo-orders")).json()
    spent = next(o for o in body["orders"] if o["order_id"] == "ORD-2024-78432")
    assert spent["status"] == "refunded"
    assert spent["refund_status"] == "processing"
    assert spent["refund_eligible"] is False
    assert spent["refund_reason"] == "already_refunded"


async def test_demo_orders_marks_unknown_fixture_unavailable(client, order_service, monkeypatch):
    monkeypatch.setattr(
        admin_routes,
        "_DEMO_ORDERS",
        (("ORD-2024-78432", "known"), ("ORD-2024-00000", "missing")),
    )
    body = (await client.get("/api/v1/admin/demo-orders")).json()
    by_id = {o["order_id"]: o for o in body["orders"]}
    assert by_id["ORD-2024-78432"]["available"] is True
    assert by_id["ORD-2024-00000"]["available"] is False
    assert "404" in by_id["ORD-2024-00000"]["error"]


async def test_demo_orders_503_when_order_service_unreachable(client, monkeypatch):
    def unreachable(request):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        admin_routes,
        "_order_service_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(unreachable), base_url="http://order-service"
        ),
    )
    resp = await client.get("/api/v1/admin/demo-orders")
    assert resp.status_code == 503
