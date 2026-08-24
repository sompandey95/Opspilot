"""HITL API route tests against a fake queue: approve/reject transitions,
single-decision semantics (409), unknown IDs (404), and the pending list."""
import uuid

import httpx
import pytest
from fastapi import FastAPI

from app.api.hitl_routes import router
from app.hitl.queue import STATUS_PENDING, HITLRequest


class FakeQueue:
    def __init__(self):
        self.requests: dict[str, HITLRequest] = {}

    def add_pending(self) -> str:
        request_id = str(uuid.uuid4())
        self.requests[request_id] = HITLRequest(
            id=request_id,
            tool_name="process_refund",
            tool_args={"order_id": "ORD-2024-55001", "reason": "delivery_delayed"},
            risk_level="high",
            status=STATUS_PENDING,
        )
        return request_id

    async def get(self, request_id):
        return self.requests.get(request_id)

    async def decide(self, request_id, approved, decided_by, reason=None):
        request = self.requests.get(request_id)
        if request is None or request.status != STATUS_PENDING:
            return None
        request.status = "approved" if approved else "rejected"
        request.decided_by = decided_by
        request.decision_reason = reason
        return request

    async def list_pending(self):
        return [r for r in self.requests.values() if r.status == STATUS_PENDING]


@pytest.fixture
def queue():
    return FakeQueue()


@pytest.fixture
async def client(queue):
    app = FastAPI()
    app.include_router(router)
    app.state.hitl_queue = queue
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_approve_pending_request(client, queue):
    request_id = queue.add_pending()
    resp = await client.post(
        f"/api/v1/hitl/approve/{request_id}", json={"decided_by": "alice"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "approved"
    assert body["decided_by"] == "alice"


async def test_reject_requires_reason(client, queue):
    request_id = queue.add_pending()
    resp = await client.post(f"/api/v1/hitl/reject/{request_id}", json={})
    assert resp.status_code == 422  # reason is mandatory

    resp = await client.post(
        f"/api/v1/hitl/reject/{request_id}",
        json={"reason": "amount looks wrong", "decided_by": "bob"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"
    assert resp.json()["decision_reason"] == "amount looks wrong"


async def test_unknown_request_404(client):
    resp = await client.post(
        f"/api/v1/hitl/approve/{uuid.uuid4()}", json={"decided_by": "alice"}
    )
    assert resp.status_code == 404


async def test_double_decision_409(client, queue):
    request_id = queue.add_pending()
    await client.post(f"/api/v1/hitl/approve/{request_id}", json={})
    resp = await client.post(
        f"/api/v1/hitl/reject/{request_id}", json={"reason": "changed my mind"}
    )
    assert resp.status_code == 409
    assert "approved" in resp.json()["detail"]


async def test_pending_list(client, queue):
    first = queue.add_pending()
    second = queue.add_pending()
    await client.post(f"/api/v1/hitl/approve/{first}", json={})

    resp = await client.get("/api/v1/hitl/pending")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["pending"][0]["id"] == second


async def test_queue_unavailable_503(queue):
    app = FastAPI()
    app.include_router(router)
    app.state.hitl_queue = None
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(f"/api/v1/hitl/approve/{uuid.uuid4()}", json={})
    assert resp.status_code == 503
