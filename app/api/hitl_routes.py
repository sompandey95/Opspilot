"""Human-facing HITL endpoints: approve/reject pending requests, list queue.

Decisions are single-transition: a request can only leave 'pending' once
(HITLQueue.decide guards with `WHERE status = 'pending'`), so a double-approve
or approve-after-reject returns 409.
"""
from __future__ import annotations

import logging
from dataclasses import asdict

from pydantic import BaseModel, Field

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/hitl", tags=["hitl"])


class ApproveRequest(BaseModel):
    decided_by: str = Field(default="supervisor", max_length=64)


class RejectRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=500)
    decided_by: str = Field(default="supervisor", max_length=64)


def _get_queue(request: Request):
    return getattr(request.app.state, "hitl_queue", None)


async def _decide(request: Request, request_id: str, approved: bool,
                  decided_by: str, reason: str | None):
    queue = _get_queue(request)
    if queue is None:
        return JSONResponse(status_code=503, content={"detail": "HITL queue unavailable"})

    try:
        decided = await queue.decide(request_id, approved=approved,
                                     decided_by=decided_by, reason=reason)
        if decided is None:
            existing = await queue.get(request_id)
    except Exception as exc:
        logger.error("HITL decision failed for %s: %s", request_id, exc)
        return JSONResponse(status_code=503, content={"detail": "HITL queue unavailable"})

    if decided is None:
        if existing is None:
            return JSONResponse(status_code=404, content={"detail": "Request not found"})
        return JSONResponse(
            status_code=409,
            content={"detail": f"Request already decided: {existing.status}"},
        )
    return asdict(decided)


@router.post("/approve/{request_id}")
async def approve(request_id: str, body: ApproveRequest, request: Request):
    return await _decide(request, request_id, approved=True,
                         decided_by=body.decided_by, reason=None)


@router.post("/reject/{request_id}")
async def reject(request_id: str, body: RejectRequest, request: Request):
    return await _decide(request, request_id, approved=False,
                         decided_by=body.decided_by, reason=body.reason)


@router.get("/pending")
async def pending(request: Request):
    queue = _get_queue(request)
    if queue is None:
        return JSONResponse(status_code=503, content={"detail": "HITL queue unavailable"})
    try:
        requests = await queue.list_pending()
    except Exception as exc:
        logger.error("HITL pending list failed: %s", exc)
        return JSONResponse(status_code=503, content={"detail": "HITL queue unavailable"})
    return {"pending": [asdict(r) for r in requests], "count": len(requests)}
