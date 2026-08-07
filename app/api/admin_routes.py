"""Admin/read-only observability endpoints (Phase 6).

All GET, all backed by app.observability.metrics (plus the HITL queue for the
pending list). Protected by the API-key middleware like every non-exempt
route. `last` windows accept "24h" / "7d" style values (default 24h).
"""
from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from app.db.postgres import fetch_one
from app.observability import metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

_MAX_WINDOW_HOURS = 24 * 90


def _parse_window(last: str) -> float | None:
    """'24h' → 24.0, '7d' → 168.0; None on junk."""
    last = last.strip().lower()
    try:
        if last.endswith("h"):
            hours = float(last[:-1])
        elif last.endswith("d"):
            hours = float(last[:-1]) * 24
        else:
            hours = float(last)
    except ValueError:
        return None
    if not 0 < hours <= _MAX_WINDOW_HOURS:
        return None
    return hours


def _bad_window() -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid 'last' window — use e.g. 24h or 7d"},
    )


def _unavailable(what: str, exc: Exception) -> JSONResponse:
    logger.error("Admin %s query failed: %s", what, exc)
    return JSONResponse(status_code=503, content={"detail": f"{what} unavailable"})


@router.get("/traces")
async def traces(last: str = Query("24h"), limit: int = Query(100, ge=1, le=1000)):
    hours = _parse_window(last)
    if hours is None:
        return _bad_window()
    try:
        rows = await metrics.recent_traces(hours, limit)
    except Exception as exc:
        return _unavailable("traces", exc)
    return {"window_hours": hours, "count": len(rows), "traces": rows}


@router.get("/traces/{trace_id}")
async def trace_detail(trace_id: str):
    try:
        tid = uuid.UUID(trace_id)
    except ValueError:
        return JSONResponse(status_code=400, content={"detail": "Invalid trace ID"})
    try:
        row = await fetch_one("SELECT * FROM traces WHERE id = $1", tid)
    except Exception as exc:
        return _unavailable("trace", exc)
    if row is None:
        return JSONResponse(status_code=404, content={"detail": "Trace not found"})
    trace = dict(row)
    for key in ("steps", "guardrail_flags", "eval_scores"):
        if isinstance(trace.get(key), str):
            trace[key] = json.loads(trace[key])
    return trace


@router.get("/metrics/summary")
async def metrics_summary(last: str = Query("24h")):
    hours = _parse_window(last)
    if hours is None:
        return _bad_window()
    try:
        return {
            "window_hours": hours,
            "summary": await metrics.summary(hours),
            "intent_distribution": await metrics.intent_distribution(hours),
        }
    except Exception as exc:
        return _unavailable("metrics", exc)


@router.get("/metrics/cost-breakdown")
async def metrics_cost_breakdown(last: str = Query("24h")):
    hours = _parse_window(last)
    if hours is None:
        return _bad_window()
    try:
        return {"window_hours": hours, "by_model": await metrics.cost_breakdown(hours)}
    except Exception as exc:
        return _unavailable("cost breakdown", exc)


@router.get("/hitl/stats")
async def hitl_stats(last: str = Query("24h")):
    hours = _parse_window(last)
    if hours is None:
        return _bad_window()
    try:
        return {"window_hours": hours, **(await metrics.hitl_stats(hours))}
    except Exception as exc:
        return _unavailable("HITL stats", exc)


@router.get("/hitl/pending")
async def hitl_pending(request: Request):
    """Same data as /api/v1/hitl/pending — mirrored here so admin dashboards
    only need the /admin prefix."""
    queue = getattr(request.app.state, "hitl_queue", None)
    if queue is None:
        return JSONResponse(status_code=503, content={"detail": "HITL queue unavailable"})
    try:
        from dataclasses import asdict

        pending = await queue.list_pending()
    except Exception as exc:
        return _unavailable("HITL pending", exc)
    return {"pending": [asdict(r) for r in pending], "count": len(pending)}


@router.get("/evals/latest")
async def evals_latest():
    try:
        run = await metrics.latest_eval()
    except Exception as exc:
        return _unavailable("eval runs", exc)
    if run is None:
        return JSONResponse(status_code=404, content={"detail": "No eval runs yet"})
    return run


@router.get("/evals/trend")
async def evals_trend(versions: str = Query(..., description="e.g. v1,v2")):
    wanted = [v.strip() for v in versions.split(",") if v.strip()]
    if not wanted:
        return JSONResponse(status_code=400, content={"detail": "No versions given"})
    try:
        rows = await metrics.eval_trend(wanted)
    except Exception as exc:
        return _unavailable("eval trend", exc)
    return {"versions": wanted, "runs": rows}
