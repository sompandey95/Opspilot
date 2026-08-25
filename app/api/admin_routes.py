"""Administrative observability and evaluation endpoints.

All GET, all backed by app.observability.metrics (plus the HITL queue for the
pending list). Protected by the API-key middleware like every non-exempt
route. `last` windows accept "24h" / "7d" style values (default 24h).
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import get_settings
from app.db.postgres import fetch_one
from app.observability import metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENARIOS_PATH = _REPO_ROOT / "evals" / "golden_dataset" / "scenarios.json"
_REPORT_DIR = _REPO_ROOT / "evals" / "reports" / "runs"

_MAX_WINDOW_HOURS = 24 * 90

_DEMO_ORDERS = (
    ("ORD-2024-55001", "Delayed UPI order — refund needs approval"),
    ("ORD-2024-78432", "Delayed card order — ₹1,299 approval walkthrough"),
    ("ORD-2024-51234", "Delivered inside the return window"),
    ("ORD-2024-52000", "Delivered — return window closed"),
    ("ORD-2024-53000", "Cancelled with a completed refund"),
    ("ORD-2024-54000", "In transit, COD — not refund eligible"),
)


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


class EvalRunRequest(BaseModel):
    subset: int | None = None
    category: str | None = None
    no_llm_judges: bool = False
    no_retrieval: bool = False


async def _run_eval_job(payload: EvalRunRequest) -> None:
    """Fired in the background by POST /evals/run. Hits real Azure OpenAI —
    result lands as a new row visible via GET /evals/latest and a report file
    under evals/reports/runs; there is no separate job-status store."""
    from evals.runners.eval_runner import load_scenarios
    from evals.runners.runner_factory import build_eval_runner

    settings = get_settings()
    try:
        runner = await build_eval_runner(
            settings,
            with_llm_judges=not payload.no_llm_judges,
            with_retrieval=not payload.no_retrieval,
        )
        scenarios = load_scenarios(_SCENARIOS_PATH)
        await runner.run(
            scenarios, subset=payload.subset, category=payload.category, report_dir=_REPORT_DIR
        )
    except Exception:
        logger.exception("Background eval run failed")


@router.post("/evals/run", status_code=202)
async def evals_run(payload: EvalRunRequest = EvalRunRequest()):
    settings = get_settings()
    if not (settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_ENDPOINT):
        return JSONResponse(
            status_code=503, content={"detail": "Azure OpenAI credentials not configured"}
        )
    asyncio.create_task(_run_eval_job(payload))
    return {
        "status": "started",
        "detail": "Running against the live agent — poll GET /api/v1/admin/evals/latest for results.",
    }


def _order_service_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=get_settings().ORDER_SERVICE_URL, timeout=5.0)


async def _fetch_demo_order(client: httpx.AsyncClient, order_id: str, label: str) -> dict:
    entry = {"order_id": order_id, "label": label, "available": False}
    try:
        order_resp = await client.get(f"/orders/{order_id}")
        if order_resp.status_code >= 400:
            return {**entry, "error": f"order service returned {order_resp.status_code}"}
        eligibility_resp = await client.get(f"/orders/{order_id}/refund-eligibility")
    except httpx.HTTPError as exc:
        return {**entry, "error": f"order service unreachable: {exc}"}

    order = order_resp.json()
    eligibility = eligibility_resp.json() if eligibility_resp.status_code < 400 else {}
    refund = order.get("refund") or {}
    return {
        **entry,
        "available": True,
        "product_name": order.get("product_name"),
        "status": order.get("status"),
        "amount_inr": order.get("amount_inr"),
        "payment_method": order.get("payment_method"),
        "refund_status": refund.get("status"),
        "refund_eligible": eligibility.get("eligible"),
        "refund_reason": eligibility.get("reason"),
        "eligible_amount_inr": eligibility.get("amount_inr"),
    }


@router.get("/demo-orders")
async def demo_orders():
    """Live state of the pinned mock-service fixtures the console demos use.

    Mutations live in the order service's memory, so a fixture stays refunded
    until that container restarts — this endpoint shows what is still usable.
    """
    try:
        async with _order_service_client() as client:
            orders = list(
                await asyncio.gather(
                    *(_fetch_demo_order(client, oid, label) for oid, label in _DEMO_ORDERS)
                )
            )
    except Exception as exc:
        return _unavailable("demo orders", exc)

    if not any(order["available"] for order in orders):
        return JSONResponse(
            status_code=503,
            content={"detail": "Order service unavailable — no demo fixtures could be read"},
        )
    return {"count": len(orders), "orders": orders}
