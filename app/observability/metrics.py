"""Aggregate metrics over the traces / hitl / eval tables.

Pure SQL readers used by the admin routes; callers handle exceptions (a dead
DB turns into the route's 503, not a crash here).
"""
from __future__ import annotations

from app.db.postgres import fetch_all, fetch_one

_SUMMARY_SQL = """
SELECT
    COUNT(*)                                                        AS total_traces,
    COALESCE(AVG(confidence), 0)                                    AS avg_confidence,
    COALESCE(AVG(total_latency_ms), 0)                              AS avg_latency_ms,
    COALESCE(percentile_cont(0.5) WITHIN GROUP (ORDER BY total_latency_ms), 0) AS p50_latency_ms,
    COALESCE(percentile_cont(0.99) WITHIN GROUP (ORDER BY total_latency_ms), 0) AS p99_latency_ms,
    COALESCE(SUM(total_tokens), 0)                                  AS total_tokens,
    COALESCE(SUM(cost_inr), 0)                                      AS total_cost_inr,
    COALESCE(AVG(CASE WHEN escalated THEN 1.0 ELSE 0.0 END), 0)     AS escalation_rate,
    COALESCE(AVG(CASE WHEN hitl_triggered THEN 1.0 ELSE 0.0 END), 0) AS hitl_rate
FROM traces
WHERE created_at > NOW() - ($1 * INTERVAL '1 hour')
"""

_COST_BREAKDOWN_SQL = """
SELECT
    COALESCE(model, 'unknown') AS model,
    COUNT(*)                   AS traces,
    COALESCE(SUM(input_tokens), 0)  AS input_tokens,
    COALESCE(SUM(output_tokens), 0) AS output_tokens,
    COALESCE(SUM(cost_inr), 0)      AS cost_inr
FROM traces
WHERE created_at > NOW() - ($1 * INTERVAL '1 hour')
GROUP BY model
ORDER BY cost_inr DESC
"""

_INTENT_DISTRIBUTION_SQL = """
SELECT COALESCE(intent, 'unknown') AS intent, COUNT(*) AS count
FROM traces
WHERE created_at > NOW() - ($1 * INTERVAL '1 hour')
GROUP BY intent
ORDER BY count DESC
"""

_HITL_STATS_SQL = """
SELECT
    decision,
    COUNT(*)                             AS count,
    COALESCE(AVG(decision_time_ms), 0)   AS avg_decision_time_ms
FROM hitl_audit_log
WHERE created_at > NOW() - ($1 * INTERVAL '1 hour')
GROUP BY decision
ORDER BY count DESC
"""

_HITL_PENDING_COUNT_SQL = "SELECT COUNT(*) AS count FROM hitl_pending WHERE status = 'pending'"

_RECENT_TRACES_SQL = """
SELECT id, session_id, query, intent, model, confidence, total_latency_ms,
       total_tokens, cost_inr, hitl_triggered, escalated, prompt_version,
       created_at
FROM traces
WHERE created_at > NOW() - ($1 * INTERVAL '1 hour')
ORDER BY created_at DESC
LIMIT $2
"""

_LATEST_EVAL_SQL = "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT 1"

_EVAL_TREND_SQL = """
SELECT DISTINCT ON (prompt_version)
    prompt_version, model, total_scenarios, retrieval_precision,
    retrieval_recall, retrieval_mrr, faithfulness_avg, hallucination_rate,
    tool_accuracy, relevance_avg, avg_latency_ms, avg_cost_inr,
    ci_gate_passed, created_at
FROM eval_runs
WHERE prompt_version = ANY($1)
ORDER BY prompt_version, created_at DESC
"""


async def summary(last_hours: float) -> dict:
    row = await fetch_one(_SUMMARY_SQL, last_hours)
    return dict(row) if row else {}


async def cost_breakdown(last_hours: float) -> list[dict]:
    return [dict(r) for r in await fetch_all(_COST_BREAKDOWN_SQL, last_hours)]


async def intent_distribution(last_hours: float) -> list[dict]:
    return [dict(r) for r in await fetch_all(_INTENT_DISTRIBUTION_SQL, last_hours)]


async def hitl_stats(last_hours: float) -> dict:
    decisions = [dict(r) for r in await fetch_all(_HITL_STATS_SQL, last_hours)]
    pending = await fetch_one(_HITL_PENDING_COUNT_SQL)
    return {
        "decisions": decisions,
        "pending_count": pending["count"] if pending else 0,
    }


async def recent_traces(last_hours: float, limit: int = 100) -> list[dict]:
    return [dict(r) for r in await fetch_all(_RECENT_TRACES_SQL, last_hours, limit)]


async def latest_eval() -> dict | None:
    row = await fetch_one(_LATEST_EVAL_SQL)
    return dict(row) if row else None


async def eval_trend(versions: list[str]) -> list[dict]:
    return [dict(r) for r in await fetch_all(_EVAL_TREND_SQL, versions)]
