import hashlib
import logging

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db.postgres import get_pool
from app.db.redis import get_redis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# Same order of magnitude as the HITL approval timeout — long enough to
# absorb an impatient customer resending the identical message while the
# first request is still in flight, short enough that asking again later
# still raises a fresh escalation.
_ESCALATION_DEDUP_TTL_SECONDS = 10 * 60

# Maps app.observability.trace.Trace.add_escalation() reason strings (set by
# react_agent.py) to the escalate_to_manager tool's fixed reason enum.
_ESCALATION_REASON_MAP = {
    "intent_escalate": "customer_requested_human",
    "low_confidence": "low_confidence",
    "max_steps_reached": "repeated_failure",
    "llm_error": "repeated_failure",
}


def _get_redis_or_none():
    try:
        return get_redis()
    except RuntimeError:
        return None


@router.get("/health", tags=["ops"])
async def health_check(request: Request) -> dict:
    postgres_ok = False
    redis_ok = False
    chroma_ok = False
    chroma_docs = 0

    try:
        await get_pool().fetchval("SELECT 1")
        postgres_ok = True
    except Exception:
        pass

    try:
        redis_ok = await get_redis().ping()
    except Exception:
        pass

    try:
        settings = get_settings()
        url = f"http://{settings.CHROMA_HOST}:{settings.CHROMA_PORT}/api/v2/heartbeat"
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
            chroma_ok = resp.status_code == 200
    except Exception:
        pass

    # Document count is read from the initialised retriever's vector store
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is not None:
        try:
            chroma_docs = retriever._vector_store.get_collection_count()
        except Exception:
            pass

    return {
        "status": "healthy",
        "postgres": postgres_ok,
        "redis": redis_ok,
        "chromadb": chroma_ok,
        "chromadb_docs": chroma_docs,
    }


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


@router.post("/chat", tags=["chat"])
async def chat(body: ChatRequest, request: Request) -> dict:
    """Main agent endpoint: input → intent → route → agent.run → response."""
    classifier = getattr(request.app.state, "intent_classifier", None)
    agent = getattr(request.app.state, "react_agent", None)
    if classifier is None or agent is None:
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "OpsPilot agent is unavailable — check Azure OpenAI "
                    "configuration and tool registry initialisation."
                )
            },
        )

    # Per-org token budget (org from header; anonymous traffic shares "default")
    org = request.headers.get("x-org-id", "default")
    budget = getattr(request.app.state, "token_budget", None)
    if budget is not None:
        status = await budget.check(org)
        if not status.allowed:
            return JSONResponse(status_code=429, content={"detail": status.reason})

    # Session history, compacted to the context budget
    history: list[dict] = []
    session_manager = getattr(request.app.state, "session_manager", None)
    context_window = getattr(request.app.state, "context_window", None)
    if body.session_id and session_manager is not None:
        history = await session_manager.get_history(body.session_id)
        if history and context_window is not None:
            history = await context_window.fit(history)

    intent = await classifier.classify(body.query)

    result = await agent.run(
        body.query, intent, session_id=body.session_id, history=history
    )

    # Input-guard flags (PII masking) were stashed by the middleware.
    guardrail_flags = list(getattr(request.state, "guardrail_flags", []) or [])

    response_text = result.response
    output_guard = getattr(request.app.state, "output_guard", None)
    if output_guard is not None:
        guarded = output_guard.check(response_text, result.trace, sentiment=intent.sentiment)
        response_text = guarded.response
        guardrail_flags.extend(guarded.flags)

    if guardrail_flags:
        result.trace.add_guardrail_flags(guardrail_flags)
        if response_text != result.response:
            result.trace.set_response(response_text)

    # Any escalation — explicit request, low confidence, max steps, or an LLM
    # error — must actually reach a human, not just tell the customer it will.
    # Skip only if the agent already executed escalate_to_manager itself this
    # turn (that already queued it and notified Slack; don't double-post).
    if result.escalated:
        already_queued = any(
            step.get("type") in ("tool_result", "tool_error")
            and step.get("tool") == "escalate_to_manager"
            for step in result.trace.steps
        )
        if not already_queued:
            registry = getattr(request.app.state, "tool_registry", None)
            escalate_tool = registry.get("escalate_to_manager") if registry else None
            if escalate_tool is not None:
                last_escalation = next(
                    (s for s in reversed(result.trace.steps) if s.get("type") == "escalation"),
                    None,
                )
                escalation_reason = (last_escalation or {}).get("reason")
                reason = _ESCALATION_REASON_MAP.get(escalation_reason, "low_confidence")

                # Called directly here (not through the agent's tool-call path),
                # so it bypasses react_agent's own idempotency wrapper — without
                # a dedup key, a customer resending the same message (or a slow
                # request being retried) would queue and Slack-notify a human
                # once per resend instead of once per underlying issue.
                dedup_key = "idempotent:escalate_to_manager:" + hashlib.sha256(
                    f"{body.session_id or ''}|{reason}|{body.query}".encode()
                ).hexdigest()
                redis = _get_redis_or_none()
                already_notified = False
                if redis is not None:
                    try:
                        already_notified = bool(await redis.get(dedup_key))
                    except Exception as exc:
                        logger.error("Escalation dedup check failed (%s) — notifying anyway", exc)

                if not already_notified:
                    try:
                        await escalate_tool.execute(
                            reason=reason,
                            summary=f"Auto-escalated ({escalation_reason or 'unspecified'}): {body.query}"[:500],
                            customer_identifier=(
                                intent.extracted_order_id or intent.extracted_customer_id
                            ),
                        )
                        if redis is not None:
                            try:
                                await redis.setex(dedup_key, _ESCALATION_DEDUP_TTL_SECONDS, "1")
                            except Exception as exc:
                                logger.error("Escalation dedup write failed: %s", exc)
                    except Exception as exc:
                        logger.error("Escalation notification failed: %s", exc)

    tracer = getattr(request.app.state, "tracer", None)
    if tracer is not None:
        await tracer.persist(result.trace)  # costs the trace, then persists
    else:
        await result.trace.persist()

    if budget is not None:
        await budget.record(org, result.trace.total_tokens)

    if body.session_id and session_manager is not None:
        # Store the compacted history + this turn, so the summary written by
        # the context window replaces the old turns in Redis too.
        history.append({"role": "user", "content": body.query})
        history.append({"role": "assistant", "content": response_text})
        await session_manager.save_history(body.session_id, history)

    return {
        "response": response_text,
        "trace_id": result.trace.trace_id,
        "intent": intent.intent.value,
        "confidence": result.confidence,
        "escalated": result.escalated,
        "pending_approval": result.pending_approval,
    }


@router.get("/rag/test", tags=["dev"])
async def rag_test(
    request: Request,
    query: str = Query(..., description="Query to retrieve chunks for"),
) -> dict:
    """Diagnostic endpoint for inspecting hybrid retrieval results."""
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        return {
            "query": query,
            "results": [],
            "count": 0,
            "error": "retriever not initialised (ChromaDB/RAG unavailable)",
        }

    results = await retriever.retrieve(query)
    return {
        "query": query,
        "results": [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "score": r.score,
                "source": r.source,
                "metadata": r.metadata,
            }
            for r in results
        ],
        "count": len(results),
    }
