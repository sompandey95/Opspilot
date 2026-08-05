import logging

from pydantic import BaseModel, Field

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

import json
import uuid

from app.config import get_settings
from app.db.postgres import fetch_one, get_pool
from app.db.redis import get_redis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


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

    intent = await classifier.classify(body.query)

    result = await agent.run(body.query, intent, session_id=body.session_id)

    # Direct intent-escalations still notify a human (best-effort).
    if result.escalated and intent.intent.value == "escalate":
        registry = getattr(request.app.state, "tool_registry", None)
        escalate_tool = registry.get("escalate_to_manager") if registry else None
        if escalate_tool is not None:
            try:
                await escalate_tool.execute(
                    reason="customer_requested_human", summary=body.query
                )
            except Exception as exc:
                logger.error("Escalation notification failed: %s", exc)

    await result.trace.persist()

    return {
        "response": result.response,
        "trace_id": result.trace.trace_id,
        "intent": intent.intent.value,
        "confidence": result.confidence,
        "escalated": result.escalated,
        "pending_approval": result.pending_approval,
    }


@router.get("/traces/{trace_id}", tags=["dev"])
async def get_trace(trace_id: str) -> dict:
    """Dev/debug endpoint — full trace by ID for the interactive console.
    Superseded by the Phase 6 admin routes."""
    try:
        tid = uuid.UUID(trace_id)
    except ValueError:
        return JSONResponse(status_code=400, content={"detail": "Invalid trace ID"})

    try:
        row = await fetch_one(
            """
            SELECT id, session_id, query, intent, model, steps, response,
                   confidence, total_latency_ms, total_tokens, input_tokens,
                   output_tokens, hitl_triggered, escalated, prompt_version,
                   created_at
            FROM traces WHERE id = $1
            """,
            tid,
        )
    except Exception as exc:
        logger.error("Trace lookup failed: %s", exc)
        return JSONResponse(status_code=503, content={"detail": "Trace store unavailable"})

    if row is None:
        return JSONResponse(status_code=404, content={"detail": "Trace not found"})

    trace = dict(row)
    trace["steps"] = json.loads(trace["steps"])
    return trace


@router.get("/rag/test", tags=["dev"])
async def rag_test(
    request: Request,
    query: str = Query(..., description="Query to retrieve chunks for"),
) -> dict:
    """Dev/debug endpoint — exercises the hybrid retriever. Will be removed later."""
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
