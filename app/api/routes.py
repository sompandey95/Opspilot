from pydantic import BaseModel

import httpx
from fastapi import APIRouter, Query, Request

from app.config import get_settings
from app.db.postgres import get_pool
from app.db.redis import get_redis

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
    query: str
    session_id: str | None = None


@router.post("/chat", tags=["chat"])
async def chat(body: ChatRequest) -> dict:
    return {"message": "not implemented yet"}


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
