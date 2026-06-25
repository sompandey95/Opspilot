from pydantic import BaseModel

import httpx
from fastapi import APIRouter

from app.config import get_settings
from app.db.postgres import get_pool
from app.db.redis import get_redis

router = APIRouter(prefix="/api/v1")


@router.get("/health", tags=["ops"])
async def health_check() -> dict:
    postgres_ok = False
    redis_ok = False
    chroma_ok = False

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

    return {
        "status": "healthy",
        "postgres": postgres_ok,
        "redis": redis_ok,
        "chromadb": chroma_ok,
    }


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


@router.post("/chat", tags=["chat"])
async def chat(body: ChatRequest) -> dict:
    return {"message": "not implemented yet"}
