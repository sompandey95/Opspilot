from typing import Any

import asyncpg

from app.config import get_settings

_pool: asyncpg.Pool | None = None


async def init_db() -> None:
    global _pool
    settings = get_settings()
    # asyncpg uses postgresql:// not postgresql+asyncpg://
    dsn = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    _pool = await asyncpg.create_pool(dsn)


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialised — call init_db() first")
    return _pool


async def execute(query: str, *args: Any) -> str:
    return await get_pool().execute(query, *args)


async def fetch_one(query: str, *args: Any) -> asyncpg.Record | None:
    return await get_pool().fetchrow(query, *args)


async def fetch_all(query: str, *args: Any) -> list[asyncpg.Record]:
    return await get_pool().fetch(query, *args)
