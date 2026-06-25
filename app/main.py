import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.db.postgres import close_db, init_db
from app.db.redis import close_redis, init_redis

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await init_redis()
    logger.info("OpsPilot started")
    yield
    await close_db()
    await close_redis()


def create_app() -> FastAPI:
    app = FastAPI(title="OpsPilot", version="0.1.0", lifespan=lifespan)
    app.include_router(router)

    @app.get("/")
    async def root() -> dict:
        return {"service": "opspilot", "version": "0.1.0"}

    return app


app = create_app()
