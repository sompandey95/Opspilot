import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.db.postgres import close_db, init_db
from app.db.redis import close_redis, init_redis

logger = logging.getLogger(__name__)


async def _init_rag(app: FastAPI) -> None:
    """
    Build the RAG stack and store the retriever on app.state.

    Designed to never crash startup: if ChromaDB is empty, Azure creds are
    missing, or the cross-encoder model isn't available, we log and continue
    with app.state.retriever left as None (or a partially-initialised retriever).
    """
    app.state.retriever = None

    settings = get_settings()

    # Import here so a missing optional dep doesn't break the whole app import.
    from app.rag.bm25_index import BM25Index
    from app.rag.embedder import AzureEmbedder
    from app.rag.reranker import Reranker
    from app.rag.retriever import HybridRetriever
    from app.rag.vector_store import ChromaStore

    try:
        vector_store = ChromaStore(settings)
        vector_store.get_or_create_collection()
        doc_count = vector_store.get_collection_count()
    except Exception as exc:
        logger.error("ChromaDB unavailable (%s) — RAG retriever disabled", exc)
        return

    if doc_count == 0:
        logger.warning(
            "ChromaDB collection is empty — run scripts/ingest_knowledge.py. "
            "RAG retriever will return no results until documents are ingested."
        )

    embedder = AzureEmbedder(settings)
    if not (settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_ENDPOINT):
        logger.error(
            "Azure OpenAI credentials missing — vector search will be skipped; "
            "retrieval falls back to BM25 keyword search only."
        )

    bm25_index = BM25Index()

    try:
        logger.info(
            "Loading cross-encoder reranker (downloads model on first run, may take a minute)…"
        )
        reranker = Reranker()
    except Exception as exc:
        logger.error("Could not load cross-encoder reranker (%s) — reranking disabled", exc)
        reranker = None

    retriever = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        bm25_index=bm25_index,
        reranker=reranker,
        settings=settings,
    )

    try:
        await retriever.initialize()
    except Exception as exc:
        logger.error("Failed to build BM25 index (%s)", exc)

    app.state.retriever = retriever
    logger.info("RAG retriever ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await init_redis()
    await _init_rag(app)
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
