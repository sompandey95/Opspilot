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


def _init_tools(app: FastAPI) -> None:
    """Build the tool registry + schema validator on app.state (fault-tolerant)."""
    app.state.tool_registry = None
    app.state.schema_validator = None
    try:
        from app.guardrails.schemas import SchemaValidator
        from app.tools.registry import build_default_registry

        registry = build_default_registry(get_settings(), retriever=app.state.retriever)
        app.state.tool_registry = registry
        app.state.schema_validator = SchemaValidator(registry)
        logger.info("Tool registry ready: %d tools", len(registry))
    except Exception as exc:
        logger.error("Tool registry init failed (%s) — tools disabled", exc)


def _init_agent(app: FastAPI) -> None:
    """Build the LLM client, intent classifier, and ReAct agent (fault-tolerant)."""
    app.state.llm_client = None
    app.state.intent_classifier = None
    app.state.react_agent = None

    settings = get_settings()
    if not (settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_ENDPOINT):
        logger.error("Azure OpenAI credentials missing — agent disabled")
        return

    try:
        from app.agent.intent_classifier import IntentClassifier
        from app.agent.react_agent import ReActAgent
        from app.llm.client import LLMClient

        llm = LLMClient(settings)
        app.state.llm_client = llm
        app.state.intent_classifier = IntentClassifier(llm)

        if app.state.tool_registry is None or app.state.schema_validator is None:
            logger.error("Tool registry unavailable — agent disabled")
            return

        agent = ReActAgent(
            llm=llm,
            tool_registry=app.state.tool_registry,
            schema_validator=app.state.schema_validator,
            settings=settings,
        )
        app.state.react_agent = agent
        logger.info("ReAct agent ready (prompt %s)", agent.prompt_version)
    except Exception as exc:
        logger.error("Agent init failed (%s) — agent disabled", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await init_redis()
    await _init_rag(app)
    _init_tools(app)
    _init_agent(app)
    logger.info("OpsPilot started")
    yield
    await close_db()
    await close_redis()


def create_app() -> FastAPI:
    app = FastAPI(title="OpsPilot", version="0.1.0", lifespan=lifespan)

    # Dev-open CORS so the local frontend console (frontend/index.html) can
    # call the API from file:// — Phase 5 middleware tightens this.
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/")
    async def root() -> dict:
        return {"service": "opspilot", "version": "0.1.0"}

    return app


app = create_app()
