"""Assemble the production pipeline (agent + tools + judges) for eval runs.

Shared by scripts/run_evals.py (CLI) and the POST /admin/evals/run endpoint
so both trigger paths build the exact same runner.
"""
from __future__ import annotations

import logging

from app.config import Settings

logger = logging.getLogger(__name__)


async def build_eval_runner(settings: Settings, with_llm_judges: bool = True, with_retrieval: bool = True):
    from app.agent.intent_classifier import IntentClassifier
    from app.agent.react_agent import ReActAgent
    from app.guardrails.input_guard import InputGuard
    from app.guardrails.output_guard import OutputGuard
    from app.guardrails.schemas import SchemaValidator
    from app.llm.client import LLMClient
    from app.tools.registry import build_default_registry
    from evals.judges.faithfulness import FaithfulnessJudge
    from evals.judges.relevance import RelevanceJudge
    from evals.runners.eval_runner import EvalRunner

    llm = LLMClient(settings)

    retriever = None
    if with_retrieval:
        try:
            from app.rag.bm25_index import BM25Index
            from app.rag.embedder import AzureEmbedder
            from app.rag.reranker import Reranker
            from app.rag.retriever import HybridRetriever
            from app.rag.vector_store import ChromaStore

            vector_store = ChromaStore(settings)
            vector_store.get_or_create_collection()
            retriever = HybridRetriever(
                vector_store=vector_store,
                embedder=AzureEmbedder(settings),
                bm25_index=BM25Index(),
                reranker=Reranker(),
                settings=settings,
            )
            await retriever.initialize()
        except Exception as exc:
            logger.warning("retrieval unavailable (%s) — retrieval metrics skipped", exc)
            retriever = None

    registry = build_default_registry(settings, retriever=retriever)
    agent = ReActAgent(
        llm=llm,
        tool_registry=registry,
        schema_validator=SchemaValidator(registry),
        settings=settings,
    )

    return EvalRunner(
        agent=agent,
        classifier=IntentClassifier(llm),
        input_guard=InputGuard(settings),
        output_guard=OutputGuard(),
        faithfulness_judge=FaithfulnessJudge(llm) if with_llm_judges else None,
        relevance_judge=RelevanceJudge(llm) if with_llm_judges else None,
        retriever=retriever,
        settings=settings,
    )
