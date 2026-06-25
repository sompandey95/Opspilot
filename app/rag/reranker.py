"""Cross-encoder reranker for retrieved chunks."""
from __future__ import annotations

import logging

from app.rag.vector_store import RetrievalResult

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        # Imported lazily so the rest of the RAG stack can be used without
        # pulling in sentence_transformers / torch when reranking is not needed.
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder model %r (CPU)…", model_name)
        self._model = CrossEncoder(model_name)
        logger.info("Cross-encoder model loaded.")

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Score each (query, content) pair, sort by score descending, and return
        the top_k results with source='reranked' and the cross-encoder score.
        """
        if not results:
            return []

        pairs = [(query, r.content) for r in results]
        scores = self._model.predict(pairs)

        rescored = [
            RetrievalResult(
                chunk_id=r.chunk_id,
                content=r.content,
                metadata=r.metadata,
                score=float(score),
                source="reranked",
            )
            for r, score in zip(results, scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]
