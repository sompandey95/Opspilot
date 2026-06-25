"""Hybrid retriever: vector + BM25 + RRF fusion + cross-encoder reranking."""
from __future__ import annotations

import logging

from app.config import Settings
from app.rag.bm25_index import BM25Index
from app.rag.embedder import AzureEmbedder
from app.rag.reranker import Reranker
from app.rag.vector_store import ChromaStore, RetrievalResult

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard reciprocal-rank-fusion constant


class HybridRetriever:
    def __init__(
        self,
        vector_store: ChromaStore,
        embedder: AzureEmbedder,
        bm25_index: BM25Index,
        reranker: Reranker | None,
        settings: Settings,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._bm25 = bm25_index
        self._reranker = reranker
        self._settings = settings

    # ------------------------------------------------------------------ #
    # Initialization                                                       #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Load all docs from ChromaDB and build the BM25 index."""
        documents = await self._vector_store.get_all_documents()
        self._bm25.build_index(documents)
        logger.info("BM25 index built with %d documents", self._bm25.size)

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or self._settings.RERANK_TOP_K
        candidate_k = self._settings.RETRIEVAL_TOP_K

        # Step 1: vector search (skip gracefully if embedding fails)
        vector_results: list[RetrievalResult] = []
        try:
            query_embedding = await self._embedder.embed_query(query)
            vector_results = await self._vector_store.query(query_embedding, top_k=candidate_k)
        except Exception as exc:
            logger.error("Vector search failed (%s) — falling back to BM25 only", exc)

        # Step 2: BM25 search
        bm25_results = self._bm25.search(query, top_k=candidate_k)

        # Step 3: reciprocal rank fusion
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results, k=_RRF_K)

        # Step 4: cross-encoder reranking over top 2*top_k fused candidates
        rerank_candidates = fused[: 2 * top_k]
        if self._reranker is None:
            logger.warning("Reranker unavailable — returning RRF-fused results")
            reranked = fused[:top_k]
        else:
            try:
                reranked = self._reranker.rerank(query, rerank_candidates, top_k=top_k)
            except Exception as exc:
                logger.error("Reranking failed (%s) — returning fused results", exc)
                reranked = fused[:top_k]

        logger.info(
            "Retrieved %d chunks: %d vector, %d bm25, reranked to %d",
            len(reranked),
            len(vector_results),
            len(bm25_results),
            top_k,
        )
        return reranked

    # ------------------------------------------------------------------ #
    # Reciprocal Rank Fusion                                               #
    # ------------------------------------------------------------------ #

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        k: int = _RRF_K,
    ) -> list[RetrievalResult]:
        """
        Merge two ranked lists via RRF: score = sum(1 / (k + rank)) across the
        lists in which a document appears (rank is 1-based). Dedupe by chunk_id.
        """
        rrf_scores: dict[str, float] = {}
        merged: dict[str, RetrievalResult] = {}

        for results in (vector_results, bm25_results):
            for rank, result in enumerate(results, start=1):
                cid = result.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
                # Keep the first-seen RetrievalResult (vector preferred over bm25)
                if cid not in merged:
                    merged[cid] = result

        ordered_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

        fused: list[RetrievalResult] = []
        for cid in ordered_ids:
            base = merged[cid]
            fused.append(
                RetrievalResult(
                    chunk_id=base.chunk_id,
                    content=base.content,
                    metadata=base.metadata,
                    score=rrf_scores[cid],
                    source=base.source,
                )
            )
        return fused
