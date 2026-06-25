"""Tests for BM25Index, RRF fusion, and dedup — pure logic, no external deps."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.bm25_index import BM25Index
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import RetrievalResult


# ------------------------------------------------------------------ #
# Fixtures / helpers                                                   #
# ------------------------------------------------------------------ #

SYNTHETIC_DOCS = [
    {"id": "doc_refund", "content": "Refunds are processed to the original payment method in 5 to 7 business days.", "metadata": {"doc_type": "policy"}},
    {"id": "doc_delivery", "content": "Standard delivery to metro cities takes 3 to 5 business days via Delhivery.", "metadata": {"doc_type": "faq"}},
    {"id": "doc_cod", "content": "Cash on delivery is available for orders under five thousand rupees.", "metadata": {"doc_type": "policy"}},
    {"id": "doc_upi", "content": "If a UPI payment is debited but the order is not placed, the refund arrives in 24 to 48 hours.", "metadata": {"doc_type": "faq"}},
]


def _result(chunk_id, source="vector", score=1.0, content="x", metadata=None):
    return RetrievalResult(
        chunk_id=chunk_id,
        content=content,
        metadata=metadata or {},
        score=score,
        source=source,
    )


def _make_retriever(reranker=None):
    """HybridRetriever with mocked vector store + embedder + settings."""
    vector_store = MagicMock()
    embedder = MagicMock()
    settings = MagicMock()
    settings.RETRIEVAL_TOP_K = 10
    settings.RERANK_TOP_K = 5
    return HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        bm25_index=BM25Index(),
        reranker=reranker,
        settings=settings,
    )


# ------------------------------------------------------------------ #
# BM25Index                                                            #
# ------------------------------------------------------------------ #


def test_bm25_builds_and_searches():
    index = BM25Index()
    index.build_index(SYNTHETIC_DOCS)
    assert index.size == 4

    results = index.search("refund payment method", top_k=3)
    assert len(results) > 0
    # The refund doc should rank first for this query
    assert results[0].chunk_id == "doc_refund"
    assert all(r.source == "bm25" for r in results)


def test_bm25_relevance_ordering():
    index = BM25Index()
    index.build_index(SYNTHETIC_DOCS)
    results = index.search("UPI payment debited not placed", top_k=4)
    assert results[0].chunk_id == "doc_upi"


def test_bm25_empty_index_returns_empty():
    index = BM25Index()
    index.build_index([])
    assert index.size == 0
    assert index.search("anything", top_k=5) == []


def test_bm25_no_match_returns_empty():
    index = BM25Index()
    index.build_index(SYNTHETIC_DOCS)
    # No token overlap → all scores zero → filtered out
    results = index.search("xyzzy zzzz qqqq", top_k=5)
    assert results == []


def test_bm25_respects_top_k():
    index = BM25Index()
    index.build_index(SYNTHETIC_DOCS)
    results = index.search("delivery business days payment refund order", top_k=2)
    assert len(results) <= 2


def test_bm25_tokenization_handles_punctuation():
    index = BM25Index()
    # Multi-doc corpus so the search term has a positive IDF (it appears in
    # only one doc). The target doc wraps the term in varied punctuation/case.
    index.build_index(
        [
            {"id": "d1", "content": "Refund, refund; REFUND! refunded.", "metadata": {}},
            {"id": "d2", "content": "Delivery takes three to five days.", "metadata": {}},
            {"id": "d3", "content": "Cash on delivery for small orders.", "metadata": {}},
        ]
    )
    results = index.search("refund", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "d1"


# ------------------------------------------------------------------ #
# Reciprocal Rank Fusion                                               #
# ------------------------------------------------------------------ #


def test_rrf_ordering_doc_in_both_lists_ranks_higher():
    retriever = _make_retriever()
    # "A" appears at rank 1 in both lists → highest RRF score
    vector = [_result("A", "vector"), _result("B", "vector"), _result("C", "vector")]
    bm25 = [_result("A", "bm25"), _result("D", "bm25"), _result("E", "bm25")]

    fused = retriever._reciprocal_rank_fusion(vector, bm25, k=60)
    assert fused[0].chunk_id == "A"

    # A: 1/61 + 1/61 ; B: 1/62 ; D: 1/62 — A strictly greater
    a_score = next(r.score for r in fused if r.chunk_id == "A")
    b_score = next(r.score for r in fused if r.chunk_id == "B")
    assert a_score > b_score


def test_rrf_dedup_by_chunk_id():
    retriever = _make_retriever()
    vector = [_result("A", "vector"), _result("B", "vector")]
    bm25 = [_result("A", "bm25"), _result("B", "bm25")]

    fused = retriever._reciprocal_rank_fusion(vector, bm25, k=60)
    ids = [r.chunk_id for r in fused]
    # Two unique docs despite appearing in both lists
    assert sorted(ids) == ["A", "B"]
    assert len(ids) == len(set(ids))


def test_rrf_score_formula():
    retriever = _make_retriever()
    # Single doc at rank 1 in one list → 1/(60+1)
    vector = [_result("A", "vector")]
    bm25 = []
    fused = retriever._reciprocal_rank_fusion(vector, bm25, k=60)
    assert fused[0].chunk_id == "A"
    assert fused[0].score == pytest.approx(1.0 / 61.0)


def test_rrf_prefers_vector_result_object_on_tie():
    retriever = _make_retriever()
    vector = [_result("A", "vector", content="from-vector")]
    bm25 = [_result("A", "bm25", content="from-bm25")]
    fused = retriever._reciprocal_rank_fusion(vector, bm25, k=60)
    assert len(fused) == 1
    # First-seen (vector) result object is kept
    assert fused[0].content == "from-vector"


def test_rrf_handles_empty_lists():
    retriever = _make_retriever()
    assert retriever._reciprocal_rank_fusion([], [], k=60) == []


# ------------------------------------------------------------------ #
# End-to-end retrieve() with mocked vector store + embedder            #
# ------------------------------------------------------------------ #


async def test_retrieve_merges_vector_and_bm25_no_reranker():
    retriever = _make_retriever(reranker=None)

    # BM25 index over synthetic docs
    retriever._bm25.build_index(SYNTHETIC_DOCS)

    # Mock embedder + vector store
    retriever._embedder.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    retriever._vector_store.query = AsyncMock(
        return_value=[_result("doc_delivery", "vector", score=0.9, content="delivery")]
    )

    # "original method" is distinctive to doc_refund (positive BM25 IDF),
    # while doc_delivery is supplied by the mocked vector search.
    results = await retriever.retrieve("original payment method", top_k=3)
    # Without a reranker, fused RRF results are returned (capped at top_k)
    assert len(results) <= 3
    ids = {r.chunk_id for r in results}
    # Both the vector hit and the BM25 hit should be present in the merged set
    assert "doc_delivery" in ids
    assert "doc_refund" in ids


async def test_retrieve_falls_back_to_bm25_when_vector_fails():
    retriever = _make_retriever(reranker=None)
    retriever._bm25.build_index(SYNTHETIC_DOCS)

    # Embedding raises → vector search skipped, BM25 still works
    retriever._embedder.embed_query = AsyncMock(side_effect=RuntimeError("no azure creds"))

    results = await retriever.retrieve("UPI payment debited", top_k=3)
    assert len(results) > 0
    assert results[0].chunk_id == "doc_upi"


async def test_retrieve_invokes_reranker_and_returns_top_k():
    fake_reranker = MagicMock()
    # Reranker reverses to put doc_cod first, returns top_k
    fake_reranker.rerank = MagicMock(
        return_value=[_result("doc_cod", "reranked", score=9.9)]
    )
    retriever = _make_retriever(reranker=fake_reranker)
    retriever._bm25.build_index(SYNTHETIC_DOCS)
    retriever._embedder.embed_query = AsyncMock(return_value=[0.1])
    retriever._vector_store.query = AsyncMock(return_value=[])

    results = await retriever.retrieve("cash on delivery", top_k=1)
    fake_reranker.rerank.assert_called_once()
    assert len(results) == 1
    assert results[0].source == "reranked"
    assert results[0].chunk_id == "doc_cod"


async def test_retrieve_uses_settings_default_top_k():
    retriever = _make_retriever(reranker=None)
    retriever._bm25.build_index(SYNTHETIC_DOCS)
    retriever._embedder.embed_query = AsyncMock(return_value=[0.1])
    retriever._vector_store.query = AsyncMock(return_value=[])

    # top_k not passed → uses settings.RERANK_TOP_K (5)
    results = await retriever.retrieve("delivery refund payment order business")
    assert len(results) <= 5
