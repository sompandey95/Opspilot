"""Tests for ingestion-time semantic deduplication."""
import pytest

from app.rag.chunker import Chunk
from app.rag.dedup import SemanticDeduplicator


def _chunk(cid: str, doc_type: str, content: str = "some content") -> Chunk:
    return Chunk(id=cid, content=content, metadata={"doc_type": doc_type})


def test_near_duplicates_dropped_keeping_authoritative():
    # FAQ restates the policy with an almost identical embedding
    chunks = [
        _chunk("faq_returns_001", "faq", "Electronics can be returned within 7 days."),
        _chunk("policy_refund_001", "policy", "Electronics: 7 day return window from delivery date."),
        _chunk("ticket_001", "ticket", "Customer asked about COD limits."),
    ]
    embeddings = [
        [1.0, 0.0, 0.01],   # ~identical to policy
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],    # unrelated
    ]

    kept, kept_emb, dropped = SemanticDeduplicator(threshold=0.95).deduplicate(chunks, embeddings)

    kept_ids = [c.id for c in kept]
    assert "policy_refund_001" in kept_ids          # authoritative copy survives
    assert "faq_returns_001" not in kept_ids        # duplicate dropped
    assert "ticket_001" in kept_ids
    assert len(kept) == len(kept_emb) == 2
    assert dropped == [("faq_returns_001", "policy_refund_001")]


def test_distinct_chunks_all_kept_in_original_order():
    chunks = [_chunk(f"c{i}", "faq") for i in range(3)]
    embeddings = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]

    kept, _, dropped = SemanticDeduplicator().deduplicate(chunks, embeddings)

    assert [c.id for c in kept] == ["c0", "c1", "c2"]
    assert dropped == []


def test_empty_input():
    kept, kept_emb, dropped = SemanticDeduplicator().deduplicate([], [])
    assert kept == [] and kept_emb == [] and dropped == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        SemanticDeduplicator().deduplicate([_chunk("a", "faq")], [])


def test_ties_within_same_type_keep_longer_content():
    chunks = [
        _chunk("faq_short", "faq", "Returns take 10 days."),
        _chunk("faq_long", "faq", "Returns take 10 days from delivery, and refunds are issued to source."),
    ]
    embeddings = [[1.0, 0.0], [1.0, 0.0]]

    kept, _, dropped = SemanticDeduplicator().deduplicate(chunks, embeddings)

    assert [c.id for c in kept] == ["faq_long"]
    assert dropped == [("faq_short", "faq_long")]
