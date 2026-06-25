"""Tests for SmartChunker using the actual ShopEasy knowledge_base files."""
from pathlib import Path

import pytest

from app.rag.chunker import Chunk, SmartChunker

KB = Path(__file__).parent.parent / "knowledge_base"
chunker = SmartChunker()


# ------------------------------------------------------------------ #
# FAQ chunking                                                         #
# ------------------------------------------------------------------ #


def test_faq_returns_one_chunk_per_qa_pair():
    chunks = chunker.chunk(KB / "faqs" / "returns_and_refunds.md")
    # returns_and_refunds.md has exactly 8 Q&A pairs
    assert len(chunks) == 8


def test_faq_chunk_content_starts_with_q():
    chunks = chunker.chunk(KB / "faqs" / "returns_and_refunds.md")
    for chunk in chunks:
        assert chunk.content.startswith("Q:"), f"Chunk {chunk.id!r} does not start with 'Q:'"


def test_faq_metadata_fields():
    chunks = chunker.chunk(KB / "faqs" / "returns_and_refunds.md")
    for chunk in chunks:
        assert chunk.metadata["doc_type"] == "faq"
        assert chunk.metadata["source_file"] == "faqs/returns_and_refunds.md"
        assert chunk.metadata["question"], "question field must be non-empty"
        assert chunk.metadata["last_updated"], "last_updated must be non-empty"
        assert chunk.metadata["category"] == "returns_and_refunds"


def test_faq_all_files_produce_chunks():
    for faq_file in sorted((KB / "faqs").glob("*.md")):
        chunks = chunker.chunk(faq_file)
        assert len(chunks) > 0, f"No chunks from {faq_file.name}"
        for chunk in chunks:
            assert chunk.metadata["doc_type"] == "faq"
            assert chunk.metadata.get("question"), f"Missing question in {chunk.id}"


def test_faq_qa_pair_never_split():
    """Each chunk must contain both a question and its answer (not split mid-pair)."""
    for faq_file in sorted((KB / "faqs").glob("*.md")):
        chunks = chunker.chunk(faq_file)
        for chunk in chunks:
            assert "Q:" in chunk.content, f"No question in {chunk.id}"
            assert "A:" in chunk.content, f"No answer in {chunk.id}"


# ------------------------------------------------------------------ #
# Policy chunking                                                      #
# ------------------------------------------------------------------ #


def test_policy_splits_on_headings():
    chunks = chunker.chunk(KB / "policies" / "refund_policy.md")
    # refund_policy.md has 9 numbered sections (## 1.–## 9.) plus a preamble
    assert len(chunks) >= 9


def test_policy_section_title_populated():
    chunks = chunker.chunk(KB / "policies" / "refund_policy.md")
    for chunk in chunks:
        assert chunk.metadata.get("section_title"), f"section_title missing in {chunk.id}"


def test_policy_effective_date_extracted():
    chunks = chunker.chunk(KB / "policies" / "refund_policy.md")
    assert any(
        c.metadata.get("effective_date") == "June 1, 2026" for c in chunks
    ), "effective_date not extracted from refund_policy.md"


def test_policy_doc_type():
    for policy_file in sorted((KB / "policies").glob("*.md")):
        chunks = chunker.chunk(policy_file)
        assert len(chunks) > 0, f"No chunks from {policy_file.name}"
        for chunk in chunks:
            assert chunk.metadata["doc_type"] == "policy"


def test_policy_all_effective_dates_present():
    """Every policy file embeds its effective date in at least one chunk's metadata."""
    for policy_file in sorted((KB / "policies").glob("*.md")):
        chunks = chunker.chunk(policy_file)
        assert any(
            c.metadata.get("effective_date") for c in chunks
        ), f"No effective_date found in chunks from {policy_file.name}"


# ------------------------------------------------------------------ #
# Ticket chunking                                                      #
# ------------------------------------------------------------------ #


def test_ticket_single_chunk():
    chunks = chunker.chunk(KB / "tickets" / "ticket_001.md")
    assert len(chunks) == 1


def test_ticket_metadata():
    chunk = chunker.chunk(KB / "tickets" / "ticket_001.md")[0]
    assert chunk.metadata["doc_type"] == "ticket"
    assert chunk.metadata["ticket_id"] == "TICKET-001"
    assert chunk.metadata["status"] == "Resolved"
    assert chunk.metadata["category"] == "delivery"
    assert chunk.metadata["resolution"]


def test_ticket_full_conversation_preserved():
    chunk = chunker.chunk(KB / "tickets" / "ticket_001.md")[0]
    assert "**Customer:**" in chunk.content
    assert "**Agent:**" in chunk.content


def test_ticket_all_files_single_chunk():
    for ticket_file in sorted((KB / "tickets").glob("*.md")):
        chunks = chunker.chunk(ticket_file)
        assert len(chunks) == 1, f"{ticket_file.name} produced {len(chunks)} chunks, expected 1"
        assert chunks[0].metadata["doc_type"] == "ticket"


def test_ticket_all_metadata_extracted():
    for ticket_file in sorted((KB / "tickets").glob("*.md")):
        chunk = chunker.chunk(ticket_file)[0]
        meta = chunk.metadata
        assert meta.get("ticket_id"), f"ticket_id missing in {ticket_file.name}"
        assert meta.get("status"), f"status missing in {ticket_file.name}"
        assert meta.get("category"), f"category missing in {ticket_file.name}"


# ------------------------------------------------------------------ #
# API doc chunking                                                     #
# ------------------------------------------------------------------ #


def test_api_doc_produces_endpoint_chunks():
    chunks = chunker.chunk(KB / "api_docs" / "order_api.md")
    # 6 endpoints + 1 overview chunk
    assert len(chunks) >= 6


def test_api_doc_endpoint_method_and_path():
    chunks = chunker.chunk(KB / "api_docs" / "order_api.md")
    endpoint_chunks = [c for c in chunks if c.metadata.get("method")]
    assert len(endpoint_chunks) == 6, f"Expected 6 endpoint chunks, got {len(endpoint_chunks)}"
    methods = {c.metadata["method"] for c in endpoint_chunks}
    assert "GET" in methods
    assert "POST" in methods


def test_api_doc_doc_type():
    chunks = chunker.chunk(KB / "api_docs" / "order_api.md")
    for chunk in chunks:
        assert chunk.metadata["doc_type"] == "api_doc"


# ------------------------------------------------------------------ #
# Cross-cutting constraints                                            #
# ------------------------------------------------------------------ #


def test_no_chunk_exceeds_1000_tokens():
    all_chunks = chunker.chunk_directory(KB)
    oversized = [c for c in all_chunks if c.token_count > 1000]
    assert oversized == [], (
        f"{len(oversized)} chunk(s) exceed 1000 tokens:\n"
        + "\n".join(f"  {c.id}: {c.token_count} tokens" for c in oversized[:10])
    )


def test_chunk_ids_are_unique():
    all_chunks = chunker.chunk_directory(KB)
    ids = [c.id for c in all_chunks]
    duplicates = [cid for cid in set(ids) if ids.count(cid) > 1]
    assert not duplicates, f"Duplicate chunk IDs: {duplicates}"


def test_all_chunks_have_required_metadata():
    all_chunks = chunker.chunk_directory(KB)
    for chunk in all_chunks:
        assert "doc_type" in chunk.metadata, f"{chunk.id} missing doc_type"
        assert "source_file" in chunk.metadata, f"{chunk.id} missing source_file"
        assert isinstance(chunk.token_count, int) and chunk.token_count > 0, (
            f"{chunk.id} has invalid token_count: {chunk.token_count}"
        )


def test_chunk_directory_covers_all_md_files():
    md_files = set(KB.rglob("*.md"))
    all_chunks = chunker.chunk_directory(KB)
    sourced_files = {c.metadata["source_file"] for c in all_chunks}
    for md_file in md_files:
        rel = SmartChunker._relative_source(md_file)
        assert rel in sourced_files, f"No chunks found for {rel}"


def test_token_count_is_positive_integer():
    chunks = chunker.chunk(KB / "faqs" / "payments_and_billing.md")
    for chunk in chunks:
        assert isinstance(chunk.token_count, int)
        assert chunk.token_count > 0
