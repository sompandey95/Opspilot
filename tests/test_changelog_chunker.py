"""Tests for changelog chunking (one chunk per dated entry)."""
from pathlib import Path

from app.rag.chunker import SmartChunker

KB = Path(__file__).parent.parent / "knowledge_base"


def test_changelog_chunks_one_per_entry():
    chunker = SmartChunker()
    chunks = chunker.chunk(KB / "changelogs" / "changelog_2026_q3.md")

    assert len(chunks) == 3
    assert [c.id for c in chunks] == [
        "changelog_changelog_2026_q3_001",
        "changelog_changelog_2026_q3_002",
        "changelog_changelog_2026_q3_003",
    ]


def test_changelog_metadata_and_date_prominent():
    chunker = SmartChunker()
    chunks = chunker.chunk(KB / "changelogs" / "changelog_2026_q3.md")

    first = chunks[0]
    assert first.metadata["doc_type"] == "changelog"
    assert first.metadata["date"] == "2026-07-15"
    assert first.metadata["category"] == "returns"
    assert "Electronics return window" in first.metadata["entry_title"]
    # Date must be prominent in the content itself (recency-aware retrieval)
    assert first.content.startswith("## 2026-07-15")
    assert "10 days" in first.content

    delivery = chunks[1]
    assert delivery.metadata["date"] == "2026-07-20"
    assert delivery.metadata["category"] == "delivery"


def test_changelog_entries_have_no_separator_rules():
    chunker = SmartChunker()
    for f in (KB / "changelogs").glob("*.md"):
        for chunk in chunker.chunk(f):
            assert "\n---" not in chunk.content


def test_chunk_directory_includes_changelogs():
    chunker = SmartChunker()
    chunks = chunker.chunk_directory(KB)
    changelog_chunks = [c for c in chunks if c.metadata.get("doc_type") == "changelog"]
    assert len(changelog_chunks) == 6  # 3 entries per quarterly file
