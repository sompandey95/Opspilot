#!/usr/bin/env python3
"""
Ingest ShopEasy knowledge base documents into ChromaDB.

Usage:
    python scripts/ingest_knowledge.py
    python scripts/ingest_knowledge.py --reset
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.rag.chunker import Chunk, SmartChunker
from app.rag.embedder import AzureEmbedder
from app.rag.vector_store import ChromaStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_EMBED_BATCH = 16


async def ingest(reset: bool = False) -> None:
    settings = get_settings()
    kb_path = Path(__file__).parent.parent / "knowledge_base"

    if not kb_path.exists():
        logger.error("knowledge_base/ not found at %s", kb_path)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 1. Chunk                                                             #
    # ------------------------------------------------------------------ #
    logger.info("Chunking knowledge base: %s", kb_path)
    chunker = SmartChunker()
    chunks = chunker.chunk_directory(kb_path)
    md_files = list(kb_path.rglob("*.md"))
    logger.info("  %d .md files → %d chunks", len(md_files), len(chunks))

    if not chunks:
        logger.warning("No chunks produced — nothing to ingest.")
        return

    # ------------------------------------------------------------------ #
    # 2. ChromaDB                                                          #
    # ------------------------------------------------------------------ #
    store = ChromaStore(settings)

    if reset:
        logger.info("--reset: deleting existing collection")
        try:
            store.delete_collection()
            logger.info("  Collection deleted.")
        except Exception as exc:
            logger.warning("  Could not delete collection (may not exist): %s", exc)

    store.get_or_create_collection()
    logger.info("  Collection ready — current count: %d", store.get_collection_count())

    # ------------------------------------------------------------------ #
    # 3. Embed                                                             #
    # ------------------------------------------------------------------ #
    embedder = AzureEmbedder(settings)
    total_batches = (len(chunks) + _EMBED_BATCH - 1) // _EMBED_BATCH
    all_embeddings: list[list[float] | None] = []

    for batch_num, start in enumerate(range(0, len(chunks), _EMBED_BATCH), start=1):
        batch: list[Chunk] = chunks[start : start + _EMBED_BATCH]
        logger.info(
            "  Embedding batch %d/%d (%d texts)…",
            batch_num,
            total_batches,
            len(batch),
        )
        try:
            embeddings = await embedder.embed_texts([c.content for c in batch])
            all_embeddings.extend(embeddings)
        except Exception as exc:
            logger.error("  Batch %d/%d failed: %s — skipping", batch_num, total_batches, exc)
            all_embeddings.extend([None] * len(batch))

    # ------------------------------------------------------------------ #
    # 4. Store                                                             #
    # ------------------------------------------------------------------ #
    valid_pairs = [
        (chunk, emb)
        for chunk, emb in zip(chunks, all_embeddings)
        if emb is not None
    ]
    skipped = len(chunks) - len(valid_pairs)
    if skipped:
        logger.warning("  %d chunks skipped due to embedding errors", skipped)

    if valid_pairs:
        valid_chunks, valid_embeddings = zip(*valid_pairs)
        await store.add_chunks(list(valid_chunks), list(valid_embeddings))

    final_count = store.get_collection_count()
    logger.info(
        "Ingestion complete: %d docs | %d chunks created | %d stored in ChromaDB",
        len(md_files),
        len(chunks),
        final_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest ShopEasy knowledge base into ChromaDB"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the ChromaDB collection before ingesting",
    )
    args = parser.parse_args()
    asyncio.run(ingest(reset=args.reset))


if __name__ == "__main__":
    main()
