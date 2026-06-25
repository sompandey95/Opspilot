"""ChromaDB vector store wrapper."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import chromadb

from app.config import Settings
from app.rag.chunker import Chunk

_DEFAULT_COLLECTION = "opspilot_knowledge"
_UPSERT_BATCH = 100  # ChromaDB recommended batch ceiling


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    metadata: dict
    score: float  # 0–1; higher = more relevant
    source: str   # "vector" | "bm25" | "reranked"


class ChromaStore:
    def __init__(self, settings: Settings) -> None:
        self._client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )
        self._collection: chromadb.Collection | None = None

    # ------------------------------------------------------------------ #
    # Collection lifecycle                                                 #
    # ------------------------------------------------------------------ #

    def get_or_create_collection(self, name: str = _DEFAULT_COLLECTION) -> chromadb.Collection:
        """
        Get or create the named collection with cosine similarity space.
        text-embedding-3-large vectors are normalised, so cosine == dot-product.
        """
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def delete_collection(self, name: str = _DEFAULT_COLLECTION) -> None:
        self._client.delete_collection(name)
        self._collection = None

    def get_collection_count(self) -> int:
        return self._collection_obj.count()

    @property
    def _collection_obj(self) -> chromadb.Collection:
        if self._collection is None:
            return self.get_or_create_collection()
        return self._collection

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    async def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunks + embeddings in batches of 100."""
        for i in range(0, len(chunks), _UPSERT_BATCH):
            batch_chunks = chunks[i : i + _UPSERT_BATCH]
            batch_embeddings = embeddings[i : i + _UPSERT_BATCH]

            # Chromadb metadata values must be str | int | float | bool
            def _sanitise(meta: dict) -> dict:
                return {
                    k: (v if isinstance(v, (str, int, float, bool)) else str(v) if v is not None else "")
                    for k, v in meta.items()
                }

            await asyncio.to_thread(
                self._collection_obj.upsert,
                ids=[c.id for c in batch_chunks],
                documents=[c.content for c in batch_chunks],
                embeddings=batch_embeddings,
                metadatas=[_sanitise(c.metadata) for c in batch_chunks],
            )

    # ------------------------------------------------------------------ #
    # Read                                                                 #
    # ------------------------------------------------------------------ #

    async def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        """
        Vector search. Returns results ordered by descending cosine similarity.
        Score = 1 - cosine_distance (chromadb cosine space).
        """
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = await asyncio.to_thread(self._collection_obj.query, **kwargs)

        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            RetrievalResult(
                chunk_id=cid,
                content=doc,
                metadata=meta or {},
                score=round(1.0 - dist, 6),
                source="vector",
            )
            for cid, doc, meta, dist in zip(ids, documents, metadatas, distances)
        ]

    async def get_all_documents(self) -> list[dict]:
        """Fetch all stored documents — used for building a BM25 index."""
        result = await asyncio.to_thread(
            self._collection_obj.get,
            include=["documents", "metadatas"],
        )
        return [
            {"id": cid, "content": doc, "metadata": meta or {}}
            for cid, doc, meta in zip(
                result["ids"],
                result["documents"],
                result["metadatas"],
            )
        ]
