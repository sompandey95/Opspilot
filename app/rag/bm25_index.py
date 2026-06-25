"""BM25 keyword-search index over ChromaDB documents."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from app.rag.vector_store import RetrievalResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []

    def build_index(self, documents: list[dict]) -> None:
        """
        Build a BM25 index from a list of {"id", "content", "metadata"} dicts
        (as returned by ChromaStore.get_all_documents()).
        """
        self._doc_ids = [d["id"] for d in documents]
        self._documents = [d["content"] for d in documents]
        self._metadatas = [d.get("metadata", {}) for d in documents]

        tokenized_corpus = [_tokenize(content) for content in self._documents]
        # BM25Okapi cannot be built on an empty corpus
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Return the top_k BM25 matches as RetrievalResult(source='bm25')."""
        if self._bm25 is None or not self._doc_ids:
            return []

        scores = self._bm25.get_scores(_tokenize(query))

        # Rank document indices by descending score
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[RetrievalResult] = []
        for idx in ranked[:top_k]:
            # Skip documents with zero relevance
            if scores[idx] <= 0:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=self._doc_ids[idx],
                    content=self._documents[idx],
                    metadata=self._metadatas[idx] or {},
                    score=float(scores[idx]),
                    source="bm25",
                )
            )
        return results
