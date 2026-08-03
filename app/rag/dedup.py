"""Semantic deduplication of chunks at ingestion time.

When 80+ documents overlap (a FAQ restating a policy clause, a ticket
resolution quoting a FAQ), near-duplicate chunks make the retriever return
three copies of the same fact instead of three diverse chunks. We drop
near-duplicates before storage, keeping the most authoritative version.
"""
from __future__ import annotations

import numpy as np

from app.rag.chunker import Chunk

# Lower = more authoritative. Policies are canonical text; changelogs are the
# newest word on a topic; FAQs/tickets restate them and lose ties.
_AUTHORITY = {
    "policy": 0,
    "changelog": 1,
    "faq": 2,
    "api_doc": 3,
    "ticket": 4,
    "text": 5,
}


class SemanticDeduplicator:
    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold

    def deduplicate(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> tuple[list[Chunk], list[list[float]], list[tuple[str, str]]]:
        """
        Greedy dedup over cosine similarity of the already-computed embeddings.

        Chunks are considered in authority order (policy first, tickets last;
        longer content wins ties) so the kept copy is the most authoritative
        and complete one. Returns (kept_chunks, kept_embeddings, dropped) where
        dropped is a list of (dropped_id, kept_duplicate_id) pairs, and kept
        lists preserve the original input order.
        """
        if not chunks:
            return [], [], []
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        vecs = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        order = sorted(
            range(len(chunks)),
            key=lambda i: (
                _AUTHORITY.get(chunks[i].metadata.get("doc_type", ""), 9),
                -len(chunks[i].content),
            ),
        )

        kept_idx: list[int] = []
        dropped: list[tuple[str, str]] = []
        for i in order:
            if kept_idx:
                sims = vecs[kept_idx] @ vecs[i]
                best = int(np.argmax(sims))
                if float(sims[best]) > self.threshold:
                    dropped.append((chunks[i].id, chunks[kept_idx[best]].id))
                    continue
            kept_idx.append(i)

        kept_set = set(kept_idx)
        kept_chunks = [c for k, c in enumerate(chunks) if k in kept_set]
        kept_embeddings = [embeddings[k] for k in range(len(chunks)) if k in kept_set]
        return kept_chunks, kept_embeddings, dropped
