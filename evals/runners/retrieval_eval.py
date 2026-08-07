"""Retrieval quality: P@K, R@K, MRR against retrieval_ground_truth.

No LLM cost (embedding + rerank only) — safe to run freely while tuning the
retriever. Works on any retriever exposing `retrieve(query, top_k)` returning
objects with `.chunk_id` (HybridRetriever in prod, fakes in tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievalReport:
    k: int
    scenario_count: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    per_scenario: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "scenario_count": self.scenario_count,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "per_scenario": self.per_scenario,
        }


def score_single(relevant: list[str], retrieved: list[str], k: int) -> tuple[float, float, float]:
    """(P@K, R@K, reciprocal rank) for one query."""
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_set)

    precision = hits / k if k else 0.0
    recall = hits / len(relevant_set) if relevant_set else 0.0
    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(top_k, start=1):
        if chunk_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break
    return precision, recall, reciprocal_rank


async def evaluate_retrieval(retriever, scenarios: list[dict], k: int = 5) -> RetrievalReport:
    """Runs every scenario that has expected_retrieved_chunks."""
    per_scenario: list[dict] = []
    precisions: list[float] = []
    recalls: list[float] = []
    rranks: list[float] = []

    for scenario in scenarios:
        relevant = scenario.get("expected_retrieved_chunks") or []
        if not relevant:
            continue
        results = await retriever.retrieve(scenario["query"], top_k=k)
        retrieved = [r.chunk_id for r in results]
        precision, recall, rrank = score_single(relevant, retrieved, k)
        precisions.append(precision)
        recalls.append(recall)
        rranks.append(rrank)
        per_scenario.append(
            {
                "scenario_id": scenario["id"],
                "precision_at_k": precision,
                "recall_at_k": recall,
                "reciprocal_rank": rrank,
                "retrieved": retrieved,
                "relevant": relevant,
            }
        )

    n = len(per_scenario)
    return RetrievalReport(
        k=k,
        scenario_count=n,
        precision_at_k=sum(precisions) / n if n else 0.0,
        recall_at_k=sum(recalls) / n if n else 0.0,
        mrr=sum(rranks) / n if n else 0.0,
        per_scenario=per_scenario,
    )
