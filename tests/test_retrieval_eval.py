"""Retrieval eval math: P@K, R@K, MRR on a fake retriever."""
from dataclasses import dataclass

import pytest

from evals.runners.retrieval_eval import evaluate_retrieval, score_single


@dataclass
class FakeResult:
    chunk_id: str


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, list[str]]):
        self._results = results_by_query

    async def retrieve(self, query: str, top_k: int = 5):
        return [FakeResult(cid) for cid in self._results[query][:top_k]]


def test_score_single_math():
    precision, recall, rrank = score_single(
        relevant=["a", "b"], retrieved=["a", "x", "b", "y", "z"], k=5
    )
    assert precision == pytest.approx(2 / 5)
    assert recall == pytest.approx(1.0)
    assert rrank == pytest.approx(1.0)  # first relevant at rank 1


def test_score_single_first_hit_at_rank_three():
    precision, recall, rrank = score_single(
        relevant=["c"], retrieved=["x", "y", "c", "z", "w"], k=5
    )
    assert precision == pytest.approx(1 / 5)
    assert recall == pytest.approx(1.0)
    assert rrank == pytest.approx(1 / 3)


def test_score_single_no_hits():
    precision, recall, rrank = score_single(relevant=["a"], retrieved=["x", "y"], k=5)
    assert precision == 0.0 and recall == 0.0 and rrank == 0.0


async def test_evaluate_retrieval_macro_averages():
    scenarios = [
        {"id": "s1", "query": "q1", "expected_retrieved_chunks": ["a", "b"]},
        {"id": "s2", "query": "q2", "expected_retrieved_chunks": ["c"]},
        {"id": "s3", "query": "q3", "expected_retrieved_chunks": []},  # skipped
    ]
    retriever = FakeRetriever({
        "q1": ["a", "x", "b", "y", "z"],
        "q2": ["x", "y", "c", "z", "w"],
    })

    report = await evaluate_retrieval(retriever, scenarios, k=5)

    assert report.scenario_count == 2
    assert report.precision_at_k == pytest.approx((2 / 5 + 1 / 5) / 2)
    assert report.recall_at_k == pytest.approx(1.0)
    assert report.mrr == pytest.approx((1.0 + 1 / 3) / 2)
    assert report.per_scenario[0]["scenario_id"] == "s1"


async def test_evaluate_retrieval_empty_when_no_ground_truth():
    report = await evaluate_retrieval(FakeRetriever({}), [{"id": "s", "query": "q", "expected_retrieved_chunks": []}])
    assert report.scenario_count == 0
    assert report.precision_at_k == 0.0 and report.mrr == 0.0
