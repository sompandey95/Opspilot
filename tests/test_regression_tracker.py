"""Regression tracker: direction-aware diffs and markdown output."""
from evals.runners.regression_tracker import compare_reports, has_regressions, to_markdown


def report(version: str, **metrics) -> dict:
    return {"prompt_version": version, "metrics": metrics}


BASELINE = report(
    "v1",
    faithfulness_avg=0.92,
    hallucination_rate=0.03,
    tool_accuracy=0.90,
    avg_latency_ms=4000.0,
    avg_cost_inr=1.50,
)


def _delta(deltas, metric):
    return next(d for d in deltas if d.metric == metric)


def test_no_change_is_clean():
    deltas = compare_reports(BASELINE, report("v2", **BASELINE["metrics"]))
    assert not has_regressions(deltas)
    assert not any(d.improved for d in deltas)


def test_higher_is_better_regression():
    candidate = report("v2", **{**BASELINE["metrics"], "faithfulness_avg": 0.85})
    deltas = compare_reports(BASELINE, candidate)
    assert _delta(deltas, "faithfulness_avg").regressed
    assert has_regressions(deltas)


def test_lower_is_better_regression():
    candidate = report("v2", **{**BASELINE["metrics"], "hallucination_rate": 0.10})
    deltas = compare_reports(BASELINE, candidate)
    assert _delta(deltas, "hallucination_rate").regressed


def test_hallucination_drop_is_improvement():
    candidate = report("v2", **{**BASELINE["metrics"], "hallucination_rate": 0.0})
    deltas = compare_reports(BASELINE, candidate)
    d = _delta(deltas, "hallucination_rate")
    assert d.improved and not d.regressed


def test_latency_regression_is_relative_not_absolute():
    # +300ms on a 4000ms baseline is 7.5% — inside the 10% band, not a regression
    ok = report("v2", **{**BASELINE["metrics"], "avg_latency_ms": 4300.0})
    assert not has_regressions(compare_reports(BASELINE, ok))
    # +500ms is 12.5% — regression
    slow = report("v2", **{**BASELINE["metrics"], "avg_latency_ms": 4500.0})
    assert _delta(compare_reports(BASELINE, slow), "avg_latency_ms").regressed


def test_missing_metric_is_neither():
    candidate = report("v2", tool_accuracy=0.95)
    deltas = compare_reports(BASELINE, candidate)
    d = _delta(deltas, "faithfulness_avg")
    assert d.delta is None and not d.regressed


def test_markdown_shows_verdict_and_flags():
    candidate = report("v2", **{**BASELINE["metrics"], "tool_accuracy": 0.70})
    deltas = compare_reports(BASELINE, candidate)
    markdown = to_markdown(BASELINE, candidate, deltas)
    assert "`v1` → `v2`" in markdown
    assert "🔴 regression" in markdown
    assert "🔴 regressions detected" in markdown
