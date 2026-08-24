"""Diff two eval runs (report JSONs) across prompt versions.

Direction-aware: higher is better for accuracy-style metrics, lower is better
for hallucination rate, latency, and cost. A change beyond EPSILON in the
wrong direction is a regression.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

EPSILON = 0.01

# metric -> higher_is_better
TRACKED_METRICS: dict[str, bool] = {
    "faithfulness_avg": True,
    "hallucination_rate": False,
    "tool_accuracy": True,
    "relevance_avg": True,
    "intent_accuracy": True,
    "hitl_accuracy": True,
    "adversarial_pass_rate": True,
    "retrieval_precision": True,
    "retrieval_recall": True,
    "retrieval_mrr": True,
    "avg_latency_ms": False,
    "avg_cost_inr": False,
}


@dataclass
class MetricDelta:
    metric: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    regressed: bool
    improved: bool


def load_report(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def compare_reports(baseline: dict, candidate: dict) -> list[MetricDelta]:
    deltas: list[MetricDelta] = []
    base_metrics = baseline.get("metrics", {})
    cand_metrics = candidate.get("metrics", {})

    for metric, higher_is_better in TRACKED_METRICS.items():
        base = base_metrics.get(metric)
        cand = cand_metrics.get(metric)
        if base is None or cand is None:
            deltas.append(MetricDelta(metric, base, cand, None, False, False))
            continue
        delta = cand - base
        # Latency/cost regressions are judged relatively (10%), scores absolutely
        threshold = abs(base) * 0.10 if metric in ("avg_latency_ms", "avg_cost_inr") else EPSILON
        if higher_is_better:
            regressed, improved = delta < -threshold, delta > threshold
        else:
            regressed, improved = delta > threshold, delta < -threshold
        deltas.append(MetricDelta(metric, base, cand, delta, regressed, improved))
    return deltas


def has_regressions(deltas: list[MetricDelta]) -> bool:
    return any(d.regressed for d in deltas)


def _format_metric(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    return str(value)


def to_markdown(baseline: dict, candidate: dict, deltas: list[MetricDelta]) -> str:
    lines = [
        f"## Eval regression: `{baseline.get('prompt_version')}` → `{candidate.get('prompt_version')}`",
        "",
        "| Metric | Baseline | Candidate | Δ | |",
        "|---|---|---|---|---|",
    ]
    for d in deltas:
        if d.baseline is None and d.candidate is None:
            continue
        flag = "🔴 regression" if d.regressed else ("🟢 improved" if d.improved else "")
        delta_str = "—" if d.delta is None else f"{d.delta:+.4f}"
        lines.append(
            f"| {d.metric} | {_format_metric(d.baseline)} | "
            f"{_format_metric(d.candidate)} | {delta_str} | {flag} |"
        )
    lines.append("")
    lines.append(
        "**Verdict:** " + ("🔴 regressions detected" if has_regressions(deltas) else "🟢 no regressions")
    )
    return "\n".join(lines)
