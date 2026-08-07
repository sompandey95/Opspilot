"""Render an eval-run report as the markdown table posted on PRs."""
from __future__ import annotations


def _fmt(value, kind: str = "score") -> str:
    if value is None:
        return "—"
    if kind == "score":
        return f"{value:.3f}"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "ms":
        return f"{value:,.0f} ms"
    if kind == "inr":
        return f"₹{value:.4f}"
    return str(value)


def to_markdown(report: dict) -> str:
    m = report["metrics"]
    gate = report.get("ci_gate", {})
    verdict = "✅ PASS" if gate.get("passed") else "❌ FAIL"

    lines = [
        f"## OpsPilot eval — prompt `{report.get('prompt_version')}` · "
        f"model `{report.get('model')}` · {m['total_scenarios']} scenarios · gate {verdict}",
        "",
        "| Metric | Value | Threshold | |",
        "|---|---|---|---|",
    ]

    thresholds = {c["metric"]: c for c in gate.get("checks", [])}

    def row(label: str, metric: str, kind: str = "score") -> None:
        check = thresholds.get(metric)
        threshold = "—"
        status = ""
        if check is not None:
            threshold = _fmt(check["threshold"], "score")
            if check["passed"] is None:
                status = "⏭ skipped"
            else:
                status = "✅" if check["passed"] else "❌"
        lines.append(f"| {label} | {_fmt(m.get(metric), kind)} | {threshold} | {status} |")

    row("Faithfulness", "faithfulness_avg")
    row("Hallucination rate", "hallucination_rate", "pct")
    row("Tool accuracy", "tool_accuracy")
    row("Retrieval P@K", "retrieval_precision")
    row("Retrieval R@K", "retrieval_recall", "score")
    row("Retrieval MRR", "retrieval_mrr")
    row("Relevance", "relevance_avg")
    row("Intent accuracy", "intent_accuracy")
    row("HITL correctness", "hitl_accuracy")
    row("Adversarial containment", "adversarial_pass_rate", "pct")
    row("Avg latency", "avg_latency_ms", "ms")
    row("Avg cost / query", "avg_cost_inr", "inr")

    if m.get("errors"):
        lines.append("")
        lines.append(f"⚠️ {m['errors']} scenario(s) errored during the run.")

    by_category = report.get("by_category") or {}
    if by_category:
        lines += [
            "",
            "<details><summary>Per-category breakdown</summary>",
            "",
            "| Category | n | Tool acc | Halluc. | Faithfulness |",
            "|---|---|---|---|---|",
        ]
        for cat in sorted(by_category):
            c = by_category[cat]
            lines.append(
                f"| {cat} | {c['count']} | {_fmt(c.get('tool_accuracy'))} | "
                f"{_fmt(c.get('hallucination_rate'), 'pct')} | {_fmt(c.get('faithfulness_avg'))} |"
            )
        lines += ["", "</details>"]

    failed = [
        s["scenario_id"]
        for s in report.get("scenarios", [])
        if s.get("error")
        or s.get("hallucinated")
        or (s.get("tool_accuracy") is not None and s["tool_accuracy"] < 1.0)
        or s.get("adversarial_pass") is False
        or s.get("hitl_correct") is False
    ]
    if failed:
        lines += ["", f"**Scenarios needing attention:** {', '.join(sorted(set(failed)))}"]

    return "\n".join(lines)
