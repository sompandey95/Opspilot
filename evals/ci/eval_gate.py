#!/usr/bin/env python
"""CI gate: read an eval-run report JSON, exit 1 if any threshold breached.

The report already carries a computed `ci_gate` (see evals/runners/eval_runner.py
EvalRunner._ci_gate) — this script just reads it, prints the same markdown table
posted on PRs, and turns the verdict into a process exit code.

    python evals/ci/eval_gate.py evals/reports/runs/<run>.json
    python evals/ci/eval_gate.py --latest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

RUNS_DIR = REPO_ROOT / "evals" / "reports" / "runs"


def latest_report_path() -> Path | None:
    runs = sorted(RUNS_DIR.glob("*.json"))
    return runs[-1] if runs else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, help="path to a run report JSON")
    parser.add_argument("--latest", action="store_true", help="use the newest report in evals/reports/runs")
    args = parser.parse_args(argv)

    from evals.reports.eval_report import to_markdown
    from evals.runners.regression_tracker import load_report

    path = args.report
    if args.latest or path is None:
        path = latest_report_path()
        if path is None:
            print("No run reports found in evals/reports/runs", file=sys.stderr)
            return 2

    if not path.exists():
        print(f"Report not found: {path}", file=sys.stderr)
        return 2

    report = load_report(path)
    print(to_markdown(report))

    gate = report.get("ci_gate", {})
    if gate.get("skipped"):
        print(f"\n⏭ skipped (no judge/data available): {', '.join(gate['skipped'])}")

    if not gate.get("passed"):
        failed = [c["metric"] for c in gate.get("checks", []) if c["passed"] is False]
        print(f"\n❌ eval gate FAILED — breached: {', '.join(failed) or 'unknown'}", file=sys.stderr)
        return 1

    print("\n✅ eval gate PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
