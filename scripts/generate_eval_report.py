#!/usr/bin/env python
"""Render an eval-run report JSON as markdown (for PR comments / terminals).

    python scripts/generate_eval_report.py evals/reports/runs/<run>.json
    python scripts/generate_eval_report.py --latest
    python scripts/generate_eval_report.py --compare baseline.json candidate.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals.reports.eval_report import to_markdown  # noqa: E402
from evals.runners.regression_tracker import (  # noqa: E402
    compare_reports,
    has_regressions,
    load_report,
)
from evals.runners.regression_tracker import to_markdown as diff_to_markdown  # noqa: E402

RUNS_DIR = REPO_ROOT / "evals" / "reports" / "runs"


def latest_report_path() -> Path | None:
    runs = sorted(RUNS_DIR.glob("*.json"))
    return runs[-1] if runs else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, help="path to a run report JSON")
    parser.add_argument("--latest", action="store_true", help="use the newest report in evals/reports/runs")
    parser.add_argument(
        "--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"),
        help="diff two run reports; exits 1 on regressions",
    )
    args = parser.parse_args()

    if args.compare:
        baseline, candidate = (load_report(p) for p in args.compare)
        deltas = compare_reports(baseline, candidate)
        print(diff_to_markdown(baseline, candidate, deltas))
        return 1 if has_regressions(deltas) else 0

    path = args.report
    if args.latest or path is None:
        path = latest_report_path()
        if path is None:
            print("No run reports found in evals/reports/runs", file=sys.stderr)
            return 2

    print(to_markdown(load_report(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
