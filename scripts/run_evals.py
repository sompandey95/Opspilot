#!/usr/bin/env python
"""Run the eval harness against the configured agent.

Prereqs: Azure OpenAI creds in .env, mock order service running
(`uvicorn mock_services.order_service.main:app --port 8001`), and — for
retrieval metrics — ChromaDB up with the KB ingested. Postgres/Redis are
optional (the run degrades to file-only reporting).

    python scripts/run_evals.py                       # full set, all judges
    python scripts/run_evals.py --subset 5            # first 5 scenarios
    python scripts/run_evals.py --category multi_step
    python scripts/run_evals.py --no-llm-judges       # deterministic metrics only
    python scripts/run_evals.py --no-retrieval        # skip Chroma-dependent metrics

Always run scripts/check_golden_dataset.py first (CI does).

CAUTION: the mock order service holds mutable in-memory state (refunds,
cancellations) that only resets on restart — running this twice in a row
against the same live instance means the second run sees post-refund/
post-cancel order state, not the pinned fixtures. Restart it
(`docker compose restart mock-order-service`) before each run when comparing
two prompt versions, or the diff will include stale-state noise alongside
real regressions.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCENARIOS_PATH = REPO_ROOT / "evals" / "golden_dataset" / "scenarios.json"
REPORT_DIR = REPO_ROOT / "evals" / "reports" / "runs"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, default=None, help="run only the first N scenarios")
    parser.add_argument("--category", default=None, help="run a single category")
    parser.add_argument("--no-llm-judges", action="store_true", help="skip GPT-4o judges")
    parser.add_argument("--no-retrieval", action="store_true", help="skip retrieval metrics")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_PATH)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    from app.config import get_settings
    from app.db.postgres import init_db
    from evals.reports.eval_report import to_markdown
    from evals.runners.eval_runner import load_scenarios

    settings = get_settings()
    if not (settings.AZURE_OPENAI_API_KEY and settings.AZURE_OPENAI_ENDPOINT):
        print("Azure OpenAI credentials missing — cannot run agent evals", file=sys.stderr)
        return 2

    try:
        await init_db()
    except Exception as exc:
        print(f"⚠ Postgres unavailable ({exc}) — eval_runs row will be skipped")

    from evals.runners.runner_factory import build_eval_runner

    runner = await build_eval_runner(
        settings,
        with_llm_judges=not args.no_llm_judges,
        with_retrieval=not args.no_retrieval,
    )
    scenarios = load_scenarios(args.scenarios)
    report = await runner.run(
        scenarios, subset=args.subset, category=args.category, report_dir=args.report_dir
    )

    print(to_markdown(report))
    if "report_path" in report:
        print(f"\nReport written to {report['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
