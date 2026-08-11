#!/usr/bin/env python
"""Golden-dataset consistency checker — run before evals (and in CI).

Guards the cross-phase invariant: mock seed data, chunker output, and the
golden dataset must agree. Checks:

- scenario schema: required fields, unique IDs, valid categories/intents/languages
- every order ID referenced (queries + tool args) exists in the mock seed —
  except scenarios flagged `expects_missing_order`, whose IDs must NOT exist
- every expected chunk ID exists in the chunker's output over knowledge_base/
- every expected tool exists in the registry and its args use real schema keys
- retrieval_ground_truth.json / tool_call_ground_truth.json exactly mirror
  scenarios.json (they are derived files — drift means someone hand-edited)

Exit 0 = consistent; exit 1 with per-scenario messages otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.config import Settings  # noqa: E402
from app.rag.chunker import SmartChunker  # noqa: E402
from app.tools.registry import build_default_registry  # noqa: E402
from mock_services.order_service.seed import build_store  # noqa: E402

DATASET_DIR = REPO_ROOT / "evals" / "golden_dataset"
SCENARIOS_PATH = DATASET_DIR / "scenarios.json"
RETRIEVAL_GT_PATH = DATASET_DIR / "retrieval_ground_truth.json"
TOOL_GT_PATH = DATASET_DIR / "tool_call_ground_truth.json"

VALID_CATEGORIES = {
    "faq_en", "faq_mixed", "single_action", "multi_step", "adversarial",
    "stale_knowledge", "out_of_scope", "edge_cases", "angry",
}
VALID_INTENTS = {"faq", "action_simple", "action_complex", "escalate", "out_of_scope", None}
VALID_LANGUAGES = {"en", "hi", "mixed"}
REQUIRED_FIELDS = [
    "id", "category", "query", "language", "expected_intent",
    "expected_retrieved_chunks", "expected_tool_calls", "reference_answer",
    "expected_hitl", "adversarial", "notes",
]

ORDER_ID_RE = re.compile(r"\bORD-\d{4}-\d{4,6}\b", re.IGNORECASE)


def check(dataset_dir: Path = DATASET_DIR) -> list[str]:
    """Return a list of problems (empty = consistent)."""
    problems: list[str] = []

    data = json.loads((dataset_dir / "scenarios.json").read_text())
    scenarios = data.get("scenarios", [])
    if not scenarios:
        return ["scenarios.json contains no scenarios"]

    # Reference universes
    _, orders = build_store()
    chunk_ids = {c.id for c in SmartChunker().chunk_directory(REPO_ROOT / "knowledge_base")}
    registry = build_default_registry(Settings(_env_file=None))

    seen_ids: set[str] = set()
    for s in scenarios:
        sid = s.get("id", "<missing id>")
        where = f"[{sid}]"

        for field in REQUIRED_FIELDS:
            if field not in s:
                problems.append(f"{where} missing field '{field}'")
        if sid in seen_ids:
            problems.append(f"{where} duplicate scenario id")
        seen_ids.add(sid)

        if s.get("category") not in VALID_CATEGORIES:
            problems.append(f"{where} invalid category {s.get('category')!r}")
        if s.get("expected_intent") not in VALID_INTENTS:
            problems.append(f"{where} invalid expected_intent {s.get('expected_intent')!r}")
        if s.get("language") not in VALID_LANGUAGES:
            problems.append(f"{where} invalid language {s.get('language')!r}")

        # Order IDs: from the query and from tool args
        referenced = set(ORDER_ID_RE.findall(s.get("query", "")))
        for call in s.get("expected_tool_calls", []):
            for value in call.get("args", {}).values():
                if isinstance(value, str):
                    referenced.update(ORDER_ID_RE.findall(value))
        expects_missing = s.get("expects_missing_order", False)
        for order_id in sorted(referenced):
            exists = order_id.upper() in orders
            if expects_missing and exists:
                problems.append(
                    f"{where} order {order_id} exists in the seed but the scenario "
                    "expects a missing order"
                )
            if not expects_missing and not exists:
                problems.append(f"{where} order {order_id} not found in mock seed")

        # Chunk IDs
        for chunk_id in s.get("expected_retrieved_chunks", []):
            if chunk_id not in chunk_ids:
                problems.append(f"{where} chunk '{chunk_id}' not produced by the chunker")

        # Tools + arg keys
        for call in s.get("expected_tool_calls", []):
            tool = registry.get(call.get("tool", ""))
            if tool is None:
                problems.append(f"{where} unknown tool '{call.get('tool')}'")
                continue
            valid_keys = set(tool.parameters.get("properties", {}))
            for key in call.get("args", {}):
                if key not in valid_keys:
                    problems.append(
                        f"{where} tool '{tool.name}' has no argument '{key}' "
                        f"(valid: {sorted(valid_keys)})"
                    )

    # Derived ground-truth files must mirror scenarios.json exactly
    problems += _check_derived(
        scenarios,
        dataset_dir / "retrieval_ground_truth.json",
        "expected_retrieved_chunks",
        "retrieval_ground_truth.json",
    )
    problems += _check_derived(
        scenarios,
        dataset_dir / "tool_call_ground_truth.json",
        "expected_tool_calls",
        "tool_call_ground_truth.json",
    )

    return problems


def _check_derived(scenarios: list[dict], path: Path, field: str, label: str) -> list[str]:
    if not path.exists():
        return [f"{label} is missing"]
    ground_truth = json.loads(path.read_text()).get("ground_truth", {})
    expected = {s["id"]: s[field] for s in scenarios if s.get(field)}
    problems = []
    for sid in expected.keys() | ground_truth.keys():
        if expected.get(sid) != ground_truth.get(sid):
            problems.append(
                f"[{sid}] {label} out of sync with scenarios.json "
                f"(expected {expected.get(sid)}, found {ground_truth.get(sid)})"
            )
    return problems


def main() -> int:
    problems = check()
    data = json.loads(SCENARIOS_PATH.read_text())
    scenarios = data["scenarios"]

    from collections import Counter

    counts = Counter(s.get("category") for s in scenarios)
    print(f"{len(scenarios)} scenarios: " + ", ".join(f"{c}={n}" for c, n in sorted(counts.items())))

    if problems:
        print(f"\nFAIL — {len(problems)} problem(s):")
        for p in problems:
            print(f"  ✗ {p}")
        return 1
    print("OK — golden dataset is consistent with the mock seed and chunker output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
