"""Golden-dataset consistency: the shipped dataset must pass the checker, and
the checker must actually catch each class of drift it exists to catch."""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "evals" / "golden_dataset"


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_golden_dataset", REPO_ROOT / "scripts" / "check_golden_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


@pytest.fixture
def dataset_copy(tmp_path):
    target = tmp_path / "golden_dataset"
    target.mkdir()
    for name in ("scenarios.json", "retrieval_ground_truth.json", "tool_call_ground_truth.json"):
        shutil.copy(DATASET_DIR / name, target / name)
    return target


def _edit_scenarios(dataset_dir: Path, mutate) -> None:
    path = dataset_dir / "scenarios.json"
    data = json.loads(path.read_text())
    mutate(data["scenarios"])
    path.write_text(json.dumps(data))


def test_shipped_dataset_is_consistent():
    assert checker.check() == []


def test_cli_exits_zero():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_golden_dataset.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_unknown_chunk_id_caught(dataset_copy):
    _edit_scenarios(
        dataset_copy,
        lambda s: s[0]["expected_retrieved_chunks"].append("faq_nonexistent_042"),
    )
    problems = checker.check(dataset_copy)
    assert any("faq_nonexistent_042" in p for p in problems)


def test_unknown_order_id_caught(dataset_copy):
    def mutate(scenarios):
        s = next(x for x in scenarios if x["id"] == "single_action_001")
        s["query"] = "Where is my order ORD-2024-31337?"

    _edit_scenarios(dataset_copy, mutate)
    problems = checker.check(dataset_copy)
    assert any("ORD-2024-31337" in p and "not found in mock seed" in p for p in problems)


def test_expects_missing_order_must_not_exist(dataset_copy):
    def mutate(scenarios):
        s = next(x for x in scenarios if x["id"] == "edge_case_001")
        s["query"] = "Where is my order ORD-2024-55001?"
        s["expected_tool_calls"] = []

    _edit_scenarios(dataset_copy, mutate)
    problems = checker.check(dataset_copy)
    assert any("expects a missing order" in p for p in problems)


def test_unknown_tool_and_bad_arg_caught(dataset_copy):
    def mutate(scenarios):
        scenarios[0]["expected_tool_calls"] = [
            {"tool": "teleport_order", "args": {}},
            {"tool": "check_order_status", "args": {"orderid": "ORD-2024-55001"}},
        ]

    _edit_scenarios(dataset_copy, mutate)
    problems = checker.check(dataset_copy)
    assert any("unknown tool 'teleport_order'" in p for p in problems)
    assert any("no argument 'orderid'" in p for p in problems)


def test_ground_truth_drift_caught(dataset_copy):
    gt_path = dataset_copy / "tool_call_ground_truth.json"
    data = json.loads(gt_path.read_text())
    data["ground_truth"]["multi_step_002"] = [{"tool": "check_order_status", "args": {}}]
    gt_path.write_text(json.dumps(data))

    problems = checker.check(dataset_copy)
    assert any("tool_call_ground_truth.json out of sync" in p for p in problems)


def test_duplicate_ids_caught(dataset_copy):
    _edit_scenarios(dataset_copy, lambda s: s.append(dict(s[0])))
    problems = checker.check(dataset_copy)
    assert any("duplicate scenario id" in p for p in problems)
