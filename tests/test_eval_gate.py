"""CI gate: exit code follows the report's precomputed ci_gate.passed."""
import json

from evals.ci import eval_gate


def _write_report(tmp_path, passed: bool, name="run.json"):
    report = {
        "prompt_version": "v1",
        "model": "gpt-5.4",
        "metrics": {
            "total_scenarios": 30,
            "faithfulness_avg": 0.95 if passed else 0.5,
            "hallucination_rate": 0.0,
            "tool_accuracy": 0.9,
            "retrieval_precision": 0.9,
        },
        "by_category": {},
        "scenarios": [],
        "ci_gate": {
            "passed": passed,
            "checks": [
                {"metric": "faithfulness_avg", "value": 0.95 if passed else 0.5, "threshold": 0.90, "passed": passed},
                {"metric": "hallucination_rate", "value": 0.0, "threshold": 0.05, "passed": True},
                {"metric": "tool_accuracy", "value": 0.9, "threshold": 0.85, "passed": True},
                {"metric": "retrieval_precision", "value": 0.9, "threshold": 0.85, "passed": True},
            ],
            "skipped": [],
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(report))
    return path


def test_gate_passes_on_passing_report(tmp_path, capsys):
    path = _write_report(tmp_path, passed=True)
    code = eval_gate.main([str(path)])
    assert code == 0
    assert "PASSED" in capsys.readouterr().out


def test_gate_fails_on_breached_threshold(tmp_path, capsys):
    path = _write_report(tmp_path, passed=False)
    code = eval_gate.main([str(path)])
    assert code == 1
    out, err = capsys.readouterr()
    assert "faithfulness_avg" in err


def test_gate_reports_missing_file(tmp_path, capsys):
    code = eval_gate.main([str(tmp_path / "nope.json")])
    assert code == 2


def test_gate_no_reports_found(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_gate, "RUNS_DIR", tmp_path)
    code = eval_gate.main(["--latest"])
    assert code == 2


def test_gate_latest_picks_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_gate, "RUNS_DIR", tmp_path)
    _write_report(tmp_path, passed=False, name="20260101T000000Z_v1.json")
    _write_report(tmp_path, passed=True, name="20260102T000000Z_v2.json")
    code = eval_gate.main(["--latest"])
    assert code == 0
