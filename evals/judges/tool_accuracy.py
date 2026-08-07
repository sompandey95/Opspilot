"""Tool-call accuracy — pure Python, no LLM.

Compares the agent's actual business-tool calls against a scenario's ground
truth. `search_knowledge` is excluded on both sides by design: retrieval
quality is scored by the retrieval eval, and expected_retrieved_chunks already
encodes whether retrieval should happen.

Matching: expected calls are matched greedily, in order, against unmatched
actual calls with the same tool name.
- exact:   name matches and every ground-truth arg matches (extra actual args
           are fine — ground truth lists only the args that matter)
- partial: name matches but at least one ground-truth arg differs
- missing: expected call with no actual call of that name left
- extra:   actual business calls no expected call claimed

score = (exact + 0.5·partial) / (exact + partial + missing + extra)
Both sides empty ⇒ 1.0 (correctly doing nothing is a pass).
"""
from __future__ import annotations

from dataclasses import dataclass, field

IGNORED_TOOLS = {"search_knowledge"}


@dataclass
class ToolAccuracyResult:
    exact: int = 0
    partial: int = 0
    missing: int = 0
    extra: int = 0
    details: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        total = self.exact + self.partial + self.missing + self.extra
        if total == 0:
            return 1.0
        return (self.exact + 0.5 * self.partial) / total


def _args_match(expected_args: dict, actual_args: dict) -> bool:
    return all(actual_args.get(k) == v for k, v in expected_args.items())


def score_tool_calls(expected: list[dict], actual: list[dict]) -> ToolAccuracyResult:
    """expected: [{"tool": name, "args": {...}}]; actual: same shape (from the
    trace). Order-insensitive apart from greedy first-match."""
    result = ToolAccuracyResult()
    remaining = [
        {"tool": a.get("tool"), "args": a.get("args") or {}}
        for a in actual
        if a.get("tool") not in IGNORED_TOOLS
    ]

    for exp in expected:
        name, args = exp.get("tool"), exp.get("args") or {}
        if name in IGNORED_TOOLS:
            continue
        same_name = [a for a in remaining if a["tool"] == name]
        if not same_name:
            result.missing += 1
            result.details.append(f"missing: {name}({args})")
            continue
        exact = next((a for a in same_name if _args_match(args, a["args"])), None)
        if exact is not None:
            result.exact += 1
            remaining.remove(exact)
        else:
            result.partial += 1
            chosen = same_name[0]
            remaining.remove(chosen)
            result.details.append(f"partial: {name} expected {args}, got {chosen['args']}")

    for leftover in remaining:
        result.extra += 1
        result.details.append(f"extra: {leftover['tool']}({leftover['args']})")

    return result
