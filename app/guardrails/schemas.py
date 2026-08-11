"""Structured-output validation for tool call arguments.

The agent validates every LLM-produced tool call against the tool's JSON
schema before execution; on failure the error is fed back as a tool message so
the agent can correct itself instead of crashing on bad JSON.
"""
from __future__ import annotations

from dataclasses import dataclass

from jsonschema import Draft202012Validator

from app.tools.registry import ToolRegistry


@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None


class SchemaValidator:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._validators: dict[str, Draft202012Validator] = {}

    def _validator_for(self, tool_name: str) -> Draft202012Validator | None:
        if tool_name not in self._validators:
            tool = self._registry.get(tool_name)
            if tool is None:
                return None
            self._validators[tool_name] = Draft202012Validator(tool.parameters)
        return self._validators[tool_name]

    def validate(self, tool_name: str, args: dict | None) -> ValidationResult:
        validator = self._validator_for(tool_name)
        if validator is None:
            available = ", ".join(sorted(t.name for t in self._registry.all()))
            return ValidationResult(
                valid=False,
                error=f"Unknown tool '{tool_name}'. Available tools: {available}",
            )

        errors = sorted(validator.iter_errors(args or {}), key=lambda e: list(e.path))
        if errors:
            messages = []
            for err in errors[:3]:
                location = ".".join(str(p) for p in err.path) or "(root)"
                messages.append(f"{location}: {err.message}")
            return ValidationResult(valid=False, error="; ".join(messages))

        return ValidationResult(valid=True)
