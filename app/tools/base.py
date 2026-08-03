"""Abstract Tool interface: every tool declares its risk profile up front.

- risk_level drives the HITL gate (Phase 5): HIGH tools block for approval.
- is_state_changing drives idempotency protection in the agent loop (Phase 4).
- parameters is a JSON schema used both for the LLM tool spec and for
  argument validation (app/guardrails/schemas.py).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolExecutionError(Exception):
    """Raised for unrecoverable tool failures the agent should observe."""


@dataclass
class ToolResult:
    success: bool
    data: dict | list | None = None
    error: str | None = None
    from_cache: bool = field(default=False, compare=False)

    def to_message(self) -> str:
        """Serialise for the LLM 'tool' role message."""
        if self.success:
            return json.dumps({"success": True, "data": self.data}, default=str)
        return json.dumps({"success": False, "error": self.error})

    def to_json(self) -> str:
        return json.dumps(
            {"success": self.success, "data": self.data, "error": self.error},
            default=str,
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> "ToolResult":
        payload = json.loads(raw)
        return cls(
            success=payload["success"],
            data=payload.get("data"),
            error=payload.get("error"),
            from_cache=True,
        )


class Tool(ABC):
    name: str
    description: str
    risk_level: RiskLevel = RiskLevel.NONE
    is_state_changing: bool = False
    parameters: dict = {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        ...

    def get_schema(self) -> dict:
        """OpenAI-compatible tool/function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
