"""Per-request trace for model, tool, guardrail, and approval activity.

Persistence is best effort: a database failure must not suppress an otherwise
valid customer response. Metrics aggregate the persisted rows.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from app.llm.client import LLMResponse

logger = logging.getLogger(__name__)

_MAX_RESULT_CHARS = 1500  # keep steps JSONB bounded

_INSERT_SQL = """
INSERT INTO traces (
    id, session_id, query, intent, model, steps, response, confidence,
    total_latency_ms, total_tokens, input_tokens, output_tokens,
    hitl_triggered, escalated, prompt_version, guardrail_flags, cost_inr
) VALUES (
    $1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11, $12, $13, $14, $15,
    $16::jsonb, $17
)
"""


@dataclass
class Trace:
    query: str
    intent: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    session_id: str | None = None
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: list[dict] = field(default_factory=list)
    response: str | None = None
    confidence: float | None = None
    escalated: bool = False
    hitl_triggered: bool = False
    guardrail_flags: list[str] = field(default_factory=list)
    cost_inr: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    _started: float = field(default_factory=time.perf_counter, repr=False)

    # ------------------------------------------------------------------ #
    # Step recording                                                       #
    # ------------------------------------------------------------------ #

    def add_llm_step(self, step: int, response: LLMResponse) -> None:
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.steps.append(
            {
                "step": step,
                "type": "llm",
                "thought": response.content,
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments, "parse_error": tc.parse_error}
                    for tc in response.tool_calls
                ],
                "model": response.model,
                "finish_reason": response.finish_reason,
                "latency_ms": response.latency_ms,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        )

    def add_classifier_usage(self, usage) -> None:
        if usage is not None:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens

    def add_validation_failure(self, step: int, tool_name: str, error: str) -> None:
        self.steps.append(
            {"step": step, "type": "validation_error", "tool": tool_name, "error": error}
        )

    def add_hitl_decision(
        self, step: int, tool_name: str, status: str, reason: str | None = None
    ) -> None:
        self.hitl_triggered = True
        self.steps.append(
            {"step": step, "type": "hitl", "tool": tool_name, "status": status, "reason": reason}
        )

    def add_tool_result(
        self,
        step: int,
        tool_name: str,
        args: dict,
        result,
        latency_ms: int,
        retrieval_scores: list[float] | None = None,
    ) -> None:
        data_preview = None
        if result.data is not None:
            data_preview = json.dumps(result.data, default=str)[:_MAX_RESULT_CHARS]
        entry: dict = {
            "step": step,
            "type": "tool_result",
            "tool": tool_name,
            "arguments": args,
            "success": result.success,
            "error": result.error,
            "from_cache": result.from_cache,
            "data_preview": data_preview,
            "latency_ms": latency_ms,
        }
        if retrieval_scores is not None:
            entry["retrieval_scores"] = retrieval_scores
        self.steps.append(entry)

    def add_tool_error(self, step: int, tool_name: str, args: dict, error: str) -> None:
        self.steps.append(
            {"step": step, "type": "tool_error", "tool": tool_name, "arguments": args, "error": error}
        )

    def add_escalation(self, reason: str, detail: str | float | int | None = None) -> None:
        self.escalated = True
        self.steps.append({"type": "escalation", "reason": reason, "detail": detail})

    def add_guardrail_flags(self, flags: list[str]) -> None:
        self.guardrail_flags.extend(flags)

    def set_confidence(self, confidence: float) -> None:
        self.confidence = round(confidence, 3)

    def set_response(self, response: str) -> None:
        self.response = response

    # ------------------------------------------------------------------ #
    # Aggregates                                                           #
    # ------------------------------------------------------------------ #

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_latency_ms(self) -> int:
        return int((time.perf_counter() - self._started) * 1000)

    # ------------------------------------------------------------------ #
    # Persistence (best-effort)                                            #
    # ------------------------------------------------------------------ #

    async def persist(self) -> None:
        from app.db.postgres import execute

        session_uuid = None
        if self.session_id:
            try:
                session_uuid = uuid.UUID(self.session_id)
            except ValueError:
                pass

        try:
            await execute(
                _INSERT_SQL,
                uuid.UUID(self.trace_id),
                session_uuid,
                self.query,
                self.intent,
                self.model,
                json.dumps(self.steps, default=str),
                self.response,
                self.confidence,
                self.total_latency_ms,
                self.total_tokens,
                self.input_tokens,
                self.output_tokens,
                self.hitl_triggered,
                self.escalated,
                self.prompt_version,
                json.dumps(self.guardrail_flags) if self.guardrail_flags else None,
                self.cost_inr,
            )
        except Exception as exc:
            logger.error("Failed to persist trace %s: %s", self.trace_id, exc)
