"""ReAct loop: thought → action → observation → repeat.

Safety properties baked in:
- max MAX_AGENT_STEPS iterations, per-step LLM timeout
- every tool call validated against its JSON schema; malformed calls are fed
  back as tool errors so the agent self-corrects instead of crashing
- HITL gate before every risky tool execution: app.hitl.gate.HITLApprovalGate
  applies the risk matrix (HIGH blocks for human approval); AutoApproveHITLGate
  remains as the no-DB fallback
- idempotency for state-changing tools via Redis (prevents double refunds)
- low confidence or exhausted steps ⇒ escalate, never guess
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Protocol

from app.agent.confidence import ConfidenceScorer
from app.agent.intent_classifier import IntentResult, IntentType
from app.agent.model_router import ModelRouter
from app.agent.prompt_loader import SystemPrompt, load_current_prompt
from app.config import Settings
from app.guardrails.schemas import SchemaValidator
from app.llm.client import LLMClient, ToolCallRequest
from app.observability.trace import Trace
from app.tools.base import RiskLevel, Tool, ToolExecutionError, ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

ESCALATION_MESSAGE = (
    "I'm having trouble resolving this myself, so I'm connecting you with a "
    "human support specialist who can help. They'll have the full context of "
    "your request."
)
LOW_CONFIDENCE_MESSAGE = (
    "I'm not fully confident in my answer, so I'm connecting you with a human "
    "agent to make sure you get this resolved correctly."
)
PENDING_APPROVAL_MESSAGE = (
    "This action needs supervisor approval, which is pending. "
    "You'll be notified as soon as it's processed."
)

_IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60


# --------------------------------------------------------------------- #
# HITL gate seam                                                          #
# --------------------------------------------------------------------- #

class ApprovalStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class ApprovalDecision:
    status: ApprovalStatus
    reason: str | None = None


class HITLGate(Protocol):
    async def request_approval(
        self, tool: Tool, args: dict, trace: Trace
    ) -> ApprovalDecision:
        ...


class AutoApproveHITLGate:
    """Fallback gate used when the real one (app.hitl.gate.HITLApprovalGate)
    can't be built — e.g. tests, or Postgres missing at startup. Auto-approves
    everything but still records the decision on the trace."""

    async def request_approval(
        self, tool: Tool, args: dict, trace: Trace
    ) -> ApprovalDecision:
        if tool.risk_level == RiskLevel.HIGH:
            logger.warning(
                "AutoApproveHITLGate approving high-risk tool '%s' without human review",
                tool.name,
            )
        return ApprovalDecision(status=ApprovalStatus.APPROVED, reason="phase4_stub_auto_approve")


class _PendingApproval(Exception):
    """Control-flow signal: an approval timed out; surface a pending response."""


# --------------------------------------------------------------------- #
# Agent                                                                   #
# --------------------------------------------------------------------- #

@dataclass
class AgentResult:
    response: str
    trace: Trace
    confidence: float | None = None
    escalated: bool = False
    pending_approval: bool = False


class ReActAgent:
    def __init__(
        self,
        llm: LLMClient,
        tool_registry: ToolRegistry,
        schema_validator: SchemaValidator,
        settings: Settings,
        hitl_gate: HITLGate | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
        redis_client=None,
        system_prompt: SystemPrompt | None = None,
    ) -> None:
        self._llm = llm
        self._registry = tool_registry
        self._validator = schema_validator
        self._settings = settings
        self._hitl_gate: HITLGate = hitl_gate or AutoApproveHITLGate()
        self._confidence = confidence_scorer or ConfidenceScorer(settings)
        self._redis = redis_client
        self._prompt = system_prompt or load_current_prompt()

    @property
    def prompt_version(self) -> str:
        return self._prompt.version

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        query: str,
        intent: IntentResult,
        session_id: str | None = None,
        history: list[dict] | None = None,
    ) -> AgentResult:
        route = ModelRouter.get_route(intent.intent)
        trace = Trace(
            query=query,
            intent=intent.intent.value,
            prompt_version=self._prompt.version,
            session_id=session_id,
        )
        trace.add_classifier_usage(intent.usage)

        if route.role is None:  # ESCALATE — no LLM needed
            trace.add_escalation("intent_escalate", intent.reasoning)
            return AgentResult(response=ESCALATION_MESSAGE, trace=trace, escalated=True)

        messages = self._build_messages(query, intent, history)
        tool_schemas = self._registry.get_tool_schemas()
        step_timeout = float(self._settings.AGENT_TIMEOUT_SECONDS)

        for step in range(self._settings.MAX_AGENT_STEPS):
            try:
                response = await self._llm.complete(
                    role=route.role,
                    messages=messages,
                    tools=tool_schemas,
                    max_completion_tokens=route.max_completion_tokens,
                    timeout=step_timeout,
                )
            except Exception as exc:
                logger.error("LLM step %d failed: %s", step, exc)
                trace.add_escalation("llm_error", str(exc))
                trace.set_response(ESCALATION_MESSAGE)
                return AgentResult(response=ESCALATION_MESSAGE, trace=trace, escalated=True)

            trace.model = response.model
            trace.add_llm_step(step, response)

            # No tool call ⇒ final answer
            if not response.has_tool_calls:
                return self._finalise(query, response.content or "", trace)

            messages.append(response.to_assistant_message())

            try:
                for tool_call in response.tool_calls:
                    observation = await self._handle_tool_call(step, tool_call, trace)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": observation,
                        }
                    )
            except _PendingApproval:
                trace.set_response(PENDING_APPROVAL_MESSAGE)
                return AgentResult(
                    response=PENDING_APPROVAL_MESSAGE, trace=trace, pending_approval=True
                )

        trace.add_escalation("max_steps_reached", self._settings.MAX_AGENT_STEPS)
        trace.set_response(ESCALATION_MESSAGE)
        return AgentResult(response=ESCALATION_MESSAGE, trace=trace, escalated=True)

    # ------------------------------------------------------------------ #
    # Final answer + confidence                                            #
    # ------------------------------------------------------------------ #

    def _finalise(self, query: str, answer: str, trace: Trace) -> AgentResult:
        confidence = self._confidence.score(query, answer, trace)
        trace.set_confidence(confidence)

        # A polite scope-decline is the correct terminal state, not a shaky
        # answer — escalating it to a human would page someone for nothing.
        skip_gate = trace.intent == IntentType.OUT_OF_SCOPE.value

        if not skip_gate and confidence < self._settings.CONFIDENCE_THRESHOLD:
            trace.add_escalation("low_confidence", confidence)
            trace.set_response(LOW_CONFIDENCE_MESSAGE)
            return AgentResult(
                response=LOW_CONFIDENCE_MESSAGE,
                trace=trace,
                confidence=confidence,
                escalated=True,
            )

        trace.set_response(answer)
        return AgentResult(response=answer, trace=trace, confidence=confidence)

    # ------------------------------------------------------------------ #
    # Tool handling                                                        #
    # ------------------------------------------------------------------ #

    async def _handle_tool_call(
        self, step: int, tool_call: ToolCallRequest, trace: Trace
    ) -> str:
        """Validate → HITL gate → execute (with idempotency). Returns the
        observation string for the tool message; the agent retries on errors."""
        if tool_call.parse_error is not None:
            trace.add_validation_failure(step, tool_call.name, tool_call.parse_error)
            return f"Error: {tool_call.parse_error}"

        args = tool_call.arguments or {}
        validation = self._validator.validate(tool_call.name, args)
        if not validation.valid:
            trace.add_validation_failure(step, tool_call.name, validation.error)
            return f"Error: {validation.error}"

        tool = self._registry.get(tool_call.name)

        if tool.risk_level != RiskLevel.NONE:
            # The gate implements the full risk matrix (LOW/MEDIUM auto-approve
            # with audit, HIGH blocks for a human). Auto-approvals of sub-HIGH
            # tools are audited in Postgres but don't mark the trace as
            # HITL-triggered — that flag means "a human was in the loop".
            decision = await self._hitl_gate.request_approval(tool, args, trace)
            if tool.risk_level == RiskLevel.HIGH or decision.status != ApprovalStatus.APPROVED:
                trace.add_hitl_decision(step, tool.name, decision.status.value, decision.reason)
            if decision.status == ApprovalStatus.REJECTED:
                return (
                    "Action was rejected by a supervisor. "
                    f"Reason: {decision.reason or 'not provided'}"
                )
            if decision.status == ApprovalStatus.TIMEOUT:
                raise _PendingApproval()

        started = time.perf_counter()
        try:
            result = await self._execute_with_idempotency(tool, args)
        except ToolExecutionError as exc:
            trace.add_tool_error(step, tool.name, args, str(exc))
            return f"Tool error: {exc}"
        except Exception as exc:
            logger.exception("Unexpected error executing tool '%s'", tool.name)
            trace.add_tool_error(step, tool.name, args, f"unexpected: {exc}")
            return f"Tool error: {exc}"
        latency_ms = int((time.perf_counter() - started) * 1000)

        trace.add_tool_result(
            step,
            tool.name,
            args,
            result,
            latency_ms,
            retrieval_scores=self._extract_retrieval_scores(tool.name, result),
        )
        return result.to_message()

    @staticmethod
    def _extract_retrieval_scores(tool_name: str, result: ToolResult) -> list[float] | None:
        if tool_name != "search_knowledge" or not result.success or not isinstance(result.data, list):
            return None
        return [r["score"] for r in result.data if isinstance(r, dict) and "score" in r]

    # ------------------------------------------------------------------ #
    # Idempotency                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _idempotency_key(tool_name: str, args: dict) -> str:
        canonical = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return f"idempotent:{tool_name}:{digest}"

    def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            from app.db.redis import get_redis

            return get_redis()
        except RuntimeError:
            return None

    async def _execute_with_idempotency(self, tool: Tool, args: dict) -> ToolResult:
        if not tool.is_state_changing:
            return await tool.execute(**args)

        redis = self._get_redis()
        key = self._idempotency_key(tool.name, args)

        if redis is not None:
            try:
                cached = await redis.get(key)
            except Exception as exc:
                logger.error("Idempotency cache read failed (%s) — executing anyway", exc)
                cached = None
                redis = None
            if cached:
                logger.info("Idempotent replay of '%s' — returning cached result", tool.name)
                return ToolResult.from_json(cached)
        else:
            logger.warning(
                "Redis unavailable — executing state-changing tool '%s' without "
                "idempotency protection",
                tool.name,
            )

        result = await tool.execute(**args)

        # Cache only successes: failures must stay retryable.
        if redis is not None and result.success:
            try:
                await redis.setex(key, _IDEMPOTENCY_TTL_SECONDS, result.to_json())
            except Exception as exc:
                logger.error("Idempotency cache write failed: %s", exc)

        return result

    # ------------------------------------------------------------------ #
    # Message building                                                     #
    # ------------------------------------------------------------------ #

    def _build_messages(
        self, query: str, intent: IntentResult, history: list[dict] | None
    ) -> list[dict]:
        context_lines = [f"Today's date: {date.today().isoformat()}."]
        if intent.extracted_order_id:
            context_lines.append(f"Order ID mentioned by the customer: {intent.extracted_order_id}")
        if intent.extracted_customer_id:
            context_lines.append(f"Customer identifier: {intent.extracted_customer_id}")
        if intent.sentiment == "angry":
            context_lines.append("The customer sounds upset — acknowledge their frustration first.")
        if intent.language in ("hi", "mixed"):
            context_lines.append("Reply in the customer's Hindi-English mix.")
        if intent.intent == IntentType.OUT_OF_SCOPE:
            context_lines.append(
                "This query appears out of scope for ShopEasy support — politely decline."
            )

        messages: list[dict] = [
            {"role": "system", "content": self._prompt.text},
            {"role": "system", "content": "\n".join(context_lines)},
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})
        return messages
