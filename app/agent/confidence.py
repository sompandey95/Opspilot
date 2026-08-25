"""Deterministic 0–1 confidence score from the trace.

Components (no LLM call — cheap, reproducible, unit-testable):
- retrieval quality: best sigmoid-normalised score among search_knowledge
  results (cross-encoder outputs are unbounded logits)
- tool reliability: fraction of tool executions that succeeded
- reasoning consistency: validation failures and near-max step usage
- answer sanity: implausibly short final answers

Score < CONFIDENCE_THRESHOLD (0.7) ⇒ the agent auto-escalates.
"""
from __future__ import annotations

import math
import re

from app.config import Settings
from app.guardrails.output_guard import has_verifiable_claim
from app.observability.trace import Trace

_BASE = 0.5
_RETRIEVAL_MAX_BONUS = 0.25
_FAQ_NO_RETRIEVAL_PENALTY = 0.15
# All-success tool flows must clear CONFIDENCE_THRESHOLD (0.7) without any
# retrieval bonus: 0.5 + 0.3 = 0.8. Any tool failure pulls below it — the
# agent leans escalate when its actions didn't all land.
_TOOL_SUCCESS_BONUS = 0.30
_TOOL_FAILURE_PENALTY = 0.30
_VALIDATION_FAILURE_PENALTY = 0.05
_VALIDATION_PENALTY_CAP = 0.15
_NEAR_MAX_STEPS_PENALTY = 0.10
_SHORT_ANSWER_PENALTY = 0.10
_MIN_ANSWER_CHARS = 20
# Asking the customer for missing info (e.g. an order ID) is the right move
# when a query gave the agent nothing to act on — not low confidence. Detected
# by "?" or information-request phrasing (including Hindi), and only when the
# answer asserts no checkable fact: a reply that states an amount, date, or
# window is answering, not asking, so it stays subject to the grounding
# penalties even if it ends with a question.
_CLARIFYING_QUESTION_BONUS = 0.25
_INFO_REQUEST_RE = re.compile(
    r"please\s+(?:share|provide|paste|send|confirm|tell|enter|reply)"
    r"|could\s+you|can\s+you\s+(?:share|provide|send)"
    r"|i\s+need\s+(?:your|the)|what\s+is\s+your"
    # The prompt tells the agent to quote this placeholder whenever it asks for
    # an order ID, which makes it the most reliable request signal we have.
    r"|ord-yyyy-nnnnn"
    r"|batayein|bataiye|share\s+karein",
    re.IGNORECASE,
)
_CLARIFIABLE_INTENTS = ("faq", "action_simple", "action_complex")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class ConfidenceScorer:
    def __init__(self, settings: Settings) -> None:
        self._max_steps = settings.MAX_AGENT_STEPS

    def score(self, query: str, answer: str, trace: Trace) -> float:
        score = _BASE

        retrieval_scores: list[float] = []
        tool_results = 0
        tool_failures = 0
        validation_failures = 0
        llm_steps = 0

        for step in trace.steps:
            step_type = step.get("type")
            if step_type == "llm":
                llm_steps += 1
            elif step_type == "validation_error":
                validation_failures += 1
            elif step_type in ("tool_result", "tool_error"):
                tool_results += 1
                unsuccessful = step_type == "tool_error" or not step.get("success", False)
                # A "no such record" result is a definitive answer, not a
                # malfunction — see ToolResult.not_found.
                if unsuccessful and not step.get("not_found", False):
                    tool_failures += 1
                retrieval_scores.extend(step.get("retrieval_scores") or [])

        # Asking the customer for more information is not an answer, so it is
        # exempt from the grounding penalties an answer would attract.
        asks_for_info = (
            "?" in answer or _INFO_REQUEST_RE.search(answer) is not None
        ) and not has_verifiable_claim(answer)

        # Retrieval quality
        if retrieval_scores:
            score += _RETRIEVAL_MAX_BONUS * _sigmoid(max(retrieval_scores))
        elif trace.intent == "faq" and tool_results == 0 and not asks_for_info:
            # A "faq" answer grounded in neither the knowledge base nor any
            # tool call is suspicious. But the intent classifier sometimes
            # tags an order-status question "faq" even though check_order_status
            # (not search_knowledge) is the right tool for it — that answer is
            # grounded via the tool result, just not via retrieval. Only
            # penalise when there's no grounding of either kind.
            score -= _FAQ_NO_RETRIEVAL_PENALTY

        # Tool reliability
        if tool_results:
            failure_ratio = tool_failures / tool_results
            score += _TOOL_SUCCESS_BONUS * (1.0 - failure_ratio)
            score -= _TOOL_FAILURE_PENALTY * failure_ratio

        # Reasoning consistency
        score -= min(
            _VALIDATION_FAILURE_PENALTY * validation_failures, _VALIDATION_PENALTY_CAP
        )
        if llm_steps >= self._max_steps - 1:
            score -= _NEAR_MAX_STEPS_PENALTY

        # Clarifying question on a query with nothing to act on. "faq" is
        # included because a one-word query like "refund" lands there, and
        # asking what the customer wants to know beats paging a human.
        if (
            trace.intent in _CLARIFIABLE_INTENTS
            and tool_results == 0
            and validation_failures == 0
            and asks_for_info
        ):
            score += _CLARIFYING_QUESTION_BONUS

        # Answer sanity
        if len(answer.strip()) < _MIN_ANSWER_CHARS:
            score -= _SHORT_ANSWER_PENALTY

        return max(0.0, min(1.0, score))
