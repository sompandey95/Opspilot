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

from app.config import Settings
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
# when an action query gave the agent nothing to act on — not low confidence.
# Detected by "?" or common info-request phrasings (incl. Hindi); Phase 5's
# output guard replaces this heuristic with real claim extraction.
_CLARIFYING_QUESTION_BONUS = 0.25
_INFO_REQUEST_PHRASES = (
    "please share", "please provide", "share your", "provide your",
    "could you", "can you share", "i need your", "what is your",
    "batayein", "bataiye", "share karein",
)


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
                if step_type == "tool_error" or not step.get("success", False):
                    tool_failures += 1
                retrieval_scores.extend(step.get("retrieval_scores") or [])

        # Retrieval quality
        if retrieval_scores:
            score += _RETRIEVAL_MAX_BONUS * _sigmoid(max(retrieval_scores))
        elif trace.intent == "faq":
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

        # Clarifying question on an action query with nothing to act on
        lowered = answer.lower()
        asks_for_info = "?" in answer or any(p in lowered for p in _INFO_REQUEST_PHRASES)
        if (
            trace.intent in ("action_simple", "action_complex")
            and tool_results == 0
            and validation_failures == 0
            and asks_for_info
        ):
            score += _CLARIFYING_QUESTION_BONUS

        # Answer sanity
        if len(answer.strip()) < _MIN_ANSWER_CHARS:
            score -= _SHORT_ANSWER_PENALTY

        return max(0.0, min(1.0, score))
