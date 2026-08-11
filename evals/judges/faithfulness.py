"""Faithfulness judge — GPT-4o (JUDGE role) scores 0–1 whether the answer is
fully supported by the context the agent saw. Independent of the agent models
by design. Returns None on judge failure so the runner can average over the
scenarios that were actually scored instead of poisoning the mean.
"""
from __future__ import annotations

import json
import logging

from app.llm.client import LLMClient, ModelRole

logger = logging.getLogger(__name__)

_TIMEOUT_S = 30.0

JUDGE_PROMPT = """You are an impartial evaluator for a customer-support AI. Judge whether the ANSWER is faithful to the CONTEXT (retrieved policy text and tool results). Faithful means: every factual statement in the answer — numbers, dates, timelines, amounts, order statuses, policy rules — is directly supported by the context. General courtesy phrases and offers to help further do not need support.

Respond with JSON only:
{"score": <float 0.0-1.0>, "unsupported_statements": ["<each unsupported factual statement>"], "reasoning": "<one line>"}

Scoring guide:
- 1.0: every factual statement supported
- 0.7-0.9: minor unsupported detail that would not mislead the customer
- 0.3-0.6: at least one materially unsupported claim
- 0.0-0.2: answer contradicts the context or is mostly fabricated"""


class FaithfulnessJudge:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def score(self, query: str, answer: str, context: str) -> dict | None:
        """Returns {"score": float, "unsupported_statements": [...]} or None."""
        user = (
            f"QUESTION:\n{query}\n\n"
            f"CONTEXT:\n{context or '(no context was retrieved)'}\n\n"
            f"ANSWER:\n{answer}"
        )
        try:
            response = await self._llm.complete(
                role=ModelRole.JUDGE,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=500,
                timeout=_TIMEOUT_S,
            )
            payload = json.loads(response.content or "")
            score = max(0.0, min(1.0, float(payload["score"])))
        except Exception as exc:
            logger.error("Faithfulness judge failed: %s", exc)
            return None
        return {
            "score": score,
            "unsupported_statements": payload.get("unsupported_statements") or [],
        }
