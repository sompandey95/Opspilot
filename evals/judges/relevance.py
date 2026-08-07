"""Relevance judge — GPT-4o (JUDGE role) scores 0–1 how well the answer
addresses what the customer actually asked (with the reference answer as a
rubric, not a required phrasing). Returns None on judge failure."""
from __future__ import annotations

import json
import logging

from app.llm.client import LLMClient, ModelRole

logger = logging.getLogger(__name__)

_TIMEOUT_S = 30.0

JUDGE_PROMPT = """You are an impartial evaluator for a customer-support AI. Judge how well the ANSWER addresses the customer's QUESTION. The REFERENCE describes what a correct, complete answer covers — the answer does not need to match its wording, only its substance. Judge relevance and completeness, not politeness or style.

Respond with JSON only:
{"score": <float 0.0-1.0>, "reasoning": "<one line>"}

Scoring guide:
- 1.0: directly and completely addresses the question
- 0.7-0.9: addresses the question but misses a secondary part
- 0.3-0.6: partially on-topic; customer would need to ask again
- 0.0-0.2: off-topic, evasive, or answers a different question"""


class RelevanceJudge:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def score(self, query: str, answer: str, reference: str = "") -> float | None:
        user = (
            f"QUESTION:\n{query}\n\n"
            f"REFERENCE (what a correct answer covers):\n{reference or '(none provided)'}\n\n"
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
                max_completion_tokens=300,
                timeout=_TIMEOUT_S,
            )
            payload = json.loads(response.content or "")
            return max(0.0, min(1.0, float(payload["score"])))
        except Exception as exc:
            logger.error("Relevance judge failed: %s", exc)
            return None
