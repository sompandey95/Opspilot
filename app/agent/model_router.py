"""Static intent → model routing. Cost-aware routing lives inside the agent,
not as a separate service: cheapest model that can handle the intent."""
from __future__ import annotations

from dataclasses import dataclass

from app.agent.intent_classifier import IntentType
from app.llm.client import ModelRole


@dataclass(frozen=True)
class RouteConfig:
    role: ModelRole | None          # None ⇒ no LLM call (direct escalation)
    max_completion_tokens: int | None = None


ROUTING_TABLE: dict[IntentType, RouteConfig] = {
    IntentType.FAQ: RouteConfig(ModelRole.AGENT_MINI, 500),
    IntentType.ACTION_SIMPLE: RouteConfig(ModelRole.AGENT_MINI, 800),
    IntentType.ACTION_COMPLEX: RouteConfig(ModelRole.AGENT, 1500),
    IntentType.ESCALATE: RouteConfig(None),
    IntentType.OUT_OF_SCOPE: RouteConfig(ModelRole.AGENT_MINI, 200),
}


class ModelRouter:
    @staticmethod
    def get_route(intent: IntentType) -> RouteConfig:
        return ROUTING_TABLE[intent]
