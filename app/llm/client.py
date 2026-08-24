"""Thin AsyncAzureOpenAI chat wrapper — the single LLM entry point.

All agent/eval code goes through LLMClient, never raw openai. Callers pick a
ModelRole; the client maps it to the Azure deployment from Settings, applies a
per-call timeout, and returns a normalized LLMResponse (content, parsed tool
calls, token usage, latency).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import openai

from app.config import Settings

logger = logging.getLogger(__name__)


class ModelRole(str, Enum):
    AGENT = "agent"            # GPT-5.4 — complex multi-step reasoning
    AGENT_MINI = "agent_mini"  # GPT-5.4-mini — FAQ / simple actions
    CLASSIFIER = "classifier"  # GPT-5.4-mini — intent classification
    SUMMARIZER = "summarizer"  # GPT-5.4-mini — session summaries
    JUDGE = "judge"            # GPT-4o — independent eval judge


class LLMConfigError(Exception):
    """Raised when the deployment for a role is not configured."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments_raw: str
    arguments: dict | None = None
    parse_error: str | None = None


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str | None = None
    latency_ms: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_assistant_message(self) -> dict:
        """OpenAI-format assistant message — must precede the paired tool
        messages in history or the API rejects the request."""
        message: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments_raw},
                }
                for tc in self.tool_calls
            ]
        return message


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = openai.AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        self._deployments: dict[ModelRole, str] = {
            ModelRole.AGENT: settings.AZURE_DEPLOYMENT_GPT54,
            ModelRole.AGENT_MINI: settings.AZURE_DEPLOYMENT_GPT54_MINI,
            ModelRole.CLASSIFIER: settings.AZURE_DEPLOYMENT_GPT54_MINI,
            ModelRole.SUMMARIZER: settings.AZURE_DEPLOYMENT_GPT54_MINI,
            ModelRole.JUDGE: settings.AZURE_DEPLOYMENT_GPT4O,
        }

    def deployment_for(self, role: ModelRole) -> str:
        deployment = self._deployments.get(role, "")
        if not deployment:
            raise LLMConfigError(f"No Azure deployment configured for role '{role.value}'")
        return deployment

    async def complete(
        self,
        role: ModelRole,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_completion_tokens: int | None = None,
        response_format: dict | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        deployment = self.deployment_for(role)

        kwargs: dict = {}
        if tools:
            kwargs["tools"] = tools
            # One tool call per step keeps the ReAct loop and HITL gate simple.
            kwargs["parallel_tool_calls"] = False
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format

        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=deployment,
            messages=messages,
            timeout=timeout or float(self._settings.AGENT_TIMEOUT_SECONDS),
            **kwargs,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        choice = response.choices[0]

        tool_calls: list[ToolCallRequest] = []
        for tc in choice.message.tool_calls or []:
            raw = tc.function.arguments or ""
            parsed: dict | None = None
            parse_error: str | None = None
            try:
                loaded = json.loads(raw) if raw else {}
                if isinstance(loaded, dict):
                    parsed = loaded
                else:
                    parse_error = f"arguments must be a JSON object, got {type(loaded).__name__}"
            except json.JSONDecodeError as exc:
                parse_error = f"arguments are not valid JSON: {exc}"
            tool_calls.append(
                ToolCallRequest(
                    id=tc.id,
                    name=tc.function.name,
                    arguments_raw=raw,
                    arguments=parsed,
                    parse_error=parse_error,
                )
            )

        usage = Usage()
        if response.usage is not None:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            )

        return LLMResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            usage=usage,
            model=deployment,
            finish_reason=choice.finish_reason,
            latency_ms=latency_ms,
        )
