"""₹ cost from token usage, priced per model.

Prices are Azure OpenAI pay-as-you-go list prices converted at ₹88/USD
(2026-08), expressed per **1M tokens**. Azure deployment names are
user-chosen ("gpt54-prod", "gpt-5.4-mini-eu"), so lookup normalises both the
deployment name and the table keys to bare alphanumerics and picks the longest
key contained in the name — "gpt54mini" wins over "gpt54" for a mini
deployment. Unknown models are priced at the most expensive tier so a mapping
gap can only ever over-report cost.
"""
from __future__ import annotations

import re

from app.observability.trace import Trace

# (input ₹/1M tokens, output ₹/1M tokens)
PRICING_INR_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-5.4": (220.0, 880.0),                  # $2.50 / $10.00
    "gpt-5.4-mini": (13.2, 52.8),               # $0.15 / $0.60
    "gpt-4o": (220.0, 880.0),                   # $2.50 / $10.00
    "text-embedding-3-large": (11.4, 0.0),      # $0.13 / —
}

_FALLBACK = max(PRICING_INR_PER_1M.values())    # priciest tier: over-report, never under


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


_NORMALISED_PRICING = sorted(
    ((_normalise(k), v) for k, v in PRICING_INR_PER_1M.items()),
    key=lambda kv: len(kv[0]),
    reverse=True,  # longest key first: "gpt54mini" must beat "gpt54"
)


class CostCalculator:
    @staticmethod
    def pricing_for(model: str) -> tuple[float, float]:
        name = _normalise(model or "")
        for key, prices in _NORMALISED_PRICING:
            if key in name:
                return prices
        return _FALLBACK

    @classmethod
    def cost_inr(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        input_price, output_price = cls.pricing_for(model)
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    @classmethod
    def cost_for_trace(cls, trace: Trace) -> float:
        """Per-step cost using each step's own model; tokens not attributed to
        a step (the intent classifier's — always the mini model) priced at the
        mini rate."""
        cost = 0.0
        step_input = 0
        step_output = 0
        for step in trace.steps:
            if step.get("type") != "llm":
                continue
            in_tok = step.get("input_tokens") or 0
            out_tok = step.get("output_tokens") or 0
            cost += cls.cost_inr(step.get("model") or "", in_tok, out_tok)
            step_input += in_tok
            step_output += out_tok

        cost += cls.cost_inr(
            "gpt-5.4-mini",
            max(0, trace.input_tokens - step_input),
            max(0, trace.output_tokens - step_output),
        )
        return round(cost, 6)
