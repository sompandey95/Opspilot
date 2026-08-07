"""Tracer: cost the trace, then persist it — the Phase-6 layer on top of
Trace.persist(). Still best-effort end to end: a costing bug or dead DB is
logged, never surfaced to the customer request."""
from __future__ import annotations

import logging

from app.budget.cost_calculator import CostCalculator
from app.observability.trace import Trace

logger = logging.getLogger(__name__)


class Tracer:
    def __init__(self, cost_calculator: type[CostCalculator] = CostCalculator) -> None:
        self._costs = cost_calculator

    async def persist(self, trace: Trace) -> Trace:
        try:
            trace.cost_inr = self._costs.cost_for_trace(trace)
        except Exception as exc:
            logger.error("Cost calculation failed for trace %s: %s", trace.trace_id, exc)
        await trace.persist()
        return trace
