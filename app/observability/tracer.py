"""Cost and persist traces without surfacing telemetry failures to customers."""
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
