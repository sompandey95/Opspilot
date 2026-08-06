"""Postgres-backed approval queue for high-risk agent actions.

A HIGH-risk tool call creates a `hitl_pending` row; the agent then polls
`wait_for_decision` until a human approves/rejects via the HITL API routes or
the timeout elapses. Timed-out rows deliberately stay 'pending' so a human can
still decide later — `decide()` only transitions rows out of 'pending' once.

Human decisions are audited here (the gate audits automatic decisions), so
every decision produces exactly one `hitl_audit_log` row.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

from app.config import Settings

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"

_CREATE_SQL = """
INSERT INTO hitl_pending (id, trace_id, tool_name, tool_args, risk_level, agent_reasoning)
VALUES ($1, $2, $3, $4::jsonb, $5, $6)
"""

_GET_SQL = """
SELECT id, trace_id, tool_name, tool_args, risk_level, agent_reasoning,
       status, decided_by, decision_reason, created_at, decided_at
FROM hitl_pending WHERE id = $1
"""

_LIST_PENDING_SQL = """
SELECT id, trace_id, tool_name, tool_args, risk_level, agent_reasoning,
       status, decided_by, decision_reason, created_at, decided_at
FROM hitl_pending WHERE status = 'pending' ORDER BY created_at
"""

_DECIDE_SQL = """
UPDATE hitl_pending
SET status = $2, decided_by = $3, decision_reason = $4, decided_at = NOW()
WHERE id = $1 AND status = 'pending'
RETURNING id, trace_id, tool_name, tool_args, risk_level, agent_reasoning,
          status, decided_by, decision_reason, created_at, decided_at
"""

_AUDIT_SQL = """
INSERT INTO hitl_audit_log (
    trace_id, tool_name, tool_args, risk_level, agent_reasoning,
    decision, decided_by, decision_time_ms
) VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7, $8)
"""


@dataclass
class HITLRequest:
    id: str
    tool_name: str
    tool_args: dict
    risk_level: str
    status: str
    trace_id: str | None = None
    agent_reasoning: str | None = None
    decided_by: str | None = None
    decision_reason: str | None = None
    created_at: datetime | None = None
    decided_at: datetime | None = None

    @classmethod
    def from_row(cls, row) -> "HITLRequest":
        args = row["tool_args"]
        return cls(
            id=str(row["id"]),
            trace_id=str(row["trace_id"]) if row["trace_id"] else None,
            tool_name=row["tool_name"],
            tool_args=json.loads(args) if isinstance(args, str) else args,
            risk_level=row["risk_level"],
            agent_reasoning=row["agent_reasoning"],
            status=row["status"],
            decided_by=row["decided_by"],
            decision_reason=row["decision_reason"],
            created_at=row["created_at"],
            decided_at=row["decided_at"],
        )


async def record_audit(
    trace_id: str | None,
    tool_name: str,
    tool_args: dict,
    risk_level: str,
    decision: str,
    decided_by: str | None = None,
    agent_reasoning: str | None = None,
    decision_time_ms: int | None = None,
) -> None:
    """Best-effort audit row — a dead DB must never block a decision."""
    from app.db.postgres import execute

    trace_uuid = None
    if trace_id:
        try:
            trace_uuid = uuid.UUID(trace_id)
        except ValueError:
            pass

    try:
        await execute(
            _AUDIT_SQL,
            trace_uuid,
            tool_name,
            json.dumps(tool_args, default=str),
            risk_level,
            agent_reasoning,
            decision,
            decided_by,
            decision_time_ms,
        )
    except Exception as exc:
        logger.error("HITL audit write failed (%s %s): %s", tool_name, decision, exc)


class HITLQueue:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def create(
        self,
        tool_name: str,
        tool_args: dict,
        risk_level: str,
        trace_id: str | None = None,
        agent_reasoning: str | None = None,
    ) -> str:
        from app.db.postgres import execute

        request_id = str(uuid.uuid4())
        trace_uuid = None
        if trace_id:
            try:
                trace_uuid = uuid.UUID(trace_id)
            except ValueError:
                pass

        await execute(
            _CREATE_SQL,
            uuid.UUID(request_id),
            trace_uuid,
            tool_name,
            json.dumps(tool_args, default=str),
            risk_level,
            agent_reasoning,
        )
        return request_id

    async def get(self, request_id: str) -> HITLRequest | None:
        from app.db.postgres import fetch_one

        try:
            rid = uuid.UUID(request_id)
        except ValueError:
            return None
        row = await fetch_one(_GET_SQL, rid)
        return HITLRequest.from_row(row) if row else None

    async def list_pending(self) -> list[HITLRequest]:
        from app.db.postgres import fetch_all

        rows = await fetch_all(_LIST_PENDING_SQL)
        return [HITLRequest.from_row(r) for r in rows]

    async def decide(
        self,
        request_id: str,
        approved: bool,
        decided_by: str,
        reason: str | None = None,
    ) -> HITLRequest | None:
        """Record a human decision. Returns the updated request, or None if the
        request doesn't exist or was already decided (single transition only)."""
        from app.db.postgres import fetch_one

        try:
            rid = uuid.UUID(request_id)
        except ValueError:
            return None

        status = STATUS_APPROVED if approved else STATUS_REJECTED
        row = await fetch_one(_DECIDE_SQL, rid, status, decided_by, reason)
        if row is None:
            return None

        request = HITLRequest.from_row(row)
        decision_time_ms = None
        if request.created_at and request.decided_at:
            decision_time_ms = int(
                (request.decided_at - request.created_at).total_seconds() * 1000
            )
        await record_audit(
            trace_id=request.trace_id,
            tool_name=request.tool_name,
            tool_args=request.tool_args,
            risk_level=request.risk_level,
            decision=request.status,
            decided_by=decided_by,
            agent_reasoning=request.agent_reasoning,
            decision_time_ms=decision_time_ms,
        )
        return request

    async def wait_for_decision(
        self, request_id: str, timeout_seconds: float
    ) -> HITLRequest | None:
        """Poll until the request leaves 'pending'. Returns the decided request,
        or None on timeout (the row stays pending for a late human decision)."""
        poll = max(0.1, self._settings.HITL_POLL_INTERVAL_SECONDS)
        deadline = time.monotonic() + timeout_seconds

        while True:
            request = await self.get(request_id)
            if request is not None and request.status != STATUS_PENDING:
                return request
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            await asyncio.sleep(min(poll, remaining))
