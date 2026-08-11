"""hitl pending queue + audit log FK removal

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-05
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS hitl_pending (
            id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            trace_id          UUID,
            tool_name         VARCHAR(64) NOT NULL,
            tool_args         JSONB       NOT NULL,
            risk_level        VARCHAR(16) NOT NULL,
            agent_reasoning   TEXT,
            status            VARCHAR(16) NOT NULL DEFAULT 'pending',
            decided_by        VARCHAR(64),
            decision_reason   TEXT,
            created_at        TIMESTAMPTZ DEFAULT NOW(),
            decided_at        TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_hitl_pending_status ON hitl_pending (status)")

    # Audit rows are written mid-run, before the trace row is persisted at the
    # end of the request — the FK would reject every insert.
    op.execute("ALTER TABLE hitl_audit_log DROP CONSTRAINT IF EXISTS hitl_audit_log_trace_id_fkey")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS hitl_pending")
    op.execute(
        "ALTER TABLE hitl_audit_log ADD CONSTRAINT hitl_audit_log_trace_id_fkey "
        "FOREIGN KEY (trace_id) REFERENCES traces(id)"
    )
