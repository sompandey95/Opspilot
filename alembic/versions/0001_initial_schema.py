"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-25
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id        UUID,
            query             TEXT        NOT NULL,
            intent            VARCHAR(32),
            model             VARCHAR(64),
            steps             JSONB       NOT NULL,
            response          TEXT,
            confidence        NUMERIC(4,3),
            total_latency_ms  INTEGER,
            total_tokens      INTEGER,
            input_tokens      INTEGER,
            output_tokens     INTEGER,
            cost_inr          NUMERIC(10,4),
            hitl_triggered    BOOLEAN     DEFAULT FALSE,
            hitl_outcome      VARCHAR(16),
            escalated         BOOLEAN     DEFAULT FALSE,
            guardrail_flags   JSONB,
            eval_scores       JSONB,
            prompt_version    VARCHAR(16),
            created_at        TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    op.execute("CREATE INDEX IF NOT EXISTS idx_traces_created_at ON traces (created_at)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_traces_intent ON traces (intent)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_traces_hitl_triggered ON traces (hitl_triggered)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_traces_prompt_version ON traces (prompt_version)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS hitl_audit_log (
            id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            trace_id          UUID        REFERENCES traces(id),
            tool_name         VARCHAR(64) NOT NULL,
            tool_args         JSONB       NOT NULL,
            risk_level        VARCHAR(16) NOT NULL,
            agent_reasoning   TEXT,
            decision          VARCHAR(16) NOT NULL,
            decided_by        VARCHAR(64),
            override_args     JSONB,
            decision_time_ms  INTEGER,
            created_at        TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            prompt_version      VARCHAR(16) NOT NULL,
            model               VARCHAR(64) NOT NULL,
            total_scenarios     INTEGER     NOT NULL,
            retrieval_precision NUMERIC(5,4),
            retrieval_recall    NUMERIC(5,4),
            retrieval_mrr       NUMERIC(5,4),
            faithfulness_avg    NUMERIC(5,4),
            hallucination_rate  NUMERIC(5,4),
            tool_accuracy       NUMERIC(5,4),
            relevance_avg       NUMERIC(5,4),
            avg_latency_ms      INTEGER,
            avg_cost_inr        NUMERIC(10,4),
            ci_gate_passed      BOOLEAN,
            details             JSONB,
            created_at          TIMESTAMPTZ DEFAULT NOW()
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS hitl_audit_log")
    op.execute("DROP TABLE IF EXISTS eval_runs")
    op.execute("DROP INDEX IF EXISTS idx_traces_prompt_version")
    op.execute("DROP INDEX IF EXISTS idx_traces_hitl_triggered")
    op.execute("DROP INDEX IF EXISTS idx_traces_intent")
    op.execute("DROP INDEX IF EXISTS idx_traces_created_at")
    op.execute("DROP TABLE IF EXISTS traces")
