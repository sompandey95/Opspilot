CREATE_TRACES = """
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
"""

CREATE_TRACES_IDX_CREATED_AT = """
CREATE INDEX IF NOT EXISTS idx_traces_created_at
    ON traces (created_at)
"""

CREATE_TRACES_IDX_INTENT = """
CREATE INDEX IF NOT EXISTS idx_traces_intent
    ON traces (intent)
"""

CREATE_TRACES_IDX_HITL = """
CREATE INDEX IF NOT EXISTS idx_traces_hitl_triggered
    ON traces (hitl_triggered)
"""

CREATE_TRACES_IDX_PROMPT_VERSION = """
CREATE INDEX IF NOT EXISTS idx_traces_prompt_version
    ON traces (prompt_version)
"""

# trace_id is intentionally NOT a foreign key (dropped in migration 0002):
# audit rows are written mid-run, before the trace row itself is persisted.
CREATE_HITL_AUDIT_LOG = """
CREATE TABLE IF NOT EXISTS hitl_audit_log (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id          UUID,
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
"""

# Open approval requests. Rows stay 'pending' on timeout so a human can still
# decide later; decisions move them to approved/rejected (see app/hitl/queue.py).
CREATE_HITL_PENDING = """
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
"""

CREATE_HITL_PENDING_IDX_STATUS = """
CREATE INDEX IF NOT EXISTS idx_hitl_pending_status
    ON hitl_pending (status)
"""

CREATE_EVAL_RUNS = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id                      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    prompt_version          VARCHAR(16) NOT NULL,
    model                   VARCHAR(64) NOT NULL,
    total_scenarios         INTEGER     NOT NULL,
    retrieval_precision     NUMERIC(5,4),
    retrieval_recall        NUMERIC(5,4),
    retrieval_mrr           NUMERIC(5,4),
    faithfulness_avg        NUMERIC(5,4),
    hallucination_rate      NUMERIC(5,4),
    tool_accuracy           NUMERIC(5,4),
    relevance_avg           NUMERIC(5,4),
    avg_latency_ms          INTEGER,
    avg_cost_inr            NUMERIC(10,4),
    ci_gate_passed          BOOLEAN,
    details                 JSONB,
    created_at              TIMESTAMPTZ DEFAULT NOW()
)
"""

ALL_DDL = [
    CREATE_TRACES,
    CREATE_TRACES_IDX_CREATED_AT,
    CREATE_TRACES_IDX_INTENT,
    CREATE_TRACES_IDX_HITL,
    CREATE_TRACES_IDX_PROMPT_VERSION,
    CREATE_HITL_AUDIT_LOG,
    CREATE_HITL_PENDING,
    CREATE_HITL_PENDING_IDX_STATUS,
    CREATE_EVAL_RUNS,
]
