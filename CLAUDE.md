# OpsPilot — AI Support & Operations Agent

Production-grade AI support agent for **ShopEasy**, a fictional Indian e-commerce
platform. Handles customer queries end-to-end: RAG over company docs, real tool
calls (Jira, Slack, order system), human-in-the-loop approval for risky actions,
and an eval harness with a CI gate that blocks quality regressions.

Full design document: `OpsPilot_Project_Document.pdf` (source of truth for
architecture; deviations from it are listed below).

## Stack

Python 3.12 · FastAPI · PostgreSQL 16 (raw **asyncpg**, no ORM) · Redis 7 ·
ChromaDB (HTTP client, port 8100) · Docker Compose · pytest + pytest-asyncio.

**LLM provider is Azure OpenAI** — NOT plain OpenAI as the PDF says. All model
access goes through `openai.AsyncAzureOpenAI` with deployment names from config
(`AZURE_DEPLOYMENT_GPT54`, `AZURE_DEPLOYMENT_GPT54_MINI`, `AZURE_DEPLOYMENT_GPT4O`,
`AZURE_DEPLOYMENT_EMBEDDING`). Roles: GPT-5.4-mini = classifier/simple/summaries,
GPT-5.4 = complex reasoning, GPT-4o = independent eval judge.
Embeddings: text-embedding-3-large @ 3072 dims (PDF says -small; code won).

## Commands

```bash
docker compose up -d                        # postgres, redis, chromadb, mock-order-service
python scripts/ingest_knowledge.py --reset  # chunk + embed + dedup + store KB
uvicorn app.main:app --reload               # main API (port 8000)
uvicorn mock_services.order_service.main:app --port 8001   # mock order system (local dev)
pytest                                      # run tests
```

Health check: `GET /api/v1/health` · Retrieval debug: `GET /api/v1/rag/test?query=...`

## Conventions (follow these — they're established in the codebase)

- Async everywhere; blocking libs (chromadb, sentence-transformers) wrapped in
  `asyncio.to_thread`.
- DB access = raw SQL via helpers in `app/db/postgres.py` (`execute`, `fetch_one`,
  `fetch_all`). DDL lives as SQL strings in `app/observability/models.py`
  (`ALL_DDL`) and is mirrored in Alembic migrations.
- Settings: `app/config.py` `Settings` + `get_settings()` (lru_cached). New knobs
  go there with defaults, and into `.env.example`.
- Startup must never crash: every subsystem in `app/main.py` `lifespan` degrades
  gracefully (log an error, set `app.state.X = None`, continue). Preserve this.
- Dataclasses for internal value objects (`Chunk`, `RetrievalResult`, `ToolResult`);
  Pydantic only at API boundaries.
- Chunk ID scheme (eval ground truth depends on these — do not change format):
  `faq_{stem}_{NNN}`, `policy_{stem}_{NNN}`, `api_doc_{stem}_{NNN}`,
  `changelog_{stem}_{NNN}`, tickets = file stem (`ticket_001`).
- Tests live in `tests/`, plain pytest functions, `asyncio_mode = "auto"`.
  LLM/network calls are mocked; order-service tools are tested against the real
  mock app via `httpx.ASGITransport`.

## Current state (Phases 1–9 complete)

```
app/
  config.py                 # Settings (Azure OpenAI, RAG, agent, HITL, eval knobs)
  main.py                   # lifespan: logging, db, redis, RAG, hitl, guards, tools,
                            # agent, sessions/budget/tracer (fault-tolerant);
                            # middleware order: CORS outside APIMiddleware
  api/routes.py             # /health, /chat (budget check → session history →
                            # intent → route → agent → output guard → tracer
                            # persist → budget record → session save), /rag/test
  api/admin_routes.py       # GET /admin/traces[?last=24h|7d], /admin/traces/{id},
                            # /admin/metrics/summary, /admin/metrics/cost-breakdown,
                            # /admin/hitl/stats, /admin/hitl/pending,
                            # /admin/evals/latest, /admin/evals/trend?versions=v1,v2;
                            # POST /admin/evals/run (202, fire-and-forget, 503 w/o
                            # Azure creds — result lands via /evals/latest)
  api/hitl_routes.py        # POST /hitl/approve/{id}, /hitl/reject/{id} (409 on
                            # double-decide), GET /hitl/pending
  api/middleware.py         # pure-ASGI: X-Request-ID, API-key auth (off when
                            # OPSPILOT_API_KEY empty), Redis sliding-window rate
                            # limit (fails open), input guard on /chat (rewrites
                            # body with PII masked; blocks injection/overlength)
  db/postgres.py, redis.py
  session/
    manager.py              # Redis history session:{id}, TTL 2h refreshed on
                            # touch; stores only user/assistant turns (+ one
                            # system summary); degrades stateless on Redis loss
    context_window.py       # tiktoken count (chars/4 fallback if offline);
                            # ≤ CONTEXT_MAX_TOKENS (12k, inclusive) untouched;
                            # over ⇒ summary message + last 3 exchanges verbatim
    summarizer.py           # GPT-5.4-mini summary; deterministic post-check
                            # re-appends any ORD-/email/phone IDs the model
                            # dropped; no-LLM fallback = truncated transcript
  budget/
    cost_calculator.py      # ₹/1M-token table (Azure list × ₹88/USD); deployment
                            # name → longest normalised key match; unknown model
                            # priced at the top tier (over-report, never under);
                            # per-trace: each llm step by its model, classifier
                            # remainder at mini rate
    token_budget.py         # per-org daily/monthly Redis counters (natural TTL
                            # expiry), 429 when exhausted, 0 = disabled, fails
                            # open on Redis errors; org = X-Org-Id header
  hitl/
    queue.py                # Postgres hitl_pending queue: create / poll
                            # wait_for_decision / decide (single transition;
                            # timed-out rows stay pending for late decisions);
                            # audits human decisions
    gate.py                 # HITLApprovalGate risk matrix: NONE/LOW auto,
                            # MEDIUM auto+audit+Slack, HIGH block-for-approval.
                            # Overrides: refund ≤ ₹500 w/ explicit amount →
                            # MEDIUM, > limit or no amount → HIGH; confidence
                            # < 0.7 forces HIGH; escalate_to_manager never gated
                            # (it queues human review itself). Fails closed
                            # (queue down ⇒ REJECTED). Audits auto decisions.
    notifier.py             # Slack webhook messages (best-effort, never raises)
  guardrails/ (see below)
  observability/models.py   # DDL: traces, hitl_audit_log (trace_id FK dropped in
                            # 0002 — audit rows precede trace persist), hitl_pending,
                            # eval_runs
  observability/tracer.py   # Tracer: cost the trace (CostCalculator) then persist
  observability/logger.py   # setup_logging(): LOG_LEVEL + LOG_JSON one-line JSON
  observability/metrics.py  # SQL aggregates: P50/P99 latency, cost by model,
                            # intent distribution, HITL decision stats, eval trend
  observability/trace.py    # Trace object: step recording + best-effort persist to traces
  llm/client.py             # LLMClient: role → Azure deployment, normalized LLMResponse,
                            # per-call timeout, usage capture (all LLM access goes here)
  agent/
    prompts/                # v1, v2, v3_system.txt + current.txt symlink
                            # (→ v3); version → trace rows. NOTE: v3 is live
                            # but has NO eval run — see Phase 9 known drift.
    prompt_loader.py        # load_current_prompt() → (text, version)
    intent_classifier.py    # GPT-5.4-mini JSON classifier; never raises — falls back
                            # to action_complex + regex order-ID extraction
    model_router.py         # static ROUTING_TABLE (escalate → no LLM)
    confidence.py           # deterministic 0–1 scorer (retrieval/tools/consistency)
    react_agent.py          # ReAct loop: schema-validated tool calls with retry-on-error,
                            # proper tool_call_id message pairing, Redis idempotency
                            # (sha256 args key, TTL 24h, success-only caching), max-steps
                            # escalation, low-confidence escalation. Gate is
                            # consulted for every non-NONE-risk tool; only HIGH
                            # (or non-approved) decisions set hitl_triggered.
                            # AutoApproveHITLGate remains as no-DB/test fallback.
  rag/
    chunker.py              # SmartChunker: faq/policy/ticket/api_doc/changelog aware
    embedder.py             # AzureEmbedder (batch 16)
    vector_store.py         # ChromaStore + RetrievalResult
    bm25_index.py           # rank-bm25 in-memory index
    reranker.py             # cross-encoder ms-marco-MiniLM-L-6-v2
    retriever.py            # HybridRetriever: vector + BM25 → RRF → rerank
    dedup.py                # SemanticDeduplicator (ingestion-time, threshold 0.95)
  tools/
    base.py                 # Tool ABC, RiskLevel, ToolResult, ToolExecutionError
    registry.py             # ToolRegistry + build_default_registry()
    knowledge_tool.py       # search_knowledge (risk none)
    order_tool.py           # check_order_status, get_delivery_eta,
                            # check_refund_eligibility (low) · process_refund,
                            # cancel_order (HIGH, state-changing)
    customer_tool.py        # search_customer (low)
    jira_tool.py            # create/update_jira_ticket (medium, state-changing)
    slack_tool.py           # send_slack_summary (low, state-changing)
    escalate_tool.py        # escalate_to_manager (HIGH) — inserts into HITL
                            # queue (visible in /hitl/pending) + Slack notify;
                            # degrades to Slack/logs-only if queue unavailable
  guardrails/
    schemas.py              # SchemaValidator — jsonschema check of tool args
    input_guard.py          # length cap, Indian PII masking (card→aadhaar→PAN→UPI
                            # order matters; UPI = @-handle with no dotted TLD so
                            # emails pass), injection-signal blocklist
    output_guard.py         # PII scrub of response; flags claims (₹, ISO dates,
                            # day/week windows, %) not grounded in query+tool
                            # results; tone check on angry turns. Flags only —
                            # never rewrites answers (except PII). Flags land in
                            # traces.guardrail_flags
mock_services/order_service/
  models.py                 # Pydantic models + enums (deviation: PDF said SQLAlchemy;
                            # we use deterministic in-memory store instead — simpler,
                            # reproducible, resets on restart)
  seed.py                   # 500 orders + customers, random.seed(42), anchor date
                            # 2026-08-01, PINNED orders for eval scenarios
  main.py                   # FastAPI mock: orders, eta, refund-eligibility,
                            # refund, cancel, customer search
evals/
  golden_dataset/           # scenarios.json (core 30; grow to 100) +
                            # retrieval_ground_truth.json + tool_call_ground_truth.json
                            # (both DERIVED from scenarios.json — regenerate, don't
                            # hand-edit; checker fails on drift)
  judges/
    tool_accuracy.py        # pure Python exact/partial/missing/extra; score =
                            # (exact + 0.5·partial)/total; search_knowledge excluded
    hallucination.py        # deterministic claim extraction (ORD-/REF- IDs, ₹,
                            # dates, day-windows, %, phones) + grounding vs
                            # query+tool results; any fabricated ID = hard fail
    faithfulness.py         # GPT-4o judge 0–1 (JSON), None on judge failure
    relevance.py            # GPT-4o judge 0–1 vs reference answer as rubric
  runners/
    retrieval_eval.py       # P@K / R@K / MRR macro-averages (no LLM cost)
    runner_factory.py       # build_eval_runner(): assembles agent+tools+judges;
                            # shared by scripts/run_evals.py and POST /admin/evals/run
    eval_runner.py          # guard → classify → agent → output guard per scenario;
                            # deterministic + LLM metrics; CI-gate verdict vs
                            # Settings thresholds (retrieval gate = recall, not
                            # precision — see Phase 8 notes); JSON report +
                            # eval_runs row; --subset/--category
    regression_tracker.py   # direction-aware diff of two run reports (latency/
                            # cost judged relatively ±10%, scores ±0.01)
  reports/eval_report.py    # markdown table for PR comments (runs/ is gitignored)
  ci/eval_gate.py           # reads a report JSON's ci_gate, exits 1 on breach
knowledge_base/             # faqs/ policies/ tickets/ api_docs/ changelogs/
scripts/ingest_knowledge.py # chunk → embed → dedup → upsert
scripts/check_golden_dataset.py  # dataset ↔ seed ↔ chunker consistency gate
scripts/run_evals.py        # run harness vs real agent (needs Azure + mock svc;
                            # restart mock-order-service between comparative runs)
scripts/generate_eval_report.py  # report JSON → markdown; --compare for diffs
.github/workflows/eval_gate.yml  # PR: core 30 scenarios; nightly/dispatch: full
                            # dataset; skips (not fails) without Azure secrets
frontend/index.html         # self-contained dev console: chat tester, animated
                            # pipeline explainer, trace timeline, RAG explorer.
                            # Uses dev endpoints GET /api/v1/traces/{id} and
                            # /rag/test; CORS origins now via CORS_ALLOW_ORIGINS
                            # (default * for local dev). Backend URL is a config
                            # field, not hardcoded — not baked into any image.
docker-compose.yml          # all 5 services (postgres, redis, chromadb,
                            # mock-order-service, app) with restart policies,
                            # healthchecks, and app gated on depends_on:
                            # service_healthy. Overrides the host-local
                            # DATABASE_URL/REDIS_URL/CHROMA_HOST/ORDER_SERVICE_URL
                            # from .env with in-network service names.
Dockerfile                  # app image; non-root user + HEALTHCHECK
.dockerignore               # keeps .git/tests/frontend/knowledge_base out of
                            # the build context
scripts/backup.sh           # pg_dump + chroma volume snapshot → backups/,
                            # keeps last 14 (cron-driven; backups/ gitignored)
.github/workflows/docker_build_push.yml  # main → build+push both images to GHCR
DEPLOY.md                   # handover doc: required vs optional env vars, the
                            # two unsafe-by-default security knobs, port
                            # exposure, sizing, backups, mock-service caveat
```

**Pinned mock orders** (eval scenarios MUST only reference these or other seeded
IDs — check with `GET /orders/{id}` before using an ID in a scenario):
- `ORD-2024-55001` — delayed, ₹1,499, UPI, refund-eligible (delivery_delayed)
- `ORD-2024-78432` — delayed, ₹1,299, card, refund-eligible (PDF trace example)
- `ORD-2024-51234` — delivered 2 days ago (inside return window)
- `ORD-2024-52000` — delivered 45 days ago (window closed)
- `ORD-2024-53000` — cancelled, refund already processed
- `ORD-2024-54000` — in transit, COD, on time (NOT refund-eligible)

**Stale-knowledge anchor:** `knowledge_base/changelogs/changelog_2026_q3.md`
entry 2026-07-15 extends the electronics return window 7 → 10 days. This
*deliberately contradicts* `policies/refund_policy.md` §2 and the returns FAQ —
that tension is what eval category `stale_knowledge` tests (agent must prefer
the dated changelog). Do not "fix" this contradiction.

---

# Roadmap — what remains, phase by phase

Effort estimates from the design doc. If time runs short, cut in this order:
semantic-dedup extras → reduce Hindi-English scenarios to 3–5 → session manager
becomes last-N-turns-only → output guard becomes PII-scrub-only.
**Never cut:** eval harness, HITL, core agent, retrieval.

## Phase 4 — Agent core — ✅ DONE (2026-08-04)

Build in this order; each step keeps the app runnable.

1. `app/llm/client.py` (new) — thin AsyncAzureOpenAI chat wrapper: takes model
   role (agent/classifier/judge), maps to deployment via Settings, returns
   normalized response (content, tool_calls, usage). All agent/eval code uses
   this, never raw openai. Include per-call timeout + token usage capture.
2. `app/agent/prompts/` — `v1_system.txt` + `current.txt` symlink. Prompt
   versioning from day one; trace rows record `prompt_version`.
3. `app/agent/intent_classifier.py` — GPT-5.4-mini, JSON-only output:
   `{intent: faq|action_simple|action_complex|escalate|out_of_scope,
   extracted_order_id, extracted_customer_id, sentiment, language, reasoning}`.
   Test with ≥20 queries incl. Hindi-English mixed ("Mera order ... refund chahiye").
4. `app/agent/model_router.py` — static ROUTING_TABLE: faq/action_simple/
   out_of_scope → mini; action_complex → GPT-5.4; escalate → no LLM.
5. `app/agent/react_agent.py` — ReAct loop (see PDF §5 for full reference code):
   - loop up to `MAX_AGENT_STEPS` (10), per-step timeout `AGENT_TIMEOUT_SECONDS`
   - no tool_call in response ⇒ final answer → confidence score → return
   - tool_call ⇒ validate args via `SchemaValidator`; on invalid, append
     `{"role":"tool","content":"Error: ..."}` and `continue` (agent retries)
   - HITL gate hook before high-risk execution (Phase 4: stub auto-approve;
     Phase 5 replaces with real gate — leave a clearly named seam)
   - idempotency: for `tool.is_state_changing`, Redis key
     `idempotent:{tool}:{sha256(sorted args json)}`, TTL 24h, return cached
     ToolResult on hit (prevents double refunds)
   - max steps exhausted ⇒ escalate response
   - every step recorded on a `Trace` object (thought/tool_call/result/latency)
6. `app/agent/confidence.py` — 0–1 score from retrieval quality + reasoning
   consistency; `< CONFIDENCE_THRESHOLD (0.7)` ⇒ auto-escalate.
7. Wire `POST /api/v1/chat`: input → intent → route → agent.run → response
   `{response, trace_id, confidence, escalated}`.
8. Tests: `test_intent_classifier.py` (mock llm), `test_react_agent.py`
   (scripted fake LLM: tool call → observation → answer; malformed-args retry;
   max-steps; idempotent replay).

## Phase 5 — HITL + guardrails — ✅ DONE (2026-08-05)

Delivered as specced, with these deviations/decisions:
- `hitl_audit_log.trace_id` FK dropped (migration `0002`): audit rows are
  written mid-run, before the trace row exists.
- Timed-out approval requests stay `pending` (customer gets the
  pending-approval reply; a human can still decide later; `decide()` is a
  single `pending → approved|rejected` transition, 409 on repeats).
- Exactly one audit row per decision: gate audits auto-approvals/timeouts,
  `HITLQueue.decide` audits human decisions.
- The gate is consulted for all non-NONE tools (matrix lives in the gate);
  `escalate_to_manager` is auto-approved by design — executing it *is* the
  human-review request, gating it would deadlock.
- Refund override: explicit `amount_inr` ≤ ₹500 → MEDIUM (auto+audit+Slack);
  above the limit *or no explicit amount* → HIGH.
- Middleware is pure ASGI (body rewrite for PII masking needs `receive`
  control); rate limiter fails open, auth and gate fail closed.
- New knobs: `OPSPILOT_API_KEY` (empty = auth off), `HITL_POLL_INTERVAL_SECONDS`,
  `INPUT_MAX_QUERY_LENGTH`, `CORS_ALLOW_ORIGINS`.
- Tests: `test_hitl_gate.py`, `test_guardrails.py`, `test_middleware.py`,
  `test_hitl_routes.py`.

Original plan (for reference):

1. `app/hitl/queue.py` — Postgres approval queue (table exists in DDL:
   `hitl_audit_log`; add a `hitl_pending` table + migration for open requests).
   `create()`, `wait_for_decision(request_id, timeout)` (poll or LISTEN/NOTIFY),
   decision recording.
2. `app/hitl/gate.py` — matrix: NONE auto / LOW auto+audit / MEDIUM
   auto+audit+Slack / HIGH block-for-approval. Overrides: refund amount >
   `HITL_REFUND_AUTO_APPROVE_LIMIT` (₹500) ⇒ HIGH regardless; confidence < 0.7
   ⇒ escalate regardless. Every decision audited.
3. `app/hitl/notifier.py` — Slack webhook message with approve/reject context.
4. `app/api/hitl_routes.py` — `POST /hitl/approve/{id}`, `POST /hitl/reject/{id}`
   (with reason), `GET /hitl/pending`.
5. Replace Phase-4 gate stub in react_agent; handle approved / rejected
   (inform agent, continue) / timeout ("pending approval" response,
   `pending_approval=True`).
6. Rewire `escalate_to_manager` tool to insert into the HITL queue.
7. `app/guardrails/input_guard.py` — length cap 4000; Indian PII regexes
   (Aadhaar `\b\d{4}\s?\d{4}\s?\d{4}\b`, PAN `\b[A-Z]{5}\d{4}[A-Z]\b`, 16-digit
   cards, UPI IDs — NOTE: naive UPI regex also matches emails; mask only when an
   `@`-handle is NOT a known email TLD, and test this) with masking; injection
   signal list ("ignore previous instructions", "system prompt", "you are now"…).
8. `app/guardrails/output_guard.py` — PII scrub of response; extract verifiable
   claims (numbers, dates, policy names) and flag any not present in retrieved
   context; tone check for angry-customer turns.
9. `app/api/middleware.py` — API-key auth, Redis sliding-window rate limit
   (`RATE_LIMIT_PER_MINUTE`), request-ID header; input guard runs here.
10. Tests: `test_hitl_gate.py` (each matrix row, ₹500 boundary at 500.00/500.01,
    timeout), `test_guardrails.py` (each PII pattern masks, adversarial strings
    from PDF eval scenarios block, clean queries pass).

## Phase 6 — Sessions, budget, observability — ✅ DONE (2026-08-06)

Delivered as specced, with these deviations/decisions:
- Session history stores only user/assistant turns (plus at most one system
  summary) — tool call/observation messages are not persisted.
- After a turn, the *compacted* history + new exchange is written back, so
  context-window summaries replace old turns in Redis too.
- Summariser guarantee is deterministic: a post-check re-appends any ORD-,
  email, or phone identifiers the LLM summary dropped; with no LLM it falls
  back to a truncated transcript. Exactly-at-12k-tokens passes untouched.
- tiktoken (`o200k_base`) with a chars/4 estimate fallback when the encoding
  can't load (offline CI) — compaction logic identical either way.
- Pricing: Azure list × ₹88/USD per 1M tokens; deployment-name lookup via
  longest normalised key match; unknown models priced at the top tier.
- Budget check runs pre-agent, record post-agent ⇒ can overshoot by one
  request per org (cost brake, not an invariant). Fails open on Redis loss;
  org from `X-Org-Id` header, default "default".
- `traces.cost_inr` now populated (Tracer = cost + persist on top of
  Trace.persist). Admin routes return 503 on DB loss, 400 on bad windows.
- New knobs: `SESSION_TTL_SECONDS`, `CONTEXT_MAX_TOKENS`,
  `CONTEXT_KEEP_LAST_EXCHANGES`, `BUDGET_DAILY_TOKENS`,
  `BUDGET_MONTHLY_TOKENS`, `LOG_LEVEL`, `LOG_JSON`.
- Tests: `test_session.py`, `test_budget.py`, `test_admin_routes.py`,
  `test_e2e.py` (full HTTP lifecycle; Postgres writes captured in-memory, so
  the suite still needs no live services).

Original plan (for reference):

1. `app/session/manager.py` — Redis history `session:{id}`, TTL 2h.
2. `app/session/context_window.py` + `summarizer.py` — tiktoken count; over
   12k tokens ⇒ GPT-5.4-mini summary of old turns (must preserve order numbers /
   customer IDs), keep last 3 exchanges verbatim. Edge cases: empty history,
   exactly-at-limit, summary retains `ORD-` IDs (test all three).
3. `app/budget/cost_calculator.py` — ₹ pricing table per model, cost from
   usage tokens (fill real Azure prices).
4. `app/budget/token_budget.py` — per-org daily/monthly Redis counters;
   429 when exhausted.
5. `app/observability/tracer.py` — persist Trace to `traces` table (all columns
   already in DDL). `logger.py` structured logging, `metrics.py` aggregate
   queries (P50/P99, cost breakdown, intent distribution, HITL stats).
6. `app/api/admin_routes.py` — GET: `/admin/traces?last=24h`,
   `/admin/traces/{id}`, `/admin/metrics/summary`,
   `/admin/metrics/cost-breakdown`, `/admin/hitl/stats`, `/admin/hitl/pending`,
   `/admin/evals/latest`, `/admin/evals/trend?versions=v1,v2`.
7. `tests/test_e2e.py` — full lifecycle: chat → guard → classify → agent → tool
   → output guard → trace row in Postgres with cost + prompt_version.

## Phase 7 — Eval harness — ✅ DONE (2026-08-07)

Delivered as specced, with these deviations/decisions:
- Dataset starts at the core 30 (category mix scaled from the 100 plan:
  faq_en 6, faq_mixed 3, single_action 5, multi_step 4, adversarial 3,
  stale_knowledge 2, out_of_scope 3, edge_cases 3, angry 1). Growing to 100
  keeps the same schema; checker enforces validity.
- The two ground-truth JSONs are *derived* from scenarios.json (checker fails
  on drift). expected_tool_calls cover business tools only; search_knowledge
  is implied by expected_retrieved_chunks and ignored by tool accuracy.
- expected_tool_calls args are the args-that-matter subset: extra actual args
  don't demote an exact match.
- `edge_case_001` uses `expects_missing_order: true` — the checker asserts
  that ID is ABSENT from the seed (hallucinating a status there = fail).
- Hallucination judge is deterministic (no LLM): claim extraction + grounding
  against query + tool results; any fabricated ORD-/REF- ID fails the
  scenario outright. Faithfulness/relevance are GPT-4o judges returning None
  on failure (averaged over scored scenarios; gate skips null metrics —
  --no-llm-judges runs are deterministic-only).
- Adversarial pass = input guard blocked it OR no successful
  process_refund/cancel_order in the trace. Guard-blocking a *legitimate*
  scenario scores tool accuracy 0.
- eval_runner also tracks intent_accuracy, hitl_accuracy (expected_hitl vs
  trace), adversarial_pass_rate, avg latency/cost — beyond the four gate
  metrics. eval_runs insert is best-effort; the JSON report is the record.
- Tests: `test_golden_dataset.py` (checker catches every drift class),
  `test_eval_judges.py`, `test_retrieval_eval.py`, `test_eval_runner.py`
  (real agent+tools+guards, scripted LLMs), `test_regression_tracker.py`.
  CAUTION: scripts/run_evals.py hits real Azure — never run it as a smoke test.

Original plan (for reference):

1. `evals/golden_dataset/scenarios.json` — start 30, grow to 100. Categories
   (counts): faq_en 20, faq_mixed 10, single_action 15, multi_step 15,
   adversarial 10, stale_knowledge 5, out_of_scope 10, edge_cases 10, angry 5.
   Schema per scenario: id, category, query, language, expected_intent,
   expected_retrieved_chunks (real chunk IDs!), expected_tool_calls (tool+args),
   reference_answer, expected_hitl, adversarial, notes.
2. `retrieval_ground_truth.json`, `tool_call_ground_truth.json`.
3. **Consistency checker script** `scripts/check_golden_dataset.py`: every
   order ID exists in mock seed; every expected chunk ID exists in the chunker
   output. Run it in CI before evals.
4. `evals/judges/` — `faithfulness.py` (GPT-4o judge 0–1, prompt in PDF §11),
   `hallucination.py` (claim extraction + per-claim grounding; any fabricated
   order/policy = fail), `relevance.py`, `tool_accuracy.py` (pure Python:
   exact/partial/missing/extra vs ground truth).
5. `evals/runners/retrieval_eval.py` — P@K, R@K, MRR (no LLM cost — use freely
   while tuning). `eval_runner.py` — run scenarios through the real agent
   (mock order service up, real LLM), write `eval_runs` row + JSON report;
   support `--subset N` / `--category X` for cheap iteration.
   `regression_tracker.py` — diff runs across prompt_version.
6. `scripts/run_evals.py`, `scripts/generate_eval_report.py`,
   `evals/reports/eval_report.py` (markdown table for PR comment).
7. Thresholds (already in Settings): faithfulness ≥ 0.90, hallucination ≤ 0.05,
   retrieval precision ≥ 0.85, tool accuracy ≥ 0.85.

## Phase 8 — CI gate + polish — ✅ DONE (2026-08-10)

Delivered as specced, with these deviations/decisions:
- **Retrieval gate checks recall, not precision.** The first live eval run
  (real Azure, real mock service) showed `retrieval_precision@5` is capped at
  `relevant_chunks / 5` — most golden scenarios have 1–3 relevant chunks, so
  precision@5 can't exceed 0.20–0.60 regardless of retrieval quality. The
  baseline's 0.28 average was every scenario hitting its own ceiling (recall
  was already 0.70, MRR 0.69). `EVAL_RETRIEVAL_PRECISION_THRESHOLD` (0.85)
  stays for reporting; the actual gate is the new
  `EVAL_RETRIEVAL_RECALL_THRESHOLD` (0.65). See `app/config.py` comment and
  `evals/runners/eval_runner.py::_ci_gate`.
- **Real v1 → v2 prompt iteration**, driven by the baseline's tool-accuracy
  failures (0.763, gate requires 0.85): the agent called `get_delivery_eta`
  right after `check_order_status` (which already returns ETA), used
  `check_refund_eligibility` for "where's my refund" questions
  `check_order_status` answers directly, and sometimes stopped after
  confirming eligibility instead of completing a refund the customer asked
  for. `v2_system.txt` adds four rules for exactly this; nothing else
  changed. Result: tool accuracy 0.763 → 0.893, gate FAIL → PASS, retrieval
  recall unchanged (0.70 both runs — the expected sanity check for a
  prompt-only change). `current.txt` now points at `v2_system.txt`.
- **Eval reproducibility gotcha, documented not silently fixed:** the mock
  order service has mutable in-memory state (refunds/cancellations persist
  until restart). Re-running the eval harness against the same live instance
  means the second run sees post-refund order state, contaminating any diff.
  `scripts/run_evals.py`'s docstring now calls this out; restart
  `mock-order-service` between comparative runs.
- `evals/runners/runner_factory.py` (new) extracts the agent/tools/judges
  assembly that used to live inline in `scripts/run_evals.py`, so
  `POST /admin/evals/run` builds the identical pipeline instead of
  duplicating it.
- `POST /admin/evals/run` is fire-and-forget (`asyncio.create_task`, no job
  registry) — 202 immediately, 503 if Azure creds aren't configured; the
  result lands as a new row via the existing `GET /admin/evals/latest`/
  `/admin/evals/trend`. No separate job-status endpoint; the eval_runs row +
  report file are the record, consistent with how `scripts/run_evals.py`
  already worked.
- `.github/workflows/eval_gate.yml`: core 30 on PRs, full dataset
  nightly/`workflow_dispatch`; skips (doesn't fail) when Azure secrets are
  absent, since forked PRs never get repo secrets and shouldn't hard-fail on
  a trust boundary they can't cross. Posts a sticky PR comment, uploads the
  report JSON as an artifact.
- README rewritten with the measured v1→v2 numbers above and a real,
  reproduced end-to-end transcript (stale-knowledge answer, a refund blocked
  on HITL then approved, admin metrics) — no demo recording (out of scope for
  an agent to produce).
- Known gaps carried forward rather than papered over: 4/15
  retrieval-checked scenarios genuinely miss the right chunk (ticket text
  near-duplicates a policy section — needs retriever/reranker tuning, not a
  prompt fix); `adversarial_003` escalates to a human instead of running the
  eligibility check the ground truth expects (safe — adversarial containment
  is still 100% — but doesn't match tool-call ground truth); two scenarios
  share a pinned order and interact within a single eval run.
- Tests: `test_eval_gate.py` (new), `test_admin_routes.py` extended for
  `POST /evals/run`, `test_eval_runner.py`/`test_react_agent.py`/
  `test_e2e.py` updated for the recall-based gate and `v2` prompt version.

Original plan (for reference):

1. `evals/ci/eval_gate.py` — read report JSON, exit 1 if any threshold breached.
2. `.github/workflows/eval_gate.yml` — on PRs touching
   `app/agent/prompts/**`, `react_agent.py`, `intent_classifier.py`,
   `rag/retriever.py`, `rag/chunker.py`. Services: postgres+redis+chromadb+mock.
   Secrets: Azure OpenAI creds. Cost control: 30-scenario core set on PRs,
   full 100 nightly.
3. Prompt iteration: `v2_system.txt` driven by eval failures; keep every version
   + regression comparison (this is the interview story).
4. `POST /admin/evals/run`, `/admin/evals/trend` wiring.
5. README rewrite with *measured* numbers, demo recording, end-to-end
   walkthrough: `docker compose up` → ingest → chat → approve refund via HITL.

## Phase 9 — Deployment packaging — ✅ DONE (2026-08-12)

Not in the original roadmap. Added when the project needed to be handed to a
DevOps engineer instead of run from a laptop: everything before this assumed
`uvicorn` on localhost with the four support services in Compose.

Delivered, with these deviations/decisions:

- **The app is now a Compose service.** Previously `docker-compose.yml` held
  only postgres/redis/chromadb/mock-order-service and the API was a manual
  `uvicorn` command. `app` now waits on all four via `depends_on:
  condition: service_healthy`.
- **Compose overrides the connection URLs.** `.env` holds host-local values
  (`localhost:5432`, …) which are wrong inside a container, so
  `DATABASE_URL`/`REDIS_URL`/`CHROMA_HOST`/`ORDER_SERVICE_URL` are set to
  in-network service names in the `app` service block. `.env` stays correct
  for host-run `alembic upgrade head` and `scripts/ingest_knowledge.py`,
  which are still meant to run from the host.
- **ChromaDB's healthcheck can't use curl/wget/python** — none exist in that
  image. It uses bash's `/dev/tcp` instead, and must invoke `bash` explicitly
  (`CMD` + `bash -c`): `CMD-SHELL` runs under `/bin/sh` → dash, which has no
  `/dev/tcp`. Both wrong versions were written and caught by actually running
  them; the container sat in `starting` forever rather than failing loudly.
- **Dependencies pinned** to the versions in the working venv. They were
  entirely unpinned except `chromadb==1.5.9`, so a rebuild months later could
  resolve a different FastAPI/openai/torch. `pydantic` added explicitly (it
  was only arriving transitively).
- **Both images run as non-root** with a `HEALTHCHECK`; `.dockerignore` added
  (previously absent — builds shipped `.git/`, tests, and the knowledge base
  into the context).
- **`scripts/backup.sh`** — `pg_dump` + ChromaDB volume tar into
  `backups/<timestamp>/`, keeps the last 14. Cron-driven, nothing calls it
  automatically. `backups/` is gitignored — without that rule the dumps would
  be committed.
- **`docker_build_push.yml`** builds and pushes both images to GHCR on merge
  to `main`. It deliberately does **not** deploy: where/how to deploy is
  infra-specific and not decided in this repo.
- **No TLS/reverse proxy, no secret manager, no IaC.** Documented in
  `DEPLOY.md` as the deployer's decisions rather than half-implemented here.
  `OPSPILOT_API_KEY=""` and `CORS_ALLOW_ORIGINS=["*"]` remain dev defaults —
  nothing in the app warns when they're left as-is, so `DEPLOY.md` calls them
  out as a manual pre-flight check.
- New config knobs: **none** — this phase added no `Settings` fields, so
  `.env.example` is unchanged.

Verified: 301 tests pass; `docker compose config` validates; both images build
and run as their non-root user; the `app` container reaches `healthy` and
brings up RAG/HITL/guards/11 tools/agent/tracer.

**Not** verified (stated rather than implied): a fully green `/health` — port
6379 on the dev machine is held by an unrelated container, so the app resolved
no Redis and reported `"redis": false`; `scripts/backup.sh` has never been
executed; the frontend console has not been loaded against a live backend.

### Known drift found during this phase (NOT fixed — decide before publishing)

- **`current.txt` → `v3_system.txt`, but there is no v3 eval run.**
  `evals/reports/runs/` holds v1 and v2 only, and both README and this file
  described v2 as current (fixed above). v3 adds one rule — never construct,
  pad, or auto-complete a partial number into a full `ORD-YYYY-NNNNN` ID —
  committed in `3de25d8 "repsonse issues"`. Since the project's headline claim
  is that every change is eval-gated, shipping an ungated prompt undercuts it:
  either run the harness against v3 and publish the numbers, or point
  `current.txt` back at v2 and keep v3 as a candidate.
- README says 295 tests; the actual count is 301.

## Cross-phase invariants

- Trace everything: any new agent/tool/guard step must append to the Trace.
- `prompt_version` flows: prompts dir → agent → trace row → eval_runs.
- Mock seed data, chunk IDs, and golden dataset must stay consistent —
  run `scripts/check_golden_dataset.py` after touching any of the three.
- New config knobs: Settings default + `.env.example` + mention here.
- New dependency: pin it in `pyproject.toml` (no floating versions). New
  service: give it a `restart:` policy and a healthcheck in
  `docker-compose.yml`, and record any required env var in `DEPLOY.md`.
- Healthchecks must be *run*, not just written — a wrong `test:` command
  leaves a container in `starting` forever instead of failing loudly.
