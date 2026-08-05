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

## Current state (Phases 1–4 complete)

```
app/
  config.py                 # Settings (Azure OpenAI, RAG, agent, HITL, eval knobs)
  main.py                   # lifespan: db, redis, RAG, tools, agent init (fault-tolerant)
  api/routes.py             # /health, /chat (wired: intent → route → agent), /rag/test
  db/postgres.py, redis.py
  observability/models.py   # DDL: traces, hitl_audit_log, eval_runs
  observability/trace.py    # Trace object: step recording + best-effort persist to traces
  llm/client.py             # LLMClient: role → Azure deployment, normalized LLMResponse,
                            # per-call timeout, usage capture (all LLM access goes here)
  agent/
    prompts/                # v1_system.txt + current.txt symlink; version → trace rows
    prompt_loader.py        # load_current_prompt() → (text, version)
    intent_classifier.py    # GPT-5.4-mini JSON classifier; never raises — falls back
                            # to action_complex + regex order-ID extraction
    model_router.py         # static ROUTING_TABLE (escalate → no LLM)
    confidence.py           # deterministic 0–1 scorer (retrieval/tools/consistency)
    react_agent.py          # ReAct loop: schema-validated tool calls with retry-on-error,
                            # proper tool_call_id message pairing, Redis idempotency
                            # (sha256 args key, TTL 24h, success-only caching), max-steps
                            # escalation, low-confidence escalation.
                            # AutoApproveHITLGate = the PHASE 5 SEAM (auto-approves HIGH
                            # risk, records decision on trace) — replace with app/hitl/gate.
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
    escalate_tool.py        # escalate_to_manager (HIGH) — Slack notify now;
                            # Phase 5 must wire it to the real HITL queue
  guardrails/
    schemas.py              # SchemaValidator — jsonschema check of tool args
mock_services/order_service/
  models.py                 # Pydantic models + enums (deviation: PDF said SQLAlchemy;
                            # we use deterministic in-memory store instead — simpler,
                            # reproducible, resets on restart)
  seed.py                   # 500 orders + customers, random.seed(42), anchor date
                            # 2026-08-01, PINNED orders for eval scenarios
  main.py                   # FastAPI mock: orders, eta, refund-eligibility,
                            # refund, cancel, customer search
knowledge_base/             # faqs/ policies/ tickets/ api_docs/ changelogs/
scripts/ingest_knowledge.py # chunk → embed → dedup → upsert
frontend/index.html         # self-contained dev console: chat tester, animated
                            # pipeline explainer, trace timeline, RAG explorer.
                            # Uses dev endpoints GET /api/v1/traces/{id} and
                            # /rag/test; CORS is dev-open until Phase 5.
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

## Phase 5 — HITL + guardrails (~4 days)  ← NEXT

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

## Phase 6 — Sessions, budget, observability (~4 days)

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

## Phase 7 — Eval harness (~5 days, the differentiator)

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

## Phase 8 — CI gate + polish (~3 days)

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

## Cross-phase invariants

- Trace everything: any new agent/tool/guard step must append to the Trace.
- `prompt_version` flows: prompts dir → agent → trace row → eval_runs.
- Mock seed data, chunk IDs, and golden dataset must stay consistent —
  run `scripts/check_golden_dataset.py` after touching any of the three.
- New config knobs: Settings default + `.env.example` + mention here.
