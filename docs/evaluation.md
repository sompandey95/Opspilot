# Evaluation

OpsPilot includes a golden-set runner for measuring agent behaviour across
retrieval, tool use, approvals, safety, answer quality, latency, and cost.

## Dataset

`evals/golden_dataset/scenarios.json` contains 30 scenarios across nine
categories:

| Category | Scenarios |
|---|---:|
| English FAQ | 6 |
| Mixed Hindi-English FAQ | 3 |
| Single action | 5 |
| Multi-step action | 4 |
| Adversarial | 3 |
| Stale knowledge | 2 |
| Out of scope | 3 |
| Edge cases | 3 |
| Angry customer | 1 |

Fifteen scenarios include retrieval ground truth, fourteen expect business
tool calls, and four expect human approval.

The consistency checker verifies that referenced order IDs match the
deterministic mock service, expected chunk IDs match the chunker output, and
tool arguments conform to registered schemas.

## Metrics

| Metric | Method | Gate |
|---|---|---:|
| Tool accuracy | Deterministic comparison with expected calls | `>= 0.85` |
| Hallucination rate | Deterministic claim and fabricated-ID checks | `<= 0.05` |
| Faithfulness | Independent LLM judge | `>= 0.90` |
| Retrieval recall@K | Macro average over retrieval scenarios | `>= 0.65` |
| Precision@K, MRR, relevance, intent and HITL accuracy, latency, cost | Reported | Not gated |

Retrieval precision is reported but not gated. With `K=5` and only one to
three labelled relevant chunks in most scenarios, precision has a low
structural ceiling even when the relevant chunks are returned. Recall and MRR
better represent whether required evidence was found.

## Historical v1 to v2 result

The following values were recorded during local `v1` and `v2` development
runs. They describe those prompt versions, not the currently active `v3`
prompt. Run and preserve a fresh `v3` report before presenting these values as
current system performance.

| Metric | v1 | v2 | Change |
|---|---:|---:|---:|
| Tool accuracy | 0.763 | 0.893 | +13.0 percentage points |
| Faithfulness | 0.936 | 0.939 | +0.003 |
| Hallucination rate | 3.6% | 3.6% | No change |
| Retrieval recall@5 | 0.70 | 0.70 | No change |
| Average cost/query | INR 0.4286 | INR 0.3907 | -9% |
| Gate | Fail | Pass | Pass |

The failing `v1` traces showed redundant delivery lookups, an incorrect tool
choice for refund-status questions, and refund flows that stopped after
eligibility instead of performing the requested action. Four targeted rules
in `v2_system.txt` addressed those behaviours. Retrieval recall staying
unchanged was the expected result for a prompt-only modification.

`v3_system.txt` adds a stricter rule for incomplete or fabricated order IDs.
It is the active prompt through `app/agent/prompts/current.txt`.

## Running evaluations

Start dependencies, apply migrations, ingest the knowledge base, and reset the
mutable order service before comparative runs:

```bash
docker compose up -d
alembic upgrade head
python scripts/ingest_knowledge.py --reset
docker compose restart mock-order-service
python scripts/check_golden_dataset.py
python scripts/run_evals.py
python evals/ci/eval_gate.py --latest
```

The live runner calls Azure OpenAI and incurs usage charges. For a cheaper
diagnostic pass, use its subset/category flags or disable LLM judges as exposed
by `python scripts/run_evals.py --help`.

The mock order service keeps refunds and cancellations in memory until it is
restarted. Always reset it between baselines and candidates or the second run
will observe state created by the first.

## CI behaviour

`.github/workflows/eval_gate.yml` runs the core dataset on eligible pull
requests and the full dataset on manual runs. Fork pull requests without
repository secrets skip the live evaluation. Manual runs fail when required
Azure credentials are missing, so a successful manual run represents an
executed gate. Add a schedule only after the repository secrets are configured.

JSON reports under `evals/reports/runs/` are intentionally ignored because
they can be large and environment-specific. Publish a sanitized baseline or a
GitHub Actions artifact when reporting current performance.
