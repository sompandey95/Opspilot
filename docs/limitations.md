# Limitations

OpsPilot is a portfolio and evaluation project, not a deployed customer-support
product. The following limitations define the current engineering boundary.

## Human approval lifecycle

High-risk actions keep the original HTTP request open while polling for a
decision. Long approvals can exceed client, proxy, or platform timeouts. A
production design should return an operation ID immediately and deliver the
decision through polling, webhooks, server-sent events, or a queue worker.

## Idempotency under concurrency

State-changing tool results use a Redis read-execute-write sequence. This
protects sequential retries but two concurrent identical requests can both
execute before either stores its result. Production protection should use an
atomic claim (`SET NX`), a distributed lock with careful expiry, or a database
uniqueness constraint around the business operation.

## Approval fallback

`ReActAgent` retains an auto-approving fallback gate for isolated tests and
development dependency injection. If the application cannot construct the
real gate at startup, this fallback can approve a high-risk call. A production
deployment must replace this with a fail-closed gate and a startup readiness
check.

## Guardrails

The input guard uses deterministic patterns for supported Indian PII and a
small prompt-injection phrase list. It is neither a comprehensive data-loss
prevention system nor a robust adversarial classifier. The output guard records
unsupported claims but does not suppress or rewrite them.

## Retrieval coverage

Historical `v2` evaluation missed the expected chunk in four of fifteen
retrieval-checked scenarios. Refund-related support tickets can outrank the
canonical policy section because their wording is highly similar. This needs
retrieval and reranker tuning rather than prompt changes.

## Evaluation state

Some golden scenarios share mutable order fixtures. Refunds and cancellations
persist for the life of the mock service, so comparative runs must restart it.
A larger dataset should allocate an independent fixture to every state-changing
scenario.

## External integrations

The order service is deterministic and local. Jira and Slack adapters perform
HTTP calls only when credentials are configured; automated tests mock external
network access. The project does not demonstrate production integration
throughput or third-party failure recovery.

## Development defaults

API-key authentication is disabled when `OPSPILOT_API_KEY` is empty, CORS
allows every origin by default, and several persistence failures intentionally
degrade to a reduced feature set. These defaults simplify local development
and must be tightened for deployment.

## Operational validation

The project has no production SLO history, load test, multi-region deployment,
or disaster-recovery exercise. Cost values depend on the pricing table in the
repository and should be reviewed before financial reporting.
