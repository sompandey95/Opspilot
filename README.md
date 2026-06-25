# OpsPilot

OpsPilot is a FastAPI service scaffold for agent workflows, retrieval, observability, and integrations.

## Project Layout

```text
app/
  api/
  db/
  observability/
evals/
knowledge_base/
mock_services/
scripts/
tests/
alembic/
```

## Configuration

Copy `.env.example` to `.env` when you are ready to run the service locally.

## Local Development

Dependencies are declared in `pyproject.toml` but are not installed yet.

```bash
uvicorn app.main:app --reload
```

## Health Check

```bash
curl http://localhost:8000/health
```
