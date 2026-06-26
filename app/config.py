from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://opspilot:opspilot@localhost:5432/opspilot"
    REDIS_URL: str = "redis://localhost:6379/0"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8100

    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_VERSION: str = "2025-04-01-preview"
    AZURE_DEPLOYMENT_GPT54: str = ""
    AZURE_DEPLOYMENT_GPT54_MINI: str = ""
    AZURE_DEPLOYMENT_GPT4O: str = ""
    AZURE_DEPLOYMENT_EMBEDDING: str = ""

    MODEL_AGENT: str = ""
    MODEL_CLASSIFIER: str = ""
    MODEL_JUDGE: str = ""
    MODEL_VERIFIER: str = ""
    MODEL_GUARDRAILS: str = ""
    MODEL_SUMMARIZER: str = "gpt-5.4-mini"
    MODEL_QUERY_REWRITER: str = "gpt-5.4-mini"

    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSIONS: int = 3072
    RETRIEVAL_TOP_K: int = 10
    RERANK_TOP_K: int = 5
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    MAX_AGENT_STEPS: int = 10
    AGENT_TIMEOUT_SECONDS: int = 45
    CONFIDENCE_THRESHOLD: float = 0.7

    HITL_HIGH_RISK_ACTIONS: list[str] = Field(default_factory=list)
    HITL_REFUND_AUTO_APPROVE_LIMIT: float = 500.0
    HITL_APPROVAL_TIMEOUT_MINUTES: int = 30
    RATE_LIMIT_PER_MINUTE: int = 60

    JIRA_BASE_URL: str = ""
    JIRA_EMAIL: str = ""
    JIRA_API_TOKEN: str = ""
    JIRA_PROJECT_KEY: str = ""
    SLACK_WEBHOOK_URL: str = ""
    ORDER_SERVICE_URL: str = "http://localhost:8001"

    EVAL_FAITHFULNESS_THRESHOLD: float = 0.90
    EVAL_HALLUCINATION_MAX: float = 0.05
    EVAL_HALLUCINATION_MAX_RATE: float = 0.05
    EVAL_RETRIEVAL_PRECISION_THRESHOLD: float = 0.85

    ENABLE_CHAIN_OF_VERIFICATION: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
