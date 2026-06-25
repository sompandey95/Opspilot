"""Azure OpenAI embedder with automatic batching."""
from __future__ import annotations

import openai

from app.config import Settings

_MAX_BATCH = 16  # Azure OpenAI embeddings API limit per request


class AzureEmbedder:
    def __init__(self, settings: Settings) -> None:
        self._client = openai.AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        self._deployment = settings.AZURE_DEPLOYMENT_EMBEDDING
        self._dimensions = settings.EMBEDDING_DIMENSIONS

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts, batching at 16 per API call.
        Returns a list of 3072-dim float vectors in the same order as input.
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), _MAX_BATCH):
            batch = texts[i : i + _MAX_BATCH]
            response = await self._client.embeddings.create(
                model=self._deployment,
                input=batch,
                dimensions=self._dimensions,
            )
            # Sort by index to guarantee order matches input order
            ordered = sorted(response.data, key=lambda item: item.index)
            all_embeddings.extend(item.embedding for item in ordered)

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Single-text convenience wrapper. Returns one 3072-dim vector."""
        results = await self.embed_texts([query])
        return results[0]
