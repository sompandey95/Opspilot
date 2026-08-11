"""RAG retrieval exposed as a tool the agent can call."""
from __future__ import annotations

from app.tools.base import RiskLevel, Tool, ToolResult


class SearchKnowledgeTool(Tool):
    name = "search_knowledge"
    description = (
        "Search the ShopEasy knowledge base (FAQs, policies, past tickets, API "
        "docs, policy changelogs). Use this before answering any policy or "
        "how-to question. Changelog entries are dated and supersede older "
        "policy text on the same topic."
    )
    risk_level = RiskLevel.NONE
    is_state_changing = False
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 3, "description": "Natural-language search query"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            "doc_type": {
                "type": "string",
                "enum": ["faq", "policy", "ticket", "api_doc", "changelog"],
                "description": (
                    "Optional filter to a single document type. Dated changelog "
                    "entries are always included regardless of this filter, "
                    "because they supersede older policy/FAQ text."
                ),
            },
        },
        "required": ["query"],
    }

    def __init__(self, retriever) -> None:
        # HybridRetriever; typed loosely so the tool imports without RAG deps
        self._retriever = retriever

    async def execute(
        self,
        query: str,
        top_k: int | None = None,
        doc_type: str | None = None,
    ) -> ToolResult:
        if self._retriever is None:
            return ToolResult(success=False, error="Knowledge retriever is not initialised")

        try:
            results = await self._retriever.retrieve(query, top_k=top_k)
        except Exception as exc:
            return ToolResult(success=False, error=f"Retrieval failed: {exc}")

        if doc_type:
            # Changelogs are dated corrections that supersede every other doc
            # type — the stale-knowledge invariant is enforced here, not left
            # to the LLM's filter choice.
            results = [
                r for r in results
                if r.metadata.get("doc_type") in (doc_type, "changelog")
            ]

        return ToolResult(
            success=True,
            data=[
                {
                    "chunk_id": r.chunk_id,
                    "content": r.content,
                    "score": r.score,
                    "doc_type": r.metadata.get("doc_type"),
                    "source_file": r.metadata.get("source_file"),
                }
                for r in results
            ],
        )
