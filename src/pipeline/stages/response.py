"""ResponseStage — birikmiş state'ten final response dict'ini kur.

Bu, "happy path" sonu — guard/cache/fallback short-circuit'lerinin
hiçbiri tetiklenmediyse buraya geliyoruz.

Response shape, eski rag_system.ask() return'üyle TAM AYNI (backward
compat):
    question, answer, source_documents, context_used, source,
    relevance_score, retrieval_strategy, memory_strategy, memory_used,
    prompt_strategy, cache_hit, (opsiyonel) query_complexity
"""

from __future__ import annotations

from src.pipeline.base import PipelineStage, QueryState


class ResponseStage(PipelineStage):
    name = "response"

    def run(self, state: QueryState) -> QueryState:
        r = state.request
        # Source documents — content kısaltılır (300 char)
        source_docs = []
        for doc in state.docs:
            content = doc.page_content
            if len(content) > 300:
                content = content[:300] + "..."
            source_docs.append({
                "content": content,
                "source": doc.metadata.get("source", "Unknown"),
                "metadata": doc.metadata,
            })

        # Context preview (500 char)
        ctx_preview = state.context
        if len(ctx_preview) > 500:
            ctx_preview = ctx_preview[:500] + "..."

        state.response = {
            "question": r.question,
            "answer": state.answer,
            "source_documents": source_docs,
            "context_used": ctx_preview,
            "source": "rag_system",
            "relevance_score": state.relevance_score,
            "retrieval_strategy": state.extra_meta.get(
                "retrieval_label", "unknown",
            ),
            "memory_strategy": r.memory_strategy or "none",
            "memory_used": bool(state.memory_context),
            "prompt_strategy": state.strategy.name if state.strategy else "direct",
            "cache_hit": False,
        }
        # Adaptive: classify stage'i complexity yazdıysa response'a koy
        if "query_complexity" in state.extra_meta:
            state.response["query_complexity"] = state.extra_meta["query_complexity"]
        return state
