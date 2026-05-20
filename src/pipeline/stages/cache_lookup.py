"""CacheLookupStage — exact + semantic cache lookup.

İki cache'i sırayla kontrol et:
    1. ResponseCache (exact match) — hash(payload) → cevap
    2. SemanticCache — sorgunun embedding'i ile en yakın cache entry

Cache hit'inde state.response cached dict'le doldurulur ve pipeline
kısa-devre olur.

cache_payload state'te de saklanır — CacheWriteStage'in sonradan
write yapması için.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.pipeline.base import PipelineStage, QueryState


logger = logging.getLogger(__name__)


def _build_cache_payload(state: QueryState) -> Dict[str, Any]:
    """Bir cevabı değiştirebilecek her input'u içeren stable hash girdisi.

    Önemli: hash deterministik olmalı. tuple'ları stringify ediyoruz
    çünkü ResponseCache.make_key json.dumps(sort_keys=True) yapıyor.
    """
    r = state.request
    provider_id = (
        getattr(r.llm_provider, "name", None)
        or getattr(r.llm_provider, "model", None)
        or state.rag.model_type
    )
    return {
        "question": r.question,
        "k": state.extra_meta.get("effective_k", r.k),
        "retrieval_strategy": state.extra_meta.get(
            "effective_strategy", r.retrieval_strategy,
        ),
        "rerank": state.extra_meta.get("effective_rerank", r.rerank),
        "rerank_fetch_k": r.rerank_fetch_k,
        "prompt_strategy": r.prompt_strategy,
        "custom_role": r.custom_role,
        "custom_prompt_template": r.custom_prompt_template,
        "memory_strategy": r.memory_strategy,
        "deduplicate_context": r.deduplicate_context,
        "reorder_context": r.reorder_context,
        "max_context_tokens": r.max_context_tokens,
        "llm_params": dict(r.llm_params),
        "provider": provider_id,
        "history_hash": hash(r.history),
    }


class CacheLookupStage(PipelineStage):
    name = "cache_lookup"

    def run(self, state: QueryState) -> QueryState:
        # 1) Payload'u inşa et — CacheWriteStage de aynı payload ile yazsın
        state.cache_payload = _build_cache_payload(state)

        # 2) Exact response cache
        if state.request.use_response_cache:
            cached = state.rag.response_cache.get(state.cache_payload)
            if cached is not None:
                logger.info("Response cache HIT (exact)")
                cached["cache_hit"] = "exact"
                state.response = cached
                return state

        # 3) Semantic cache
        if state.request.use_semantic_cache:
            # Per-request threshold override
            from src.cache import SemanticCache as _SC
            sc = _SC(
                state.rag.semantic_cache._store.db_path,
                embed_fn=state.rag.embedding_cache.embed_query,
                similarity_threshold=state.request.semantic_cache_threshold,
            )
            cached, sim = sc.get(state.request.question)
            if cached is not None:
                logger.info(
                    "Semantic cache HIT (sim=%.3f ≥ %.2f)",
                    sim, state.request.semantic_cache_threshold,
                )
                cached["cache_hit"] = f"semantic({sim:.3f})"
                state.response = cached
                return state

        return state
