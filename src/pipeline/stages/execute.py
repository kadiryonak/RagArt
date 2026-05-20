"""ExecuteStage — LLM çağrısı (veya multi-call strategy orchestration).

Strategy.execute(ctx, ...) tüm prompt inşa + LLM call'u yapar. CoT için
"YANIT:" extract, multi-query'de ek query rewrite zaten strategy içinde.

state.answer doldurulur; ResponseStage final dict'i kurar.
"""

from __future__ import annotations

import logging

from src.pipeline.base import PipelineStage, QueryState


logger = logging.getLogger(__name__)


class ExecuteStage(PipelineStage):
    name = "execute"

    def run(self, state: QueryState) -> QueryState:
        provider = state.request.llm_provider or state.rag.llm_provider
        provider_label = getattr(provider, "model", state.rag.model_type)
        logger.info(
            "Generating (%s) via strategy=%s",
            provider_label, state.strategy.name,
        )
        state.answer = state.strategy.execute(
            state.strategy_ctx,
            question=state.request.question,
            context=state.context,
            memory_context=state.memory_context,
        )
        return state
