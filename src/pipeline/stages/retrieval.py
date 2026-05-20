"""RetrievalStage — prompt strategy resolve + retrieval (multi-query dahil).

İki şey yapar:
    1. Prompt strategy çöz + StrategyContext kur (multi-step stratejiler
       buna ihtiyaç duyacak ExecuteStage'de). state.strategy + state.strategy_ctx
    2. Retrieval: strategy.is_multi_query ise N varyant üret + fan-out + RRF;
       değilse tek retrieve.

Çıktı: state.docs (langchain Document list).
"""

from __future__ import annotations

import logging
from typing import List

from src.pipeline.base import PipelineStage, QueryState


logger = logging.getLogger(__name__)


class RetrievalStage(PipelineStage):
    name = "retrieval"

    def run(self, state: QueryState) -> QueryState:
        rag = state.rag
        r = state.request

        # Effective değerleri ClassifyStage adaptive override sonrası okuyoruz
        effective_k = state.extra_meta.get("effective_k", r.k)
        effective_strategy = state.extra_meta.get(
            "effective_strategy", r.retrieval_strategy,
        )
        effective_rerank = state.extra_meta.get("effective_rerank", r.rerank)

        # Context engineering processor chain
        context_chain = rag._build_context_chain(
            deduplicate=r.deduplicate_context,
            reorder=r.reorder_context,
            max_context_tokens=r.max_context_tokens,
        )

        # Prompt strategy + execution context
        strategy = rag._resolve_prompt_strategy(
            r.prompt_strategy,
            custom_role=r.custom_role,
            custom_prompt_template=r.custom_prompt_template,
        )
        state.strategy = strategy

        provider = r.llm_provider or rag.llm_provider

        # Local retrieve closure — multi-query'de de aynı parametrelerle çağrılır
        def _retrieve(q: str, kk: int):
            return rag.search(
                q, k=kk, strategy=effective_strategy,
                rerank=effective_rerank, rerank_fetch_k=r.rerank_fetch_k,
                context_chain=context_chain,
            )

        from src.prompt_strategies.base import StrategyContext
        state.strategy_ctx = StrategyContext(
            llm=provider,
            retrieve_fn=_retrieve,
            embed_fn=rag.embedding_manager.embed_query,
            llm_params=dict(r.llm_params),
        )

        # Retrieval — multi-query veya single
        if strategy.is_multi_query:
            variants = strategy.generate_query_variations(
                r.question, state.strategy_ctx,
            )
            logger.info("Multi-query expanded to %d variants", len(variants))
            docs = rag._fuse_retrievals(variants, effective_k, _retrieve)
        else:
            docs = _retrieve(r.question, effective_k)

        state.docs = docs

        # Strategy label for response telemetry
        label_parts = [(effective_strategy or "auto")]
        if effective_rerank:
            label_parts.append("+rerank")
        ctx_label_bits = []
        if r.deduplicate_context: ctx_label_bits.append("dedup")
        if r.max_context_tokens is not None:
            ctx_label_bits.append(f"budget={r.max_context_tokens}")
        if r.reorder_context: ctx_label_bits.append("reorder")
        if ctx_label_bits:
            label_parts.append("+ctx[" + ",".join(ctx_label_bits) + "]")
        label_parts.append(f"+prompt[{strategy.name}]")
        state.extra_meta["retrieval_label"] = "".join(label_parts)
        return state
