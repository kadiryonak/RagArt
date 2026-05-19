"""ClassifyStage — adaptive routing.

QueryClassifier ile sorgunun karmaşıklığı belirlenir. İki çıktı:
    1. complexity + adaptive_cfg state'e yazılır (sonraki stage'ler okur)
    2. Eğer "greeting" ise retrieval atlanır → fast-path response

Caller'ın açıkça verdiği k/retrieval_strategy/rerank değerleri default
ise adaptive_cfg ile override edilir. Caller explicit set ettiyse
caller değeri kazanır.

NOT: Adaptive override mantığı stage seviyesinde tutulur — RetrievalStage
sadece "state.docs'u doldur" işine bakar.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from src.pipeline.base import PipelineStage, QueryState


logger = logging.getLogger(__name__)


class ClassifyStage(PipelineStage):
    name = "classify"

    def run(self, state: QueryState) -> QueryState:
        from src.query_classifier import (
            QueryClassifier,
            greeting_response,
        )

        question = state.request.question
        complexity = QueryClassifier.classify(question)
        cfg = QueryClassifier.get_config(complexity)
        state.complexity = complexity
        state.adaptive_cfg = cfg
        logger.info("Query complexity: %s", complexity.value)

        # Greeting fast-path: pipeline'ı bitir.
        if cfg.skip_retrieval:
            logger.info("Greeting detected — skipping retrieval")
            state.response = {
                "question": question,
                "answer": greeting_response(question),
                "source_documents": [],
                "context_used": "",
                "source": "greeting",
                "relevance_score": 1.0,
                "retrieval_strategy": "none",
                "memory_strategy": "none",
                "memory_used": False,
                "prompt_strategy": "adaptive",
                "cache_hit": False,
                "query_complexity": complexity.value,
            }
            return state

        # Adaptive override — caller default değer verdiyse cfg ile değiştir.
        # (Frozen request'i değiştirmeye çalışmıyoruz; effective_* state'te
        # tutulur ve sonraki stage'ler bunları kullanır.)
        effective_k = state.request.k
        if state.request.k == 4 or state.request.k == 5:
            # Common Python default values for k — assume "not explicitly set".
            # Caller'ın 4/5 dışında özel bir sayı verdiyse cfg.k'yi atla.
            effective_k = cfg.k

        effective_strategy = state.request.retrieval_strategy or cfg.retrieval_strategy
        effective_rerank = state.request.rerank or cfg.rerank

        state.extra_meta["effective_k"] = effective_k
        state.extra_meta["effective_strategy"] = effective_strategy
        state.extra_meta["effective_rerank"] = effective_rerank
        state.extra_meta["query_complexity"] = complexity.value
        return state
