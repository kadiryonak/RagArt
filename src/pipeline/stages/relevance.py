"""RelevanceGateStage — düşük relevance'ta fallback yoluna sap.

Retrieval boşsa veya relevance_score eşik altındaysa, _fallback_response
çağrılır ve state.response set edilir → pipeline kısa-devre. Fallback
mesajı (allow_general_knowledge_fallback flag'ine göre) kullanıcıya net
"bilgi yok" mesajı verir veya LLM general-knowledge cevabı üretir.

state.relevance_score her durumda yazılır (response'a metric olarak gider).
"""

from __future__ import annotations

import logging

from src.pipeline.base import PipelineStage, QueryState


logger = logging.getLogger(__name__)


class RelevanceGateStage(PipelineStage):
    name = "relevance_gate"

    def run(self, state: QueryState) -> QueryState:
        rag = state.rag
        score = rag.calculate_relevance_score(
            state.request.question, state.docs,
        )
        state.relevance_score = score
        logger.info("Relevance score: %.3f", score)

        # Fallback'i çağırma kriteri: ya hiç doc yok ya da skor eşiğin altında
        if not state.docs or score < rag.RELEVANCE_THRESHOLD:
            logger.info("Insufficient context, using fallback")
            state.response = rag._fallback_response(
                state.request.question, score,
                llm_provider=state.request.llm_provider,
                llm_params=dict(state.request.llm_params),
                allow_general_knowledge=state.request.allow_general_knowledge_fallback,
            )
        return state
