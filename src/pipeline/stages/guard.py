"""GuardStage — prompt injection / yasaklı içerik kontrolü.

InputGuard sınıfı tarafından sorgu skorlanır; threshold'u geçerse
state.response'a red mesajı set edilir, pipeline kısa-devre yapar.

Mevcut davranış birebir korunuyor — InputGuard.check sınıf metotu,
rejection_message classmethod.
"""

from __future__ import annotations

import logging

from src.pipeline.base import PipelineStage, QueryState


logger = logging.getLogger(__name__)


class GuardStage(PipelineStage):
    name = "guard"

    def run(self, state: QueryState) -> QueryState:
        from src.guard import InputGuard  # lazy: keep cycle-free

        result = InputGuard.check(state.request.question)
        if result.is_safe:
            # Geçti; downstream'e ilet
            return state

        # Block: response'u set ederek pipeline'ı kısa-devre yap
        logger.warning(
            "Prompt injection detected (score=%.2f, reason=%s)",
            result.score, result.reason,
        )
        state.response = {
            "question": state.request.question,
            "answer": InputGuard.rejection_message(),
            "source_documents": [],
            "context_used": "",
            "source": "guard_blocked",
            "relevance_score": 0.0,
            "guard_score": result.score,
            "guard_reason": result.reason,
        }
        return state
