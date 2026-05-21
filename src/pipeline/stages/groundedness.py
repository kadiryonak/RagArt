"""GroundednessStage — cevabın context'e ne kadar sadık olduğunu skorla.

Pipeline'da ResponseStage'den SONRA, CacheWriteStage'den ÖNCE çalışır:
    ResponseStage state.response['answer']'ı set etti
    GroundednessStage state.response'a groundedness_score (ve eşik altıysa
        groundedness_warning) ekler
    CacheWriteStage final response'u (groundedness alanları dahil) cache'ler

state.response set olduğu için bu da __call__'u override eder — base
class'ın short-circuit'u burayı atlatırdı.
"""

from __future__ import annotations

import logging

from src.pipeline.base import PipelineStage, QueryState


logger = logging.getLogger(__name__)


class GroundednessStage(PipelineStage):
    name = "groundedness"

    def __call__(self, state: QueryState) -> QueryState:
        # CacheWrite gibi: response set edilmiş olsa bile çalış.
        import time
        t0 = time.perf_counter()
        try:
            return self.run(state)
        finally:
            state.timings[self.name] = round(time.perf_counter() - t0, 4)

    def run(self, state: QueryState) -> QueryState:
        # Sadece happy path için (cache hit / guard block / greeting'te
        # context yok, skor anlamsız).
        if state.response is None:
            return state
        if state.response.get("source") != "rag_system":
            return state
        if not state.context or not state.answer:
            return state

        from src.guard import GroundednessScorer

        score = GroundednessScorer.score(state.answer, state.context)
        state.response["groundedness_score"] = round(score, 3)
        if not GroundednessScorer.is_grounded(score):
            state.response["groundedness_warning"] = True
            logger.warning("Low groundedness: %.3f", score)
        return state
