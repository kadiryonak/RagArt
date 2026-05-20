"""CacheWriteStage — başarılı cevabı cache'le.

Sadece "gerçek" cevaplar cache'lenir:
    - state.response var (önceki stage tarafından üretildi)
    - cache_payload doldurulmuş (CacheLookupStage'den geliyor)
    - answer alanı boş değil
    - cache_hit=False (kendisi cache'den geldiyse tekrar yazma)

NOT: state.response set edildiği için __call__ short-circuit hits this
stage early — bu yüzden __call__'u override edip her durumda çalışmasını
sağlıyoruz. Override güvenli çünkü stage idempotent; iki kez yazılmaz.
"""

from __future__ import annotations

import logging

from src.pipeline.base import PipelineStage, QueryState


logger = logging.getLogger(__name__)


class CacheWriteStage(PipelineStage):
    name = "cache_write"

    def __call__(self, state: QueryState) -> QueryState:
        # Bu stage'in özelliği: state.response set olsa bile çalışmalı —
        # çünkü zaten görevi response'u cache'lemek.
        import time
        t0 = time.perf_counter()
        try:
            return self.run(state)
        finally:
            state.timings[self.name] = round(time.perf_counter() - t0, 4)

    def run(self, state: QueryState) -> QueryState:
        # Bir önceki stage response yapmadıysa hiç çağrılmamalı,
        # ama defansif kontrol:
        if state.response is None:
            return state

        # Cache hit'ten gelen response'u tekrar yazma
        if state.response.get("cache_hit"):
            return state

        answer = state.response.get("answer", "")
        if not answer or not answer.strip():
            return state

        payload = state.cache_payload
        if not payload:
            return state

        try:
            if state.request.use_response_cache:
                state.rag.response_cache.set(payload, state.response)
            if state.request.use_semantic_cache:
                state.rag.semantic_cache.set(state.request.question, state.response)
        except Exception as e:
            # Cache yazma hatası → cevabı bozmamalı, sadece logla
            logger.warning("Cache write failed: %s: %s", type(e).__name__, e)
        return state
