"""Ask-pipeline — RAG sorgu işleme pipeline'ı.

ÖNCE
    rag_system.ask() ~200 satır + 22 parametre, içinde 7 fazı sıkıştırıyordu.
    Test edilemiyordu (unit), debug edilemiyordu, eklenen her yeni özellik
    bir önceki şeyi kırma riski taşıyordu.

SONRA
    Pipeline + bağımsız stage'ler:
        QueryRequest → [stage1, stage2, ...] → response dict

    Her stage:
        - tek sorumluluğu var (guard / cache / retrieve / ...)
        - <50 satır
        - state'i okur, mutate eder
        - state.response set ederek pipeline'ı kısa-devre yapabilir

    rag_system.ask() bu pipeline'ı çağıran ince bir wrapper olur.

KISA-DEVRE SEMANTİĞİ
    Bir stage state.response'a dict atarsa, sonraki stage'ler çalışmaz.
    Kullanılan yerler:
        - GuardStage: prompt injection → guard_blocked response
        - ClassifyStage: greeting → fast-path response
        - CacheLookupStage: hit → cached response
        - RelevanceGateStage: low score → fallback response
"""

from src.pipeline.base import (
    PipelineStage,
    Pipeline,
    QueryRequest,
    QueryState,
)

__all__ = [
    "PipelineStage",
    "Pipeline",
    "QueryRequest",
    "QueryState",
]
