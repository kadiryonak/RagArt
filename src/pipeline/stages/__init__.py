"""Pipeline stages — her biri ask() pipeline'ında bir adım.

Stage akışı (Pipeline'a verilirken bu sırada):
    1. GuardStage         prompt injection check; block → guard response
    2. ClassifyStage      adaptive routing; greeting → fast-path response
    3. CacheLookupStage   exact + semantic cache lookup; hit → cached response
    4. RetrievalStage     resolve strategy, retrieve (with multi-query fan-out)
    5. RelevanceGateStage low score → fallback response (insufficient_data)
    6. ContextStage       docs → formatted context string
    7. MemoryStage        history + memory_strategy → memory_context
    8. ExecuteStage       strategy.execute() → answer
    9. ResponseStage      build final result dict
    10. CacheWriteStage   write to response_cache + semantic_cache (only on success)

Stage'ler birbiriyle SADECE QueryState üzerinden konuşur. Doğrudan
methods/refs paylaşılmaz — test edilebilirlik için kritik.
"""

# Stage'ler eklendikçe burada açılır. Henüz yazılmamış olanlar yorum
# satırında. Pipeline'ı son hâline getirirken hepsi açık olacak.
from src.pipeline.stages.guard import GuardStage
from src.pipeline.stages.classify import ClassifyStage
# from src.pipeline.stages.cache_lookup import CacheLookupStage
# from src.pipeline.stages.retrieval import RetrievalStage
# from src.pipeline.stages.relevance import RelevanceGateStage
# from src.pipeline.stages.context import ContextStage
# from src.pipeline.stages.memory import MemoryStage
# from src.pipeline.stages.execute import ExecuteStage
# from src.pipeline.stages.response import ResponseStage
# from src.pipeline.stages.cache_write import CacheWriteStage

__all__ = [
    "GuardStage",
    "ClassifyStage",
]
