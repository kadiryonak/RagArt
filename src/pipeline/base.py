"""Pipeline temel tipleri: Request (immutable), State (mutable), Stage, Pipeline.

Tasarım kararları:
    1. Request = frozen dataclass — bir sorgunun başında tüm parametreler
       fix'lenir; stage'ler request'i değiştiremez.
    2. State = mutable dataclass — stage'ler progressively buraya yazar
       (docs, context, answer, ...). request'i taşır ki stage'ler
       parametre erişebilsin.
    3. Stage = ABC + __call__ — short-circuit semantiği base class'ta;
       alt sınıf sadece run() yazar.
    4. Pipeline = stage listesi + run() — döngü + erken çıkış.

PROFESYONEL NOT
    Bu pattern, build sistemlerinde (bazel/cargo plan), DAG executor'larda
    (airflow, prefect), LangChain'in Runnable'ında aynı şekilde yaşıyor.
    Tek farkımız: sıralı, paralel değil — sorgu pipeline'ında dependency
    DAG'ı zaten lineer.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    # Avoid circular imports — these are only used for type hints.
    from src.llm_providers import BaseLLMProvider
    from src.memory.base import ConversationTurn
    from src.prompt_strategies.base import BasePromptStrategy, StrategyContext
    from src.rag_system import TurkishRAGSystem
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document


# ─── Request (immutable input) ─────────────────────────────────────────


@dataclass(frozen=True)
class QueryRequest:
    """Tek bir ask() çağrısının değişmez girdileri.

    frozen=True: hiçbir stage bunu mutate edemez; gerçek "request"
    kavramına uygun. State değişiklikleri ayrı bir dataclass'ta.
    """

    question: str
    # Retrieval / context
    k: int = 5
    retrieval_strategy: Optional[str] = None
    rerank: bool = False
    rerank_fetch_k: int = 20
    deduplicate_context: bool = False
    reorder_context: bool = False
    max_context_tokens: Optional[int] = None
    # LLM
    llm_provider: Any = None  # Optional[BaseLLMProvider] — Any to dodge import cycle
    llm_params: Dict[str, Any] = field(default_factory=dict)
    # Strategies
    prompt_strategy: Optional[str] = None
    custom_role: Optional[str] = None
    custom_prompt_template: Optional[str] = None
    memory_strategy: Optional[str] = None
    history: tuple = ()  # tuple of ConversationTurn (frozen-friendly)
    # Caching
    use_response_cache: bool = True
    use_semantic_cache: bool = False
    semantic_cache_threshold: float = 0.92
    # Safety
    allow_general_knowledge_fallback: bool = False


# ─── State (mutable, built by stages) ──────────────────────────────────


@dataclass
class QueryState:
    """Pipeline çalışırken stage'lerin dolduracağı mutable state.

    Bir stage bilgi ekler (e.g. RetrievalStage docs'u doldurur),
    sonraki stage o bilgiyi okur (ContextBuildStage docs'tan context
    string'i üretir).

    state.response set edilirse pipeline kısa-devre yapar — başarılı
    cache hit, greeting fast-path, low-relevance fallback gibi.
    """

    request: QueryRequest
    rag: Any  # TurkishRAGSystem — Any to dodge import cycle

    # Progressively filled:
    complexity: Optional[Any] = None              # QueryComplexity enum
    adaptive_cfg: Optional[Any] = None            # AdaptiveConfig dataclass
    cache_payload: Dict[str, Any] = field(default_factory=dict)
    docs: List[Any] = field(default_factory=list)  # langchain Documents
    relevance_score: float = 0.0
    context: str = ""
    memory_context: str = ""
    strategy: Any = None                          # BasePromptStrategy instance
    strategy_ctx: Any = None                      # StrategyContext
    answer: Optional[str] = None
    # Telemetry pieces stages drop in; ResponseBuildStage assembles these
    # into the final result["..."] fields.
    extra_meta: Dict[str, Any] = field(default_factory=dict)
    # Per-stage timing for observability (stage_name → seconds)
    timings: Dict[str, float] = field(default_factory=dict)

    # Short-circuit: any stage setting this skips downstream stages.
    response: Optional[Dict[str, Any]] = None


# ─── Stage ABC ─────────────────────────────────────────────────────────


class PipelineStage(ABC):
    """Tüm stage'lerin temel sınıfı.

    Alt sınıf sadece run()'ı implement eder. __call__ short-circuit'i
    ve timing'i handle eder.
    """

    name: str = "stage"

    def __call__(self, state: QueryState) -> QueryState:
        # Önceki bir stage response ürettiyse atla.
        if state.response is not None:
            return state
        t0 = time.perf_counter()
        try:
            return self.run(state)
        finally:
            state.timings[self.name] = round(time.perf_counter() - t0, 4)

    @abstractmethod
    def run(self, state: QueryState) -> QueryState:
        """Stage mantığı. State'i mutate edip aynı state'i döndür.

        Erken çıkış için state.response = {...} set et; sonraki stage'ler
        çağrılmaz, ama timing yine kaydedilir.
        """


# ─── Pipeline runner ───────────────────────────────────────────────────


class Pipeline:
    """Sıralı stage runner."""

    def __init__(self, stages: List[PipelineStage]):
        if not stages:
            raise ValueError("Pipeline must have at least one stage")
        self.stages = stages

    def run(self, request: QueryRequest, rag: Any) -> Dict[str, Any]:
        """Pipeline'ı çalıştır, final response dict'i döndür.

        Stage'lerden hiçbiri response set etmezse bu bir bug — RuntimeError.
        Sonuç dict'ine pipeline timing'leri 'timings' key'i altında eklenir
        (observability için faydalı, UI bunu gösterip gizleyebilir).

        ÖNEMLİ: response set edildikten sonra DA tüm stage'lere uğrarız.
        Stage.__call__ varsayılan olarak short-circuit yapar (kendi run'unu
        atlar) — ama groundedness/cache_write gibi "post-response" stage'ler
        __call__'u override ederek her durumda çalışır. Pipeline.run()
        burada erken çıkış yapmaz; bu sayede bu post-response stage'ler
        timings'e + final response'a kayıt yapabilir.
        """
        state = QueryState(request=request, rag=rag)
        for stage in self.stages:
            state = stage(state)
        if state.response is None:
            # En az bir stage response üretmek zorundaydı.
            raise RuntimeError(
                f"Pipeline finished without producing a response "
                f"(stages run: {list(state.timings.keys())})"
            )
        # Inject timings into the final dict (don't overwrite if a stage
        # already added one — they shouldn't, but be defensive).
        state.response.setdefault("timings", dict(state.timings))
        return state.response
