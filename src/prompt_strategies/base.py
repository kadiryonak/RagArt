"""Base contract for prompt strategies.

Stratejilerin iki katmanı var:
    1. Prompt inşası — build_prompt(question, context, memory_context)
       Bütün stratejiler bunu implement eder. Sonuç tek bir LLM prompt'u.
    2. Orchestration — generate_query_variations(question, llm) + execute()
       Multi-step stratejiler (multi-query, HyDE) override eder. Default
       implementation: tek sorgu, tek LLM çağrısı.

rag_system bu sözleşme üzerinden hangi stratejiyi seçtiğine bakmadan
çalışır — yeni strateji eklemek sadece factory'ye register etmek demek.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class StrategyContext:
    """Multi-step stratejilerin orchestration'ı için gereken handles.

    rag_system bunu her ask() çağrısında doldurup strategy.execute()'a
    geçirir.
    """
    llm: Any  # BaseLLMProvider
    retrieve_fn: Callable[[str, int], List[Any]]  # (query, k) -> [Document]
    embed_fn: Optional[Callable[[str], List[float]]] = None
    llm_params: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class BasePromptStrategy(ABC):
    """Tüm stratejilerin temel sınıfı."""

    name: str = "base"
    label: str = "Base"
    description_tr: str = ""
    is_multi_call: bool = False  # birden çok LLM çağrısı gerektiriyor mu?
    is_multi_query: bool = False  # birden çok retrieval çağrısı gerektiriyor mu?

    @abstractmethod
    def build_prompt(
        self,
        *,
        question: str,
        context: str,
        memory_context: str = "",
        **kwargs: Any,
    ) -> str:
        """Final LLM prompt'unu inşa et."""

    def generate_query_variations(
        self,
        question: str,
        ctx: StrategyContext,
    ) -> List[str]:
        """Retrieval için sorgu varyantları üret.

        Default: [original_question]. Multi-query strategy override eder.
        """
        return [question]

    def execute(
        self,
        ctx: StrategyContext,
        *,
        question: str,
        context: str,
        memory_context: str = "",
        **kwargs: Any,
    ) -> str:
        """Stratejiyi çalıştır, LLM'in nihai cevabını döndür.

        Default: build_prompt + tek LLM çağrısı. Multi-step stratejiler
        override edip kendi orchestration'larını yapabilir.
        """
        prompt = self.build_prompt(
            question=question, context=context,
            memory_context=memory_context, **kwargs,
        )
        return ctx.llm.generate(prompt, **ctx.llm_params)


class PromptStrategyFactory:
    """Strategy registry — UI dropdown'u bundan üretilir."""

    _registry: Dict[str, type] = {}
    _info: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, key: str, strategy_class: type) -> None:
        inst = strategy_class()
        cls._registry[key] = strategy_class
        cls._info[key] = {
            "id": key,
            "label": inst.label,
            "description": inst.description_tr,
            "is_multi_call": inst.is_multi_call,
            "is_multi_query": inst.is_multi_query,
        }

    @classmethod
    def create(cls, key: str, **kwargs: Any) -> BasePromptStrategy:
        if key not in cls._registry:
            raise ValueError(
                f"Unknown prompt strategy '{key}'. "
                f"Available: {sorted(cls._registry.keys())}"
            )
        return cls._registry[key](**kwargs)

    @classmethod
    def available(cls) -> List[Dict[str, Any]]:
        return list(cls._info.values())

    @classmethod
    def is_available(cls, key: str) -> bool:
        return key in cls._registry
