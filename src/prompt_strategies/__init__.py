"""Prompt strategies — RAG'da prompt mühendisliğini plugin'leyen sistem.

Stratejiler ne yapar?
    Bir soru + retrieved context aldığında nasıl bir LLM prompt'u
    inşa edileceğini ve bazı stratejilerde retrieval'ın kendisinin
    nasıl çalışacağını belirler (multi-query gibi).

Hiyerarşik tasarım:
    BasePromptStrategy   — sözleşme
    ├─ Single-call (1 LLM çağrısı)
    │   ├─ DirectStrategy        (default, geliştirilmiş prompt)
    │   ├─ ChainOfThoughtStrategy (adım adım düşün)
    │   ├─ FewShotStrategy       (2-3 örnek + soru)
    │   ├─ RoleBasedStrategy     (kullanıcı tanımlı uzman rolü)
    │   └─ CustomStrategy        (kullanıcı tanımlı template)
    └─ Multi-call (birden fazla LLM/retrieval çağrısı)
        ├─ MultiQueryStrategy    (N varyant üret → her biriyle ara → RRF)
        ├─ HyDEStrategy          (varsayımsal doküman üret → embed → search)
        └─ StepBackStrategy      (genel soru üret → geniş context çek)

Yeni strateji eklemek için: BasePromptStrategy implement et +
PromptStrategyFactory.register() çağır. UI dropdown'u otomatik tanır.
"""

from src.prompt_strategies.base import (
    BasePromptStrategy,
    PromptStrategyFactory,
    StrategyContext,
)
from src.prompt_strategies.direct import DirectStrategy
from src.prompt_strategies.cot import ChainOfThoughtStrategy
from src.prompt_strategies.few_shot import FewShotStrategy
from src.prompt_strategies.role_based import RoleBasedStrategy
from src.prompt_strategies.custom import CustomStrategy
from src.prompt_strategies.multi_query import MultiQueryStrategy
from src.prompt_strategies.query_rewrite import QueryRewriteStrategy
from src.prompt_strategies.self_refine import SelfRefineStrategy
from src.prompt_strategies.hyde import HyDEStrategy
from src.prompt_strategies.step_back import StepBackStrategy

__all__ = [
    "BasePromptStrategy",
    "PromptStrategyFactory",
    "StrategyContext",
    "DirectStrategy",
    "ChainOfThoughtStrategy",
    "FewShotStrategy",
    "RoleBasedStrategy",
    "CustomStrategy",
    "MultiQueryStrategy",
    "QueryRewriteStrategy",
    "SelfRefineStrategy",
    "HyDEStrategy",
    "StepBackStrategy",
]

