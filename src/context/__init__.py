"""Context engineering — retrieval'dan sonra, prompt inşasından önce.

Tipik pipeline:
    retrieve(top-N) → rerank → CONTEXT PROCESSORS → format prompt → LLM

Buradaki processor'lar üç temel problemi çözer:
    1. Redundancy   — aynı bilgiyi iki farklı chunk'tan tekrarlamak (token israfı)
    2. Ordering     — "lost in the middle" — LLM uzun context'in ortasını unutur
    3. Token budget — provider context limit'i + maliyet kontrolü

Plugin contract: BaseContextProcessor.process(query, docs) → docs.
"""

from src.context.base import BaseContextProcessor, ProcessorChain
from src.context.redundancy import RedundancyFilter
from src.context.reorder import LostInTheMiddleReorderer
from src.context.token_budget import TokenBudgetTrimmer

__all__ = [
    "BaseContextProcessor",
    "ProcessorChain",
    "RedundancyFilter",
    "LostInTheMiddleReorderer",
    "TokenBudgetTrimmer",
]
