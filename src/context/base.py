"""BaseContextProcessor sözleşmesi + ProcessorChain (zincirleme).

Her processor `process(query, docs) → docs` imzasıyla deterministik bir
dönüşüm uygular. Liste sırası "relevance order"dır (ilk = en alakalı).
Processor'lar bu sıralamayı genelde korur (reorderer hariç).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from src.retrievers.base import RetrievedDoc


class BaseContextProcessor(ABC):
    name: str = "base"

    @abstractmethod
    def process(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        """docs'u dönüştür. Boş giriş → boş çıkış."""


class ProcessorChain(BaseContextProcessor):
    """Birden çok processor'ı sırayla uygula.

    Sıra önemlidir:
        - Redundancy önce  → duplicate'ler aşağı akışta token harcamasın
        - Token budget sonra → en kötü ihtimalde "yeterli relevant chunk" kal
        - Reorderer en son  → final placement (lost-in-the-middle)
    """

    name = "chain"

    def __init__(self, processors: Sequence[BaseContextProcessor]):
        self.processors = list(processors)

    def process(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        for p in self.processors:
            docs = p.process(query, docs)
            if not docs:
                break
        return docs
