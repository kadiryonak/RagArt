"""Retriever base contract.

Retriever, bir sorgu için (page_content, metadata, score) üçlüsünden oluşan
sıralı bir RetrievedDoc listesi döndürür. Sıralama relevance'a göre azalan.

Tasarım kararları:
    - score: retriever'a özel ham skor. Dense → cosine [-1,1]; sparse → BM25
      skor [0, +∞); hybrid → RRF skor [0, ~0.03 N için]. Karşılaştırma
      yapma; sadece sıralama için kullan.
    - id: RetrievedDoc'un benzersiz anahtarı. RRF füzyonu için kritik —
      farklı retriever'ların aynı belgeyi farklı sırayla döndürdüğünü
      eşleştirmek için kullanılır. Default: (source, chunk_index).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RetrievedDoc:
    """Bir retrieval sonucu — page_content + metadata + score."""

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)
    score: float = 0.0
    doc_id: Optional[str] = None  # füzyon için benzersiz anahtar

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    def get_id(self) -> str:
        """Stable id. doc_id ya da (source, item_index/chunk hash) fallback'i."""
        if self.doc_id:
            return self.doc_id
        idx = self.metadata.get("item_index")
        if idx is not None:
            return f"{self.source}::{idx}"
        # Son çare: content hash (deterministik ama yavaş)
        return f"{self.source}::{hash(self.page_content) & 0xFFFFFFFF:08x}"


class BaseRetriever(ABC):
    """Tüm retriever'ların türetildiği temel sınıf."""

    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[RetrievedDoc]:
        """Sorguya en alakalı k belgeyi sıralı şekilde döndür."""

    def supports_filters(self) -> bool:
        """Bu retriever metadata filter destekliyor mu? (Default: hayır.)"""
        return False
