"""BaseVectorStore — vector DB providers için ortak sözleşme.

Her vector store şunları sunar:
    - upsert(documents, embeddings)  — chunk + embedding ekle/güncelle
    - similarity_search(query_embedding, k) — en yakın k chunk
    - count()                         — toplam chunk
    - delete()                        — tüm koleksiyonu temizle

LangChain'in Chroma sınıfı zaten bu interface'in büyük kısmını sağlıyor;
adapter olarak sarmalıyoruz. Yeni bir DB eklemek için BaseVectorStore'u
implement et + VectorStoreFactory'ye kaydet → UI/CLI otomatik tanır.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VectorSearchResult:
    """Bir similarity search sonucu — chunk + score + metadata."""
    page_content: str
    metadata: Dict[str, Any]
    score: float = 0.0


class BaseVectorStore(ABC):
    """Tüm vector DB adapter'larının temel sınıfı."""

    name: str = "base"

    @abstractmethod
    def upsert_documents(self, documents: List[Any]) -> None:
        """LangChain Document listesini upsert et (embedding hesaplanır)."""

    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[VectorSearchResult]:
        """Sorgu ile en yakın k chunk'ı azalan benzerlikte döndür."""

    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Any]:
        """LangChain uyumlu: (Document, distance) tuple listesi.

        Mevcut DenseRetriever bu metodu çağırıyor — yeni adapter da uymalı.
        """

    @abstractmethod
    def count(self) -> int:
        """Toplam chunk sayısı."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Tüm koleksiyonu temizle (reindex öncesi)."""

    def is_empty(self) -> bool:
        try:
            return self.count() == 0
        except Exception:
            return True


class VectorStoreFactory:
    """Vector store registry — UI dropdown'u bundan üretir."""

    _registry: Dict[str, type] = {}
    _info: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, key: str, store_class: type, *, label: str, desc: str) -> None:
        cls._registry[key] = store_class
        cls._info[key] = {"id": key, "label": label, "desc": desc}

    @classmethod
    def create(cls, key: str, **kwargs) -> BaseVectorStore:
        if key not in cls._registry:
            raise ValueError(
                f"Unknown vector store: {key}. "
                f"Available: {sorted(cls._registry.keys())}"
            )
        return cls._registry[key](**kwargs)

    @classmethod
    def available(cls) -> List[Dict[str, Any]]:
        """JSON-serializable: UI'a yollanır."""
        return list(cls._info.values())

    @classmethod
    def is_available(cls, key: str) -> bool:
        return key in cls._registry
