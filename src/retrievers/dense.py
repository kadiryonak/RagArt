"""Dense (embedding-based) retriever.

ChromaDB üzerinden similarity_search_with_score sarmalayıcı. RetrievedDoc
sözleşmesine adapte eder.

ChromaDB cosine distance döndürür (0 = aynı, 2 = zıt). Bizim "score":
    score = 1 - distance / 2     ∈ [0, 1]
sıralama amaçlı; karşılaştırma için absolute anlamı yok.
"""

from __future__ import annotations

from typing import Any, List

from src.retrievers.base import BaseRetriever, RetrievedDoc


class DenseRetriever(BaseRetriever):
    """Embedding tabanlı retriever — ChromaDB vector_store sarmalayıcı."""

    name = "dense"

    def __init__(self, vector_store: Any):
        """vector_store: langchain Chroma instance with .similarity_search_with_score()."""
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedDoc]:
        if self.vector_store is None:
            return []
        results = self.vector_store.similarity_search_with_score(query, k=k)
        out: List[RetrievedDoc] = []
        for doc, distance in results:
            # ChromaDB cosine distance ∈ [0, 2]; "yakın" daha düşük
            score = max(0.0, 1.0 - float(distance) / 2.0)
            out.append(RetrievedDoc(
                page_content=doc.page_content,
                metadata=dict(doc.metadata),
                score=score,
            ))
        return out
