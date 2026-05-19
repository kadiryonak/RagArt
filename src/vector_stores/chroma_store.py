"""ChromaVectorStore — mevcut Chroma davranışını BaseVectorStore'a uydurur."""

from __future__ import annotations

from typing import Any, List, Optional

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

import chromadb

from src.vector_stores.base import BaseVectorStore, VectorSearchResult, VectorStoreFactory


class ChromaVectorStore(BaseVectorStore):
    """LangChain Chroma adapter. Mevcut davranışı koruyor — drop-in."""

    name = "chroma"

    def __init__(
        self,
        *,
        collection_name: str,
        persist_path: str,
        embedding_function: Any,
    ):
        self.collection_name = collection_name
        self.persist_path = persist_path
        self._embeddings = embedding_function
        # Each workspace gets its own PersistentClient pointed at its data dir.
        self._client = chromadb.PersistentClient(path=persist_path)
        self._lc_store: Optional[Chroma] = self._open_existing()

    def _open_existing(self) -> Optional[Chroma]:
        try:
            names = [c.name for c in self._client.list_collections()]
            if self.collection_name in names:
                col = self._client.get_collection(self.collection_name)
                if col.count() > 0:
                    return Chroma(
                        client=self._client,
                        collection_name=self.collection_name,
                        embedding_function=self._embeddings,
                    )
        except Exception:
            pass
        return None

    @property
    def langchain_store(self) -> Optional[Chroma]:
        """Mevcut DenseRetriever bu attribute'e doğrudan ihtiyaç duyabilir."""
        return self._lc_store

    def upsert_documents(self, documents: List[Any]) -> None:
        # Eski koleksiyon varsa sil — full reindex semantiği
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._lc_store = Chroma.from_documents(
            documents=documents,
            embedding=self._embeddings,
            client=self._client,
            collection_name=self.collection_name,
        )

    def similarity_search(self, query: str, k: int = 5) -> List[VectorSearchResult]:
        if self._lc_store is None:
            return []
        docs = self._lc_store.similarity_search(query, k=k)
        return [
            VectorSearchResult(page_content=d.page_content, metadata=dict(d.metadata))
            for d in docs
        ]

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Any]:
        if self._lc_store is None:
            return []
        return self._lc_store.similarity_search_with_score(query, k=k)

    def count(self) -> int:
        try:
            col = self._client.get_collection(self.collection_name)
            return col.count()
        except Exception:
            return 0

    def delete_collection(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._lc_store = None


VectorStoreFactory.register(
    "chroma",
    ChromaVectorStore,
    label="ChromaDB (default)",
    desc="Local, embed-only, sıfır setup. Tek dosya, kolay backup.",
)
