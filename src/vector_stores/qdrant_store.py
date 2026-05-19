"""QdrantVectorStore — local file mode adapter.

Qdrant production'da Rust-based, çok hızlı. ChromaDB'ye göre:
    + daha iyi filter desteği
    + REST + gRPC API
    - kurulum biraz daha karmaşık (Docker veya local file)

Burada `qdrant-client` ile **embedded local file mode** kullanıyoruz —
ChromaDB ile aynı UX, ekstra process yok.

Bu modül qdrant-client kurulu DEĞİLSE bile import edilebilir; sadece
construction sırasında hata verir. Bu sayede Chroma-only kullanıcılar
ekstra dep yüklemeden çalışmaya devam eder.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

from src.vector_stores.base import BaseVectorStore, VectorSearchResult, VectorStoreFactory


class QdrantVectorStore(BaseVectorStore):
    """Qdrant local file mode adapter."""

    name = "qdrant"

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

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
        except ImportError as e:
            raise RuntimeError(
                "qdrant-client kurulu değil. Şu komutla kurabilirsin: "
                "pip install qdrant-client"
            ) from e

        self._client = QdrantClient(path=persist_path)
        self._VectorParams = VectorParams
        self._Distance = Distance
        # Probe collection so count() works
        self._ensure_collection_exists()

    def _vector_dim(self) -> int:
        sample = self._embeddings.embed_query("probe")
        return len(sample)

    def _ensure_collection_exists(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self._VectorParams(
                    size=self._vector_dim(),
                    distance=self._Distance.COSINE,
                ),
            )

    def upsert_documents(self, documents: List[Any]) -> None:
        # Reindex semantiği: collection'ı baştan kur
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._ensure_collection_exists()

        from qdrant_client.http.models import PointStruct

        contents = [d.page_content for d in documents]
        vectors = self._embeddings.embed_documents(contents)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "page_content": doc.page_content,
                    "metadata": dict(doc.metadata),
                },
            )
            for doc, vec in zip(documents, vectors)
        ]
        # Batch upsert
        self._client.upsert(collection_name=self.collection_name, points=points)

    def _search(self, query_vector: List[float], k: int) -> List[Any]:
        return self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
        )

    def similarity_search(self, query: str, k: int = 5) -> List[VectorSearchResult]:
        if self.count() == 0:
            return []
        vec = self._embeddings.embed_query(query)
        results = self._search(vec, k)
        return [
            VectorSearchResult(
                page_content=r.payload.get("page_content", ""),
                metadata=r.payload.get("metadata", {}),
                score=float(r.score),
            )
            for r in results
        ]

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Any]:
        """LangChain Document + Qdrant cosine distance (1-similarity)."""
        try:
            from langchain_core.documents import Document
        except ImportError:
            from langchain.schema import Document

        if self.count() == 0:
            return []
        vec = self._embeddings.embed_query(query)
        results = self._search(vec, k)
        out = []
        for r in results:
            doc = Document(
                page_content=r.payload.get("page_content", ""),
                metadata=r.payload.get("metadata", {}),
            )
            # Qdrant 1=identical, 0=opposite for cosine; convert to "distance"
            # so DenseRetriever's score=1-d/2 formula stays approximately right
            distance = max(0.0, 2.0 * (1.0 - float(r.score)))
            out.append((doc, distance))
        return out

    def count(self) -> int:
        try:
            info = self._client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def delete_collection(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass


# Sadece qdrant-client yüklüyse register et — kurulu değilse UI listesinde gözükmesin
try:
    import qdrant_client  # noqa: F401
    VectorStoreFactory.register(
        "qdrant",
        QdrantVectorStore,
        label="Qdrant (local file)",
        desc="Rust-based, hızlı, gelişmiş filter desteği. Embedded mode.",
    )
except ImportError:
    pass
