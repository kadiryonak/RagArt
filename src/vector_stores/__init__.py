"""Vector store plugins.

Soyut katman: tüm vektör DB'leri aynı BaseVectorStore sözleşmesini uygular.
Kullanıcı workspace oluştururken hangi DB'yi kullanacağını seçer.

Mevcut implementasyonlar:
    ChromaVectorStore  — ChromaDB (default, mevcut)
    QdrantVectorStore  — Qdrant (local file mode)

Sonradan eklenebilecekler: Weaviate, Pinecone, Milvus, FAISS, pgvector.
Her biri BaseVectorStore'u implement etmesi yeterli — UI dropdown'u
otomatik kayıtlardan üretir.
"""

from src.vector_stores.base import BaseVectorStore, VectorStoreFactory
from src.vector_stores.chroma_store import ChromaVectorStore
from src.vector_stores.qdrant_store import QdrantVectorStore

__all__ = [
    "BaseVectorStore",
    "VectorStoreFactory",
    "ChromaVectorStore",
    "QdrantVectorStore",
]
