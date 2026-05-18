"""Retriever plugins.

Mimari prensip: tüm retriever'lar aynı BaseRetriever sözleşmesini uygular.
Bu sayede dense/sparse/hybrid arasında geçiş kodun geri kalanını etkilemez.

Mevcut implementasyonlar:
    DenseRetriever   — ChromaDB üzerinden embedding similarity
    BM25Retriever    — In-memory BM25Okapi (sparse, exact-match için)
    HybridRetriever  — Dense + Sparse Reciprocal Rank Fusion (RRF)
"""

from src.retrievers.base import BaseRetriever, RetrievedDoc
from src.retrievers.dense import DenseRetriever
from src.retrievers.sparse import BM25Retriever
from src.retrievers.hybrid import HybridRetriever
from src.retrievers.reranker import RerankedRetriever

__all__ = [
    "BaseRetriever",
    "RetrievedDoc",
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "RerankedRetriever",
]
