"""Hybrid retriever — Reciprocal Rank Fusion (RRF) of multiple retrievers.

RRF, farklı retriever'ların sıralamalarını skor karşılaştırması yapmadan
birleştirmenin standart yöntemidir. Formül:

    rrf_score(d) = Σ_r  weight_r / (k_rrf + rank_r(d))

burada r her retriever, rank_r(d) belgenin o retriever'daki 1-tabanlı
sırası (yoksa terim eklenmez). k_rrf=60 makaledeki ampirik default.

Niye RRF?
    - Skor normalizasyonu gerekmez (BM25 ile cosine farklı ölçüde).
    - Outlier robust.
    - Hyperparam-free (sadece k_rrf).

Tipik kullanım:
    dense  = DenseRetriever(vector_store)
    sparse = BM25Retriever(docs)
    hybrid = HybridRetriever(dense=dense, sparse=sparse)
    docs   = hybrid.retrieve(query, k=5)
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

from src.retrievers.base import BaseRetriever, RetrievedDoc


class HybridRetriever(BaseRetriever):
    """RRF fusion of dense + sparse retrievers."""

    name = "hybrid"

    DEFAULT_K_RRF = 60
    DEFAULT_OVERSAMPLE = 4  # her retriever'dan k_final * 4 al, sonra füzyon

    def __init__(
        self,
        dense: BaseRetriever,
        sparse: BaseRetriever,
        *,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        k_rrf: int = DEFAULT_K_RRF,
        oversample: int = DEFAULT_OVERSAMPLE,
    ):
        self.dense = dense
        self.sparse = sparse
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.k_rrf = k_rrf
        self.oversample = max(1, int(oversample))

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedDoc]:
        if not query.strip():
            return []

        retrieve_k = k * self.oversample

        dense_docs = self.dense.retrieve(query, k=retrieve_k)
        sparse_docs = self.sparse.retrieve(query, k=retrieve_k)

        # id → (RetrievedDoc, rrf_score)
        scores: dict = defaultdict(float)
        docs_by_id: dict = {}

        for rank, doc in enumerate(dense_docs):
            did = doc.get_id()
            scores[did] += self.dense_weight / (self.k_rrf + rank + 1)
            # ilk gördüğümüz versiyonu sakla (rank-0 dense)
            docs_by_id.setdefault(did, doc)

        for rank, doc in enumerate(sparse_docs):
            did = doc.get_id()
            scores[did] += self.sparse_weight / (self.k_rrf + rank + 1)
            # Eğer dense'te yoksa, sparse versiyonunu ekle
            docs_by_id.setdefault(did, doc)

        # Azalan rrf_score sırasına göre top-k
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        out: List[RetrievedDoc] = []
        for did, rrf_score in ranked:
            base = docs_by_id[did]
            out.append(RetrievedDoc(
                page_content=base.page_content,
                metadata=base.metadata,
                score=rrf_score,
                doc_id=did,
            ))
        return out
