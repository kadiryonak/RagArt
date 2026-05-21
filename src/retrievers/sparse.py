"""Sparse (BM25) retriever.

BM25 (Best Match 25) klasik bir term-frequency × inverse-document-frequency
sıralamasıdır. Embedding'in zayıf olduğu yerlerde — özel isim, kısaltma,
kod, isim hatası vb. — exact match yapar.

Implementation: rank_bm25.BM25Okapi (basit ve hızlı, ~50KB pure Python).
Index in-memory; reindex sırasında rebuild edilir.

Türkçe için: L3 evaluator'deki ``tokenize`` fonksiyonunu kullanıyoruz —
hafif suffix stripping ile çekim cezasını azaltır.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from rank_bm25 import BM25Okapi

from src.retrievers.base import BaseRetriever, RetrievedDoc

# Shared Turkish tokenizer — the L3 lexical evaluator uses the same one.
from src.text_utils import tokenize as _tokenize


class BM25Retriever(BaseRetriever):
    """In-memory BM25Okapi retriever."""

    name = "sparse"

    def __init__(
        self,
        documents: Sequence[Any],  # langchain Document or similar
        *,
        k1: float = 1.5,
        b: float = 0.75,
        stem: bool = True,
    ):
        """
        Args:
            documents: page_content + metadata olan belgeler listesi.
            k1, b:     BM25 hyperparam'ları. Default'lar production'da yaygın.
            stem:      Türkçe hafif suffix stripping uygula.
        """
        self.documents = list(documents)
        self.stem = stem
        self._tokenized_corpus = [
            _tokenize(d.page_content, stem=stem) for d in self.documents
        ]
        # rank_bm25 boş corpus'a izin vermez
        if not self._tokenized_corpus:
            self._bm25 = None
        else:
            self._bm25 = BM25Okapi(self._tokenized_corpus, k1=k1, b=b)

    @property
    def is_empty(self) -> bool:
        return self._bm25 is None

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedDoc]:
        if self.is_empty:
            return []

        tokens = _tokenize(query, stem=self.stem)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)  # numpy array

        # Top-k indices, score'a göre azalan
        # numpy argsort hızlı; manuel olarak da yapılabilir
        import numpy as np
        top_idx = np.argsort(-scores)[:k]

        out: List[RetrievedDoc] = []
        for idx in top_idx:
            s = float(scores[idx])
            if s <= 0.0:
                continue  # match yok
            doc = self.documents[idx]
            out.append(RetrievedDoc(
                page_content=doc.page_content,
                metadata=dict(doc.metadata),
                score=s,
            ))
        return out
