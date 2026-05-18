"""Cross-encoder reranker — multi-stage retrieval'in son adımı.

NEDEN RERANKER?
    Bi-encoder retriever'lar (dense, BM25, hybrid) hızlıdır ama kaba bir
    relevance ölçer:
        - Dense:  query ve doc ayrı ayrı embed edilir, cosine alınır
        - BM25:   keyword istatistikleri, doc-içi context bilgisi kaybolur

    Cross-encoder ise (query, doc) çiftini birlikte encode eder ve relevance
    skorunu doğrudan tahmin eder. Çok daha doğru ama her bir doc için ayrı
    forward pass gerekir — yavaş.

    Klasik production pattern: bi-encoder ile top-N (örn. 50) hızlıca
    getir, cross-encoder ile top-K'ya (örn. 5) daralt.

MODEL
    Default: BAAI/bge-reranker-v2-m3 — multilingual, CPU'da çalışır,
    ~400MB. Türkçe için sentence-transformers tabanlı multilingual modeller
    içinde en güçlülerden.

LATENCY
    bge-reranker-v2-m3, CPU üzerinde tipik 30-50ms / çift. top-20 → top-5
    için ~600-1000ms eklenir. GPU varsa 10x hızlanır.

API
    RerankedRetriever(base_retriever, fetch_k=20, top_k=5, model=...)
    base_retriever'dan fetch_k al, reranker ile sırala, top_k döndür.
    BaseRetriever sözleşmesini uygular → diğer retriever'larla
    interchangeable.

TEST EDİLEBİLİRLİK
    CrossEncoder örneklemesi pahalı (~5-10 saniye yükleme + model). Bu
    yüzden:
        - Model lazy-load (ilk retrieve çağrısında)
        - Constructor'da inject edilebilir bir `cross_encoder` parametresi
          (mock testler için)
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

from src.retrievers.base import BaseRetriever, RetrievedDoc

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


CrossEncoderProtocol = Any  # ducktype: .predict(List[Tuple[str, str]]) -> List[float]


class RerankedRetriever(BaseRetriever):
    """Bi-encoder retriever'ı cross-encoder ile rerank eder.

    Args:
        base_retriever:  Aday üreten retriever (dense, BM25, hybrid).
        fetch_k:         Reranker için kaç aday istenecek (default 20).
                         Daha çok aday = daha yavaş ama recall artar.
                         retrieve(k=N) çağrısında etkin sayı max(fetch_k, N).
        model:           Cross-encoder model adı (HF). Default bge-reranker-v2-m3.
        cross_encoder:   Inject edilen önceden yüklenmiş model (testler için).
                         Verilirse `model` yok-sayılır.
        loader:          Custom yükleyici (testler için). (model_name) → encoder.
    """

    name = "reranked"

    def __init__(
        self,
        base_retriever: BaseRetriever,
        *,
        fetch_k: int = 20,
        model: str = DEFAULT_RERANKER_MODEL,
        cross_encoder: Optional[CrossEncoderProtocol] = None,
        loader: Optional[Callable[[str], CrossEncoderProtocol]] = None,
    ):
        if fetch_k < 1:
            raise ValueError(f"fetch_k must be >= 1, got {fetch_k}")
        self.base_retriever = base_retriever
        self.fetch_k = fetch_k
        self.model_name = model
        self._cross_encoder: Optional[CrossEncoderProtocol] = cross_encoder
        self._loader = loader or self._default_loader

    @staticmethod
    def _default_loader(model_name: str) -> CrossEncoderProtocol:
        # Lazy import: import maliyeti yüklemeye kadar ertelenir
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)

    @property
    def cross_encoder(self) -> CrossEncoderProtocol:
        """Lazy-load the model on first access."""
        if self._cross_encoder is None:
            self._cross_encoder = self._loader(self.model_name)
        return self._cross_encoder

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedDoc]:
        if not query.strip():
            return []

        # Etkin fetch_k: oversample, ama en az k
        effective_fetch = max(self.fetch_k, k)
        candidates = self.base_retriever.retrieve(query, k=effective_fetch)
        if not candidates:
            return []

        # Cross-encoder relevance prediction
        pairs: List[Tuple[str, str]] = [(query, c.page_content) for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Liste/np.ndarray → liste
        scores_list = [float(s) for s in scores]

        # Skora göre azalan sıralama
        ranked: List[Tuple[float, RetrievedDoc]] = sorted(
            zip(scores_list, candidates),
            key=lambda pair: -pair[0],
        )

        # Yeni RetrievedDoc'lar: cross-encoder skoru ile
        out: List[RetrievedDoc] = []
        for score, doc in ranked[:k]:
            out.append(RetrievedDoc(
                page_content=doc.page_content,
                metadata=doc.metadata,
                score=score,
                doc_id=doc.get_id(),
            ))
        return out
