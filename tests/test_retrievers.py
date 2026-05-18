"""Unit tests for retriever plugins (dense / sparse / hybrid)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.retrievers import BaseRetriever, RetrievedDoc, BM25Retriever, HybridRetriever
from src.retrievers.dense import DenseRetriever


# ----- Helpers -----


@dataclass
class _Doc:
    """Mini langchain-Document benzeri yapı."""

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def make_docs(*pairs):
    """make_docs(('source1.json', 'algoritma adım'), ...) → [_Doc...]."""
    out = []
    for i, (src, content) in enumerate(pairs):
        out.append(_Doc(page_content=content, metadata={"source": src, "item_index": i}))
    return out


# ----- RetrievedDoc -----


class TestRetrievedDoc:
    def test_source_property(self):
        d = RetrievedDoc(page_content="x", metadata={"source": "foo.json"})
        assert d.source == "foo.json"

    def test_source_default_unknown(self):
        d = RetrievedDoc(page_content="x")
        assert d.source == "unknown"

    def test_id_from_explicit(self):
        d = RetrievedDoc(page_content="x", doc_id="explicit")
        assert d.get_id() == "explicit"

    def test_id_from_metadata(self):
        d = RetrievedDoc(page_content="x", metadata={"source": "a.json", "item_index": 7})
        assert d.get_id() == "a.json::7"

    def test_id_fallback_to_hash(self):
        d = RetrievedDoc(page_content="some content", metadata={"source": "a.json"})
        id1 = d.get_id()
        d2 = RetrievedDoc(page_content="some content", metadata={"source": "a.json"})
        assert id1 == d2.get_id()  # deterministic


# ----- DenseRetriever -----


class TestDenseRetriever:
    def test_calls_vector_store_similarity(self):
        vs = MagicMock()
        vs.similarity_search_with_score.return_value = [
            (_Doc("Algoritma adım adım", {"source": "Algoritma.json"}), 0.4),
            (_Doc("Python kod yazımı", {"source": "Python.json"}), 1.2),
        ]
        r = DenseRetriever(vs)
        out = r.retrieve("algoritma", k=2)
        assert len(out) == 2
        # ChromaDB distance → score = 1 - d/2
        assert out[0].score == pytest.approx(0.8)
        assert out[1].score == pytest.approx(0.4)
        vs.similarity_search_with_score.assert_called_once_with("algoritma", k=2)

    def test_empty_vector_store(self):
        r = DenseRetriever(None)
        assert r.retrieve("test") == []

    def test_score_clamped_to_zero(self):
        vs = MagicMock()
        vs.similarity_search_with_score.return_value = [
            (_Doc("x", {}), 3.0),  # huge distance → would give negative score
        ]
        out = DenseRetriever(vs).retrieve("q", k=1)
        assert out[0].score == 0.0


# ----- BM25Retriever -----


class TestBM25Retriever:
    def test_exact_keyword_match_wins(self):
        docs = make_docs(
            ("Veri_bilimi.json", "Veri bilimi istatistik makine öğrenmesi"),
            ("Veri_yapıları.json", "Veri yapıları dizi liste yığın kuyruk"),
            ("Python.json", "Python programlama dili"),
        )
        r = BM25Retriever(docs)
        out = r.retrieve("veri yapıları nedir", k=2)
        # exact term match: "yapıları" → Veri_yapıları.json en üstte olmalı
        assert len(out) >= 1
        assert out[0].source == "Veri_yapıları.json"

    def test_query_with_no_match_returns_empty(self):
        docs = make_docs(("a.json", "algoritma"), ("b.json", "veri"))
        r = BM25Retriever(docs)
        out = r.retrieve("xyzqwerty nonsense", k=5)
        assert out == []

    def test_empty_documents_safe(self):
        r = BM25Retriever([])
        assert r.is_empty
        assert r.retrieve("anything") == []

    def test_empty_query_returns_empty(self):
        docs = make_docs(("a.json", "test"))
        r = BM25Retriever(docs)
        assert r.retrieve("") == []
        assert r.retrieve("   ") == []

    def test_top_k_respected(self):
        # Need varied content so BM25 IDF produces non-zero scores
        docs = make_docs(
            ("a.json", "algoritma adım adım yöntem matematik bilim"),
            ("b.json", "algoritma temel yapı taşı tasarım problem"),
            ("c.json", "algoritma analiz karmaşıklık verimlilik"),
            ("d.json", "algoritma örnekleri sıralama arama"),
            ("e.json", "algoritma tasarımı pseudocode programlama"),
            ("f.json", "yazılım mühendisliği sürecleri"),
            ("g.json", "veri tabanı yönetimi"),
            ("h.json", "şu konu algoritma tamamen farklı bir alan"),
        )
        r = BM25Retriever(docs)
        out = r.retrieve("algoritma karmaşıklık analiz", k=3)
        assert len(out) == 3

    def test_results_score_sorted_descending(self):
        docs = make_docs(
            ("a.json", "kelime kelime kelime"),       # 3 hit
            ("b.json", "kelime"),                      # 1 hit
            ("c.json", "kelime kelime"),               # 2 hit
        )
        r = BM25Retriever(docs)
        out = r.retrieve("kelime", k=3)
        # azalan score
        for i in range(len(out) - 1):
            assert out[i].score >= out[i + 1].score


# ----- HybridRetriever (RRF) -----


class _StaticRetriever(BaseRetriever):
    """Verilen sıralı listeyi olduğu gibi döndüren basit retriever (test için)."""

    def __init__(self, docs: List[RetrievedDoc], name: str = "static"):
        self._docs = docs
        self.name = name

    def retrieve(self, query: str, k: int = 5) -> List[RetrievedDoc]:
        return list(self._docs[:k])


class TestRRFMath:
    def test_both_retrievers_agree_top_doc_wins(self):
        d1 = RetrievedDoc("A", {"source": "a.json", "item_index": 0}, score=0.9)
        d2 = RetrievedDoc("B", {"source": "b.json", "item_index": 1}, score=0.8)
        d3 = RetrievedDoc("C", {"source": "c.json", "item_index": 2}, score=0.7)

        dense  = _StaticRetriever([d1, d2, d3])
        sparse = _StaticRetriever([d1, d2, d3])

        hy = HybridRetriever(dense=dense, sparse=sparse, k_rrf=60, oversample=1)
        out = hy.retrieve("q", k=3)
        assert [d.source for d in out] == ["a.json", "b.json", "c.json"]

    def test_disagreement_returns_union(self):
        # Tam zıt sıralamada, RRF tüm dokümanları en az bir kez döndürmeli
        # (tie-break sırası uygulamaya bağlı — biz sadece set'i doğruluyoruz).
        d1 = RetrievedDoc("A", {"source": "a.json", "item_index": 0})
        d2 = RetrievedDoc("B", {"source": "b.json", "item_index": 1})
        d3 = RetrievedDoc("C", {"source": "c.json", "item_index": 2})

        dense  = _StaticRetriever([d1, d2, d3])
        sparse = _StaticRetriever([d3, d2, d1])

        hy = HybridRetriever(dense=dense, sparse=sparse, k_rrf=60, oversample=1)
        out = hy.retrieve("q", k=3)
        assert {d.source for d in out} == {"a.json", "b.json", "c.json"}

    def test_top_ranks_in_both_beat_only_in_one(self):
        # X dense ve sparse'da rank 1; Y sadece dense'in arka kısmında.
        # X her zaman üstte olmalı.
        x = RetrievedDoc("X", {"source": "x.json", "item_index": 0})
        y = RetrievedDoc("Y", {"source": "y.json", "item_index": 1})
        z = RetrievedDoc("Z", {"source": "z.json", "item_index": 2})

        dense  = _StaticRetriever([x, y, z])
        sparse = _StaticRetriever([x])  # Y/Z sparse'ta yok

        hy = HybridRetriever(dense=dense, sparse=sparse, oversample=1)
        out = hy.retrieve("q", k=3)
        assert out[0].source == "x.json"

    def test_only_in_sparse(self):
        # Dense'in kaçırdığı doküman sparse'da bulunur → hybrid hâlâ döndürür
        d_dense = RetrievedDoc("D", {"source": "d.json", "item_index": 0})
        d_sparse_only = RetrievedDoc("S", {"source": "s.json", "item_index": 1})

        dense  = _StaticRetriever([d_dense])
        sparse = _StaticRetriever([d_sparse_only])

        hy = HybridRetriever(dense=dense, sparse=sparse, oversample=1)
        out = hy.retrieve("q", k=2)
        sources = {d.source for d in out}
        assert "d.json" in sources
        assert "s.json" in sources

    def test_dense_weight_dominates(self):
        # Aynı sıralama farkıyla, dense_weight yüksek → dense'in tercihi kazanır
        d_top = RetrievedDoc("T", {"source": "t.json", "item_index": 0})
        d_bot = RetrievedDoc("B", {"source": "b.json", "item_index": 1})

        dense  = _StaticRetriever([d_top, d_bot])
        sparse = _StaticRetriever([d_bot, d_top])

        # Dense ağırlığı sparse'ten 3x büyük
        hy = HybridRetriever(
            dense=dense, sparse=sparse,
            dense_weight=3.0, sparse_weight=1.0, oversample=1,
        )
        out = hy.retrieve("q", k=2)
        assert out[0].source == "t.json"

    def test_oversample_pulls_more_from_each(self):
        # Bu test sadece davranışsal: oversample=4 ile her retriever k*4 çağrılır
        d_calls = MagicMock(wraps=_StaticRetriever([], name="d"))
        s_calls = MagicMock(wraps=_StaticRetriever([], name="s"))

        hy = HybridRetriever(dense=d_calls, sparse=s_calls, oversample=4)
        hy.retrieve("q", k=5)
        d_calls.retrieve.assert_called_with("q", k=20)
        s_calls.retrieve.assert_called_with("q", k=20)

    def test_empty_query_short_circuit(self):
        hy = HybridRetriever(
            dense=_StaticRetriever([RetrievedDoc("X")]),
            sparse=_StaticRetriever([RetrievedDoc("X")]),
        )
        assert hy.retrieve("") == []
        assert hy.retrieve("   ") == []


# ----- BM25 → Hybrid integration (real BM25 + static dense) -----


class TestHybridWithRealBM25:
    """v0/v1 baseline'da bulduğumuz retrieval-bias problemini doğrula:
       'veri yapıları' sorusu → Veri_yapıları.json üstte olmalı."""

    def test_bm25_alone_fixes_medium_03(self):
        # BM25 tek başına 'yapıları' exact-match yaparak Veri_yapıları'yı en üstte verir
        docs = make_docs(
            ("Veri_bilimi.json", "veri bilimi istatistik makine öğrenmesi alanı"),
            ("Veri_yapıları.json", "veri yapıları dizi bağlı liste yığın kuyruk ağaç"),
            ("Python.json", "Python yorumlamalı yüksek seviye programlama"),
        )
        bm25 = BM25Retriever(docs)
        out = bm25.retrieve("veri yapıları nedir", k=3)
        assert out[0].source == "Veri_yapıları.json"

    def test_hybrid_promotes_doc_dense_missed(self):
        # Dense Veri_yapıları'yı en altta (rank 3); BM25 en üstte (rank 1).
        # Hybrid: dense'in tek başına yapamayacağı şekilde top-2'ye sokar.
        docs = make_docs(
            ("Veri_bilimi.json", "veri bilimi istatistik makine öğrenmesi alanı"),
            ("Veri_yapıları.json", "veri yapıları dizi bağlı liste yığın kuyruk ağaç"),
            ("Python.json", "Python yorumlamalı yüksek seviye programlama"),
        )
        bm25 = BM25Retriever(docs)

        # Dense neredeyse hiç işe yaramıyor: Veri_yapıları'yı en altta veriyor
        dense_fake = _StaticRetriever([
            RetrievedDoc("veri bilimi…", {"source": "Veri_bilimi.json", "item_index": 0}),
            RetrievedDoc("Python…", {"source": "Python.json", "item_index": 2}),
            RetrievedDoc("veri yapıları…", {"source": "Veri_yapıları.json", "item_index": 1}),
        ])

        hybrid = HybridRetriever(dense=dense_fake, sparse=bm25, oversample=1)
        out = hybrid.retrieve("veri yapıları nedir", k=2)
        sources = [d.source for d in out]
        # Veri_yapıları top-2'de olmalı (dense tek başına oraya koyamaz)
        assert "Veri_yapıları.json" in sources, f"Expected Veri_yapıları in top-2, got: {sources}"
