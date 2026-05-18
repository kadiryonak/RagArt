"""Unit tests for RerankedRetriever — CrossEncoder fully mocked.

Test prensibi: gerçek bge-reranker-v2-m3 ~400MB ve yüklemesi 5-10sn.
Bu yüzden:
    - cross_encoder parametresi ile MagicMock inject
    - loader ile lazy-load davranışı test edilebilir
    - Asla gerçek model indirilmez
"""

from __future__ import annotations

from typing import List, Tuple
from unittest.mock import MagicMock

import pytest

from src.retrievers import BaseRetriever, RerankedRetriever, RetrievedDoc


class _StaticRetriever(BaseRetriever):
    def __init__(self, docs: List[RetrievedDoc]):
        self._docs = docs

    def retrieve(self, query: str, k: int = 5):
        return list(self._docs[:k])


def _doc(content: str, src: str, idx: int = 0) -> RetrievedDoc:
    return RetrievedDoc(
        page_content=content,
        metadata={"source": src, "item_index": idx},
        score=0.5,
    )


def _mock_encoder(score_for_content: dict):
    """Belirli içerikler için tahmini skor döndüren mock cross-encoder."""
    m = MagicMock()
    def predict(pairs: List[Tuple[str, str]]):
        return [score_for_content.get(content, 0.0) for _, content in pairs]
    m.predict.side_effect = predict
    return m


class TestRerankerOrdering:
    def test_reorders_by_cross_encoder_score(self):
        docs = [
            _doc("Yapay zeka makinelerin öğrenmesi", "AI.json", 0),
            _doc("Algoritma adım adım yönergeler", "Algo.json", 1),
            _doc("Python yorumlamalı bir dildir", "Py.json", 2),
        ]
        base = _StaticRetriever(docs)

        # Mock encoder: Algoritma sorgusuna en yüksek skoru Algo.json'a verir
        encoder = _mock_encoder({
            docs[0].page_content: 0.1,
            docs[1].page_content: 0.9,   # winner
            docs[2].page_content: 0.3,
        })

        rr = RerankedRetriever(base, fetch_k=3, cross_encoder=encoder)
        out = rr.retrieve("algoritma nedir", k=2)

        assert out[0].source == "Algo.json"
        assert out[1].source == "Py.json"
        # En yüksek skor da reflect edilmiş olmalı
        assert out[0].score == pytest.approx(0.9)

    def test_top_k_respected(self):
        docs = [_doc(f"content-{i}", f"f{i}.json", i) for i in range(10)]
        encoder = _mock_encoder({d.page_content: float(10 - i) for i, d in enumerate(docs)})

        rr = RerankedRetriever(_StaticRetriever(docs), fetch_k=10, cross_encoder=encoder)
        out = rr.retrieve("query", k=3)
        assert len(out) == 3


class TestOversampling:
    def test_fetches_fetch_k_from_base(self):
        base = MagicMock(spec=BaseRetriever)
        base.retrieve.return_value = []
        encoder = _mock_encoder({})

        rr = RerankedRetriever(base, fetch_k=20, cross_encoder=encoder)
        rr.retrieve("q", k=5)
        # fetch_k=20 ile base'i çağırmalı, k=5 değil
        base.retrieve.assert_called_with("q", k=20)

    def test_k_larger_than_fetch_k_still_fetches_k(self):
        # Çağıran k=50 isterse fetch_k=10'dan büyükse 50 kullan
        base = MagicMock(spec=BaseRetriever)
        base.retrieve.return_value = []
        encoder = _mock_encoder({})

        rr = RerankedRetriever(base, fetch_k=10, cross_encoder=encoder)
        rr.retrieve("q", k=50)
        base.retrieve.assert_called_with("q", k=50)

    def test_fetch_k_zero_raises(self):
        with pytest.raises(ValueError, match="fetch_k"):
            RerankedRetriever(MagicMock(spec=BaseRetriever), fetch_k=0)


class TestEmptyEdgeCases:
    def test_empty_query_short_circuit(self):
        encoder = _mock_encoder({})
        rr = RerankedRetriever(_StaticRetriever([_doc("x", "x.json")]),
                               fetch_k=5, cross_encoder=encoder)
        assert rr.retrieve("") == []
        assert rr.retrieve("   ") == []
        # Encoder hiç çağrılmadı
        encoder.predict.assert_not_called()

    def test_empty_base_results(self):
        encoder = _mock_encoder({})
        rr = RerankedRetriever(_StaticRetriever([]), fetch_k=5, cross_encoder=encoder)
        assert rr.retrieve("q", k=3) == []
        encoder.predict.assert_not_called()


class TestLazyLoading:
    def test_no_model_load_until_first_retrieve(self):
        loader = MagicMock(return_value=MagicMock(predict=MagicMock(return_value=[])))
        rr = RerankedRetriever(
            _StaticRetriever([]),
            fetch_k=5,
            model="fake-model-name",
            loader=loader,
        )
        # Constructor sonrası loader hiç çağrılmamalı
        loader.assert_not_called()

        # Boş base → retrieve loader'ı hâlâ tetiklememeli (short circuit)
        rr.retrieve("q", k=2)
        loader.assert_not_called()

    def test_loader_called_with_model_name(self):
        loader = MagicMock(return_value=MagicMock(predict=MagicMock(return_value=[0.5])))
        rr = RerankedRetriever(
            _StaticRetriever([_doc("x", "x.json")]),
            fetch_k=5,
            model="my-custom-model",
            loader=loader,
        )
        rr.retrieve("q", k=1)
        loader.assert_called_once_with("my-custom-model")

    def test_loader_called_once(self):
        loader = MagicMock(return_value=MagicMock(predict=MagicMock(return_value=[0.5])))
        rr = RerankedRetriever(
            _StaticRetriever([_doc("x", "x.json")]),
            fetch_k=5,
            loader=loader,
        )
        rr.retrieve("q1", k=1)
        rr.retrieve("q2", k=1)
        rr.retrieve("q3", k=1)
        # Model yalnızca bir kez yüklenir
        assert loader.call_count == 1


class TestScoreSemantics:
    def test_reranker_score_replaces_base_score(self):
        # base 0.5 score veriyordu; reranker 0.92 vermeli
        docs = [_doc("foo", "f.json", 0)]
        encoder = _mock_encoder({"foo": 0.92})

        rr = RerankedRetriever(_StaticRetriever(docs), fetch_k=1, cross_encoder=encoder)
        out = rr.retrieve("q", k=1)
        assert out[0].score == pytest.approx(0.92)

    def test_doc_id_preserved_through_rerank(self):
        # fusion için kritik: rerank id'leri bozmuyor
        original = RetrievedDoc(
            page_content="content",
            metadata={"source": "x.json", "item_index": 7},
            doc_id="explicit-id",
        )
        encoder = _mock_encoder({"content": 0.8})
        rr = RerankedRetriever(_StaticRetriever([original]), fetch_k=1, cross_encoder=encoder)
        out = rr.retrieve("q", k=1)
        assert out[0].get_id() == "explicit-id"
