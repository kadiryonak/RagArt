"""Unit tests for L2 vector similarity evaluator.

Mock embedder kullanır — gerçek model yüklemeye gerek yok.
"""

from __future__ import annotations

import math

import pytest

from tests.evaluation.dataset import GoldenItem
from tests.evaluation.layers.base import RAGOutput
from tests.evaluation.layers.l2_vector import (
    L2VectorEvaluator,
    cosine_similarity,
    normalize_to_unit,
)


def _item(reference: str = "ref") -> GoldenItem:
    return GoldenItem(
        id="t",
        question="q?",
        reference_answer=reference,
        keywords=[],
        expected_sources=[],
        difficulty="easy",
        category="factual",
    )


class TestCosineHelper:
    def test_identical_vectors(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector_safe(self):
        assert cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_normalize_range(self):
        assert normalize_to_unit(1.0) == 1.0
        assert normalize_to_unit(0.0) == 0.5
        assert normalize_to_unit(-1.0) == 0.0


class TestL2Evaluator:
    def test_identical_strings_score_one(self):
        """Aynı string için fake embedder hep aynı vektörü verir → cos=1."""
        def fake_embed(text: str):
            return [1.0, 0.0, 0.0]

        ev = L2VectorEvaluator(embed_fn=fake_embed)
        r = ev.evaluate(_item("aynı"), RAGOutput(answer="aynı"))
        assert r.details["answer_vs_reference"]["cosine"] == pytest.approx(1.0)
        assert r.score >= 0.9

    def test_orthogonal_low_score(self):
        """answer ve reference için farklı vektör → cos=0 → normalize=0.5."""
        def fake_embed(text: str):
            return [1.0, 0.0] if "ans" in text else [0.0, 1.0]

        ev = L2VectorEvaluator(embed_fn=fake_embed, ctx_weight=0.0)
        r = ev.evaluate(_item("ref"), RAGOutput(answer="ans"))
        assert r.details["answer_vs_reference"]["cosine"] == pytest.approx(0.0)
        assert r.score == pytest.approx(0.5)

    def test_with_context(self):
        """ctx_weight verildiğinde context skoru da hesaba katılır."""
        def fake_embed(text: str):
            mapping = {"a": [1, 0, 0], "r": [1, 0, 0], "c": [1, 0, 0]}
            return mapping.get(text, [0, 0, 0])

        ev = L2VectorEvaluator(embed_fn=fake_embed, ctx_weight=0.5)
        r = ev.evaluate(
            _item("r"),
            RAGOutput(answer="a", retrieved_context="c"),
        )
        # Hem ref hem context aynı → her ikisi de 1.0 → toplam 1.0
        assert r.score == pytest.approx(1.0)
        assert r.details["answer_vs_context"] is not None

    def test_empty_answer_returns_zero(self):
        def fake_embed(text: str):
            return [1.0, 0.0]

        ev = L2VectorEvaluator(embed_fn=fake_embed)
        r = ev.evaluate(_item(), RAGOutput(answer=""))
        assert r.score == 0.0
        assert r.details["reason"] == "empty_answer"

    def test_embed_fn_called_correctly(self):
        calls = []

        def tracking_embed(text: str):
            calls.append(text)
            return [1.0]

        ev = L2VectorEvaluator(embed_fn=tracking_embed)
        ev.evaluate(_item("REF"), RAGOutput(answer="ANS"))
        assert "ANS" in calls
        assert "REF" in calls
