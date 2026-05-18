"""Unit tests for L4 Groq LLM judge — Groq API tamamen mock'lanır."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tests.evaluation.dataset import GoldenItem
from tests.evaluation.layers.base import RAGOutput
from tests.evaluation.layers.l4_judge import (
    L4JudgeEvaluator,
    _extract_json,
    _normalize_likert,
)


def _critical_item() -> GoldenItem:
    return GoldenItem(
        id="t", question="q", reference_answer="ref",
        keywords=[], expected_sources=[],
        difficulty="medium", category="factual", critical=True,
    )


def _non_critical_item() -> GoldenItem:
    item = _critical_item()
    item.critical = False
    return item


def _mock_groq_response(faithfulness=5, relevance=5, completeness=5, reasoning="iyi"):
    """Groq API başarılı response'unu taklit eden mock requests modülü."""
    mock = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "faithfulness": faithfulness,
                    "relevance": relevance,
                    "completeness": completeness,
                    "reasoning": reasoning,
                })
            }
        }]
    }
    mock.post.return_value = response
    return mock


class TestNormalizeLikert:
    def test_min_score(self):
        assert _normalize_likert(1.0) == 0.0

    def test_max_score(self):
        assert _normalize_likert(5.0) == 1.0

    def test_mid_score(self):
        assert _normalize_likert(3.0) == 0.5

    def test_out_of_range_clamped(self):
        assert _normalize_likert(0) == 0.0
        assert _normalize_likert(10) == 1.0


class TestJsonExtraction:
    def test_clean_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_json_with_surrounding_text(self):
        text = 'Here is my answer: {"faithfulness": 5} done.'
        assert _extract_json(text) == {"faithfulness": 5}

    def test_invalid_returns_none(self):
        assert _extract_json("no json here") is None


class TestAvailability:
    def test_not_available_without_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        ev = L4JudgeEvaluator(api_key=None)
        assert ev.is_available() is False

    def test_available_with_key(self):
        ev = L4JudgeEvaluator(api_key="fake")
        assert ev.is_available() is True


class TestEvaluation:
    def test_skipped_when_not_critical(self):
        ev = L4JudgeEvaluator(api_key="fake", only_critical=True)
        r = ev.evaluate(_non_critical_item(), RAGOutput(answer="x"))
        assert r.details["skipped"] is True
        assert r.score == 0.0

    def test_perfect_scores(self):
        mock = _mock_groq_response(5, 5, 5)
        ev = L4JudgeEvaluator(api_key="fake", client=mock)
        r = ev.evaluate(_critical_item(), RAGOutput(answer="iyi cevap"))
        assert r.score == pytest.approx(1.0)
        assert r.passed is True
        assert r.details["faithfulness"]["raw"] == 5

    def test_partial_scores(self):
        # faithfulness=3, relevance=4, completeness=2 → normalize:
        # 0.5, 0.75, 0.25
        # weighted: 0.45*0.5 + 0.30*0.75 + 0.25*0.25 ≈ 0.5125
        mock = _mock_groq_response(3, 4, 2)
        ev = L4JudgeEvaluator(api_key="fake", client=mock)
        r = ev.evaluate(_critical_item(), RAGOutput(answer="orta cevap"))
        assert r.score == pytest.approx(0.5125, abs=1e-3)

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        ev = L4JudgeEvaluator(api_key=None)
        r = ev.evaluate(_critical_item(), RAGOutput(answer="x"))
        # BaseEvaluator wrap'i error'u tutar
        assert r.error is not None
        assert "GROQ_API_KEY" in r.error

    def test_api_error_propagates(self):
        mock = MagicMock()
        response = MagicMock()
        response.status_code = 500
        response.text = "internal error"
        mock.post.return_value = response

        ev = L4JudgeEvaluator(api_key="fake", client=mock)
        r = ev.evaluate(_critical_item(), RAGOutput(answer="x"))
        assert r.error is not None
        assert "500" in r.error

    def test_unparseable_response_handled(self):
        mock = MagicMock()
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "choices": [{"message": {"content": "garbage no json"}}]
        }
        mock.post.return_value = response

        ev = L4JudgeEvaluator(api_key="fake", client=mock)
        r = ev.evaluate(_critical_item(), RAGOutput(answer="x"))
        assert r.error is not None
