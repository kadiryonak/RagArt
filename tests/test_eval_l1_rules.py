"""Unit tests for L1 rule-based evaluator."""

from __future__ import annotations

import pytest

from tests.evaluation.dataset import GoldenItem
from tests.evaluation.layers.base import RAGOutput
from tests.evaluation.layers.l1_rules import L1RulesEvaluator


def _item(**kw) -> GoldenItem:
    defaults = dict(
        id="t",
        question="Algoritma nedir?",
        reference_answer="Algoritma adım adım bir çözüm yoludur.",
        keywords=["algoritma", "adım"],
        expected_sources=[],
        difficulty="easy",
        category="factual",
        critical=False,
    )
    defaults.update(kw)
    return GoldenItem(**defaults)


def _output(answer: str) -> RAGOutput:
    return RAGOutput(answer=answer, retrieved_sources=[], retrieved_context="")


class TestLength:
    def test_too_short_fails(self):
        ev = L1RulesEvaluator()
        r = ev.evaluate(_item(), _output("kısa"))
        assert r.details["length"]["verdict"] == "too_short"
        assert r.details["length"]["score"] == 0.0

    def test_ok_length_passes(self):
        ev = L1RulesEvaluator()
        ans = "Algoritma, bir problemi çözmek için adım adım takip edilen yöntemdir."
        r = ev.evaluate(_item(), _output(ans))
        assert r.details["length"]["verdict"] == "ok"
        assert r.details["length"]["score"] == 1.0

    def test_too_long_partial_penalty(self):
        ev = L1RulesEvaluator()
        long_ans = "algoritma adım " * 500  # ~7500 chars
        r = ev.evaluate(_item(), _output(long_ans))
        assert r.details["length"]["verdict"] == "too_long"
        assert r.details["length"]["score"] == 0.5


class TestLanguage:
    def test_turkish_text_high_score(self):
        ev = L1RulesEvaluator()
        ans = "Algoritma, bir problemi çözmek için adım adım uygulanan yöntemdir."
        r = ev.evaluate(_item(), _output(ans))
        assert r.details["language"]["score"] >= 0.7

    def test_pure_english_low_score(self):
        ev = L1RulesEvaluator()
        ans = "An algorithm is a step by step procedure to solve a problem."
        r = ev.evaluate(_item(), _output(ans))
        assert r.details["language"]["score"] == 0.0


class TestKeywords:
    def test_all_keywords_present(self):
        ev = L1RulesEvaluator()
        ans = "Algoritma adım adım yapılan işlemdir."
        r = ev.evaluate(_item(keywords=["algoritma", "adım"]), _output(ans))
        assert r.details["keywords"]["score"] == 1.0
        assert r.details["keywords"]["missing"] == []

    def test_partial_keywords(self):
        ev = L1RulesEvaluator()
        ans = "Algoritma bir tanımlamadır."  # 'adım' yok
        r = ev.evaluate(_item(keywords=["algoritma", "adım"]), _output(ans))
        assert r.details["keywords"]["score"] == 0.5
        assert "adım" in r.details["keywords"]["missing"]

    def test_no_required_keywords(self):
        ev = L1RulesEvaluator()
        r = ev.evaluate(_item(keywords=[]), _output("herhangi bir cevap çok güzel oldu."))
        assert r.details["keywords"]["score"] == 1.0


class TestBadPatterns:
    @pytest.mark.parametrize(
        "answer",
        [
            "I don't know the answer to that question.",
            "As an AI language model, I cannot help.",
            "Bu konuyla ilgili bilgim yok.",
            "Yardımcı olamıyorum bu konuda maalesef.",
        ],
    )
    def test_bad_pattern_zeroes_score(self, answer):
        ev = L1RulesEvaluator()
        r = ev.evaluate(_item(), _output(answer))
        assert r.details["bad_patterns"]["score"] == 0.0

    def test_good_answer_no_hits(self):
        ev = L1RulesEvaluator()
        ans = "Algoritma, problemleri çözmek için adım adım uygulanan tekniklerdir."
        r = ev.evaluate(_item(), _output(ans))
        assert r.details["bad_patterns"]["score"] == 1.0


class TestOverall:
    def test_perfect_answer_passes(self):
        ev = L1RulesEvaluator()
        ans = "Algoritma, bir problemi çözmek için adım adım uygulanan yöntemdir ve birçok alanda kullanılır."
        r = ev.evaluate(_item(keywords=["algoritma", "adım"]), _output(ans))
        assert r.passed is True
        assert r.score > 0.85

    def test_empty_answer_fails(self):
        ev = L1RulesEvaluator()
        r = ev.evaluate(_item(), _output(""))
        assert r.passed is False
        assert r.score == 0.0

    def test_evaluator_never_throws(self):
        """BaseEvaluator wrap'i exception'ları yakalamalı."""
        ev = L1RulesEvaluator()
        r = ev.evaluate(_item(), _output(None))  # type: ignore
        assert isinstance(r.score, float)
