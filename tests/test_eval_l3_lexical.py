"""Unit tests for L3 lexical evaluator (BLEU/ROUGE)."""

from __future__ import annotations

import pytest

from tests.evaluation.dataset import GoldenItem
from tests.evaluation.layers.base import RAGOutput
from tests.evaluation.layers.l3_lexical import (
    L3LexicalEvaluator,
    _lcs_length,
    bleu_n,
    bleu_score,
    brevity_penalty,
    rouge_l,
    tokenize,
)


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert tokenize("Algoritma Adım Adım", stem=False) == ["algoritma", "adım", "adım"]

    def test_removes_punctuation(self):
        assert tokenize("A, B. C!", stem=False) == ["a", "b", "c"]

    def test_keeps_turkish_chars(self):
        toks = tokenize("çöğüş", stem=False)
        assert toks == ["çöğüş"]

    def test_stemming_strips_common_suffix(self):
        # 'algoritmaları' → 'algoritma' (suffix 'ları' düşer)
        toks = tokenize("algoritmaları", stem=True)
        assert toks == ["algoritma"]


class TestBLEU:
    def test_identical_strings_perfect_score(self):
        a = ["bir", "iki", "üç", "dört"]
        result = bleu_score(a, a, max_n=2)
        assert result["bleu_1_precision"] == 1.0
        assert result["bleu_2_precision"] == 1.0
        assert result["brevity_penalty"] == 1.0

    def test_no_overlap_zero(self):
        a = ["bir", "iki"]
        b = ["üç", "dört"]
        assert bleu_n(a, b, 1) == 0.0

    def test_brevity_penalty_short_candidate(self):
        # candidate < reference → BP < 1
        bp = brevity_penalty(["a"], ["a", "b", "c", "d"])
        assert 0 < bp < 1

    def test_brevity_penalty_equal_or_longer(self):
        assert brevity_penalty(["a", "b"], ["a", "b"]) == 1.0
        assert brevity_penalty(["a", "b", "c"], ["a", "b"]) == 1.0

    def test_partial_overlap(self):
        a = ["algoritma", "bir", "yöntem"]
        b = ["algoritma", "adım", "adım", "yöntem"]
        p1 = bleu_n(a, b, 1)
        # cand'da 3 unigram: algoritma (match 1), bir (0), yöntem (1) → 2/3
        assert p1 == pytest.approx(2 / 3)


class TestROUGEL:
    def test_lcs_basic(self):
        assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_lcs_partial(self):
        # a, c → LCS=2
        assert _lcs_length(["a", "b", "c"], ["a", "x", "c"]) == 2

    def test_lcs_empty(self):
        assert _lcs_length([], ["a"]) == 0
        assert _lcs_length(["a"], []) == 0

    def test_rouge_l_identical(self):
        a = ["a", "b", "c"]
        r = rouge_l(a, a)
        assert r["rouge_l_p"] == 1.0
        assert r["rouge_l_r"] == 1.0
        assert r["rouge_l_f"] == pytest.approx(1.0)

    def test_rouge_l_no_overlap(self):
        r = rouge_l(["a", "b"], ["x", "y"])
        assert r["rouge_l_f"] == 0.0


def _item(ref: str) -> GoldenItem:
    return GoldenItem(
        id="t", question="q", reference_answer=ref,
        keywords=[], expected_sources=[], difficulty="easy", category="factual",
    )


class TestL3Evaluator:
    def test_identical_text_high_score(self):
        ev = L3LexicalEvaluator()
        text = "Algoritma adım adım uygulanan bir yöntemdir."
        r = ev.evaluate(_item(text), RAGOutput(answer=text))
        assert r.score >= 0.95

    def test_completely_different_text_low_score(self):
        # Tamamen farklı içerik, stopword örtüşmesi de minimum tutuldu
        ev = L3LexicalEvaluator()
        r = ev.evaluate(
            _item("Algoritma tanımıdır."),
            RAGOutput(answer="Quantum fizik karmaşık dalıdır."),
        )
        assert r.score < 0.2

    def test_paraphrase_partial_score(self):
        ev = L3LexicalEvaluator()
        ref = "Algoritma adım adım uygulanan bir yöntemdir."
        ans = "Algoritma sıralı adımlardan oluşan bir prosedürdür."
        r = ev.evaluate(_item(ref), RAGOutput(answer=ans))
        # Paraphrase → orta düzey skor beklenir, 0 değil 1 değil
        assert 0.05 < r.score < 0.8

    def test_empty_answer(self):
        ev = L3LexicalEvaluator()
        r = ev.evaluate(_item("ref"), RAGOutput(answer=""))
        assert r.score == 0.0
