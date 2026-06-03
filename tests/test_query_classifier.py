"""Unit tests for src/query_classifier.py — adaptive retrieval classifier."""

from __future__ import annotations

import pytest

from src.query_classifier import (
    ADAPTIVE_CONFIGS,
    AdaptiveConfig,
    QueryClassifier,
    QueryComplexity,
    greeting_response,
)


class TestClassifyGreeting:
    @pytest.mark.parametrize("q", ["merhaba", "Selam", "HELLO", "günaydın", "naber"])
    def test_greetings(self, q):
        assert QueryClassifier.classify(q) == QueryComplexity.GREETING

    def test_greeting_with_surrounding_text_is_not_greeting(self):
        # Anchored regex → only pure greetings match
        assert QueryClassifier.classify("merhaba algoritma nedir") != QueryComplexity.GREETING


class TestClassifySimple:
    def test_one_word(self):
        assert QueryClassifier.classify("algoritma") == QueryComplexity.SIMPLE

    def test_short_question(self):
        assert QueryClassifier.classify("Python nedir") == QueryComplexity.SIMPLE

    def test_short_but_complex_signal_not_simple(self):
        # "karşılaştır" is a complex signal even if short
        assert QueryClassifier.classify("X Y karşılaştır") != QueryComplexity.SIMPLE


class TestClassifyComplex:
    def test_comparison_keyword(self):
        q = "React ile Vue arasındaki fark nedir ve hangisi daha iyidir?"
        assert QueryClassifier.classify(q) == QueryComplexity.COMPLEX

    def test_multi_comma_long_sentence(self):
        q = ("Bana algoritmaların tarihçesini, kullanım alanlarını, "
             "avantajlarını ve dezavantajlarını detaylı açıkla")
        assert QueryClassifier.classify(q) == QueryComplexity.COMPLEX

    def test_compare_english(self):
        assert QueryClassifier.classify("compare Python vs Java performance") == \
            QueryComplexity.COMPLEX


class TestClassifyModerate:
    def test_default_moderate(self):
        q = "Veri yapıları konusunu bana biraz anlatabilir misin lütfen"
        assert QueryClassifier.classify(q) == QueryComplexity.MODERATE


class TestGetConfig:
    def test_each_complexity_has_config(self):
        for c in QueryComplexity:
            cfg = QueryClassifier.get_config(c)
            assert isinstance(cfg, AdaptiveConfig)

    def test_greeting_skips_retrieval(self):
        cfg = QueryClassifier.get_config(QueryComplexity.GREETING)
        assert cfg.skip_retrieval is True
        assert cfg.k == 0

    def test_simple_uses_hybrid_with_headroom(self):
        # Regression: SIMPLE was dense-only k=2, which starved short proper-noun
        # queries ("Kadir kimdir?") of BM25 exact-match recall. It now uses
        # hybrid (retrieval_strategy=None → hybrid if available) with k>=4.
        cfg = QueryClassifier.get_config(QueryComplexity.SIMPLE)
        assert cfg.k >= 4
        assert cfg.retrieval_strategy is None  # None → hybrid
        assert cfg.rerank is False

    def test_complex_reranks_with_high_k(self):
        cfg = QueryClassifier.get_config(QueryComplexity.COMPLEX)
        assert cfg.k == 10
        assert cfg.rerank is True

    def test_config_table_covers_all_levels(self):
        assert set(ADAPTIVE_CONFIGS.keys()) == set(QueryComplexity)


class TestGreetingResponse:
    def test_returns_nonempty_turkish_greeting(self):
        out = greeting_response("merhaba")
        assert "Merhaba" in out
        assert len(out) > 10
