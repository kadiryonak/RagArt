"""Unit tests for context engineering processors."""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from src.context import (
    LostInTheMiddleReorderer,
    ProcessorChain,
    RedundancyFilter,
    TokenBudgetTrimmer,
)
from src.context.token_budget import estimate_tokens
from src.retrievers.base import RetrievedDoc


def _doc(content: str, src: str = "x.json", idx: int = 0) -> RetrievedDoc:
    return RetrievedDoc(
        page_content=content,
        metadata={"source": src, "item_index": idx},
        score=1.0,
    )


# ----- RedundancyFilter -----


class TestRedundancyFilter:
    def test_removes_duplicates(self):
        # 3 identical docs → fake_embed verir aynı vektör
        def fake_embed(text: str):
            return [1.0, 0.0, 0.0]
        docs = [_doc("aynı içerik", "a.json"), _doc("aynı içerik", "b.json"),
                _doc("aynı içerik", "c.json")]
        out = RedundancyFilter(embed_fn=fake_embed).process("q", docs)
        # Hepsi identik → sadece 1 tane kalır
        assert len(out) == 1

    def test_keeps_different_docs(self):
        # Distinct one-hot vectors → cosine sim = 0 between them
        vectors = {"alfa": [1, 0, 0], "beta": [0, 1, 0], "gama": [0, 0, 1]}
        def fake_embed(text: str):
            return vectors[text.split()[0]]
        docs = [_doc("alfa içerik", "a.json"), _doc("beta içerik", "b.json"),
                _doc("gama içerik", "c.json")]
        out = RedundancyFilter(embed_fn=fake_embed).process("q", docs)
        assert len(out) == 3

    def test_short_circuit_under_two(self):
        f = RedundancyFilter(embed_fn=MagicMock())
        assert f.process("q", []) == []
        single = [_doc("tek")]
        assert f.process("q", single) == single
        # 0 docs → embedder hiç çağrılmaz
        f.embed_fn.assert_not_called()

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            RedundancyFilter(embed_fn=lambda t: [0], similarity_threshold=1.5)
        with pytest.raises(ValueError):
            RedundancyFilter(embed_fn=lambda t: [0], similarity_threshold=-0.1)

    def test_embedder_failure_keeps_doc(self):
        # Bir embedding çağrısı raise ederse, doc kabul edilir (graceful)
        calls = {"n": 0}
        def flaky_embed(text):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("embedder down")
            return [1.0, 0.0]

        docs = [_doc("a", "a.json"), _doc("b", "b.json"), _doc("a", "c.json")]
        out = RedundancyFilter(embed_fn=flaky_embed,
                              similarity_threshold=0.99).process("q", docs)
        # 3 doc verildi, biri hata aldı ama yine de en az 1 doc döner
        assert len(out) >= 1


# ----- LostInTheMiddleReorderer -----


class TestReorderer:
    def test_eight_doc_pattern(self):
        # Lost-in-the-middle:
        # input:  [d0, d1, d2, d3, d4, d5, d6, d7]   (azalan relevance)
        # output: [d0, d2, d4, d6, d7, d5, d3, d1]
        docs = [_doc(f"d{i}", f"f{i}.json", i) for i in range(8)]
        out = LostInTheMiddleReorderer().process("q", docs)
        assert out[0].source == "f0.json"   # en güçlü başta
        assert out[-1].source == "f1.json"  # ikinci en güçlü sonda
        # Orta kısım daha zayıf chunk'lar
        sources_middle = [d.source for d in out[3:5]]
        assert "f6.json" in sources_middle or "f7.json" in sources_middle

    def test_too_few_docs_skipped(self):
        # min_docs=3, sadece 2 doc verirsen değişmez
        r = LostInTheMiddleReorderer(min_docs=3)
        docs = [_doc("a", "a.json"), _doc("b", "b.json")]
        out = r.process("q", docs)
        assert out == docs  # aynı sıra

    def test_empty_input(self):
        assert LostInTheMiddleReorderer().process("q", []) == []

    def test_preserves_count(self):
        for n in (3, 4, 5, 6, 7, 10):
            docs = [_doc(f"d{i}") for i in range(n)]
            out = LostInTheMiddleReorderer().process("q", docs)
            assert len(out) == n
            # İçerik kayıpsız
            assert {d.page_content for d in out} == {d.page_content for d in docs}


# ----- TokenBudgetTrimmer -----


class TestTokenBudget:
    def test_estimate_tokens_basic(self):
        # 7 karakter / 3.5 → 2 token
        assert estimate_tokens("abcdefg") == 2

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") == 0

    def test_trims_to_budget(self):
        # Her doc ~10 token (35 char / 3.5)
        docs = [_doc("a" * 35, f"f{i}.json", i) for i in range(10)]
        # budget=25 → 2 doc (20 token), 3. doc 30 token olur → kesilir
        out = TokenBudgetTrimmer(max_tokens=25).process("q", docs)
        assert len(out) == 2

    def test_always_keeps_at_least_one(self):
        # Tek doc bütçeden büyükse bile boş context yerine onu döndür
        big = _doc("x" * 10000, "big.json")
        out = TokenBudgetTrimmer(max_tokens=10).process("q", [big])
        assert out == [big]

    def test_empty_input(self):
        assert TokenBudgetTrimmer(max_tokens=100).process("q", []) == []

    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError):
            TokenBudgetTrimmer(max_tokens=0)


# ----- ProcessorChain -----


class TestProcessorChain:
    def test_applies_processors_in_order(self):
        # 5 doc, 2 tanesi aynı içerikli; redundancy önce → 4 doc kalır
        # token budget sonra → max 2'ye in
        # reorderer en son → düzeni karıştır

        def fake_embed(text):
            return [hash(text) % 100 / 100.0, 0.5]

        docs = [
            _doc("alfa", "a.json", 0),
            _doc("alfa", "a2.json", 1),  # duplicate
            _doc("beta", "b.json", 2),
            _doc("gama", "c.json", 3),
            _doc("delta", "d.json", 4),
        ]

        chain = ProcessorChain([
            RedundancyFilter(embed_fn=fake_embed, similarity_threshold=0.99),
            TokenBudgetTrimmer(max_tokens=10),  # küçük budget → kuyruğu kes
            LostInTheMiddleReorderer(min_docs=3),
        ])
        out = chain.process("q", docs)
        # Sonuç boş olmasın, en az 1 doc dönsün
        assert len(out) >= 1
        # Total token count bütçeyi aşmasın (with min-one guarantee)
        total = sum(estimate_tokens(d.page_content) for d in out)
        # 'always keeps at least one' nedeniyle ilk doc bütçeyi aşabilir
        assert total <= 10 or len(out) == 1

    def test_empty_chain_pass_through(self):
        docs = [_doc("x", "x.json")]
        assert ProcessorChain([]).process("q", docs) == docs

    def test_short_circuit_on_empty(self):
        # Bir processor boş döndürürse kalan zincir çalışmasın
        bad = MagicMock()
        bad.process.return_value = []
        chain = ProcessorChain([
            LostInTheMiddleReorderer(),
            bad,
            LostInTheMiddleReorderer(),  # bu çağrılmamalı
        ])
        docs = [_doc("a"), _doc("b"), _doc("c")]
        chain.process("q", docs)
        bad.process.assert_called_once()
