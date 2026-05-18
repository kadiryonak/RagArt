"""Unit tests for memory strategies."""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from src.memory import (
    ConversationTurn,
    NoMemory,
    SlidingWindowMemory,
    SummaryBufferMemory,
    VectorRetrievalMemory,
)
from src.memory.base import format_turns


def _turns(*pairs) -> List[ConversationTurn]:
    """_turns(('user', 'q1'), ('assistant', 'a1'), ...) → list."""
    return [ConversationTurn(role=r, content=c) for r, c in pairs]


# ----- ConversationTurn -----


class TestConversationTurn:
    def test_from_dict_valid(self):
        t = ConversationTurn.from_dict({"role": "user", "content": "merhaba"})
        assert t.role == "user"
        assert t.content == "merhaba"

    def test_from_dict_invalid_role_normalized(self):
        t = ConversationTurn.from_dict({"role": "robot", "content": "x"})
        assert t.role == "user"  # default

    def test_from_dict_missing_content(self):
        t = ConversationTurn.from_dict({"role": "user"})
        assert t.content == ""

    def test_to_dict_roundtrip(self):
        t = ConversationTurn(role="assistant", content="cevap")
        assert ConversationTurn.from_dict(t.to_dict()) == t


class TestFormatTurns:
    def test_formats_turkish_labels(self):
        out = format_turns(_turns(("user", "soru"), ("assistant", "cevap")))
        assert "Kullanıcı: soru" in out
        assert "Asistan: cevap" in out

    def test_empty(self):
        assert format_turns([]) == ""


# ----- NoMemory -----


class TestNoMemory:
    def test_returns_empty_no_matter_what(self):
        m = NoMemory()
        assert m.apply([], "q") == ""
        assert m.apply(_turns(("user", "x"), ("assistant", "y")), "q") == ""

    def test_name(self):
        assert NoMemory.name == "none"


# ----- SlidingWindowMemory -----


class TestSlidingWindow:
    def test_empty_history(self):
        m = SlidingWindowMemory()
        assert m.apply([], "q") == ""

    def test_under_window_returns_all(self):
        m = SlidingWindowMemory(window_size=5)
        h = _turns(("user", "soru1"), ("assistant", "cevap1"))
        out = m.apply(h, "q")
        assert "Kullanıcı: soru1" in out
        assert "Asistan: cevap1" in out

    def test_over_window_trims_old(self):
        m = SlidingWindowMemory(window_size=2)
        # 4 user+assistant pair → 8 turns; window=2 → son 4 turn
        h = []
        for i in range(4):
            h += _turns(("user", f"q{i}"), ("assistant", f"a{i}"))
        out = m.apply(h, "current")
        # En eski turn'ler dışarda kalır
        assert "q0" not in out
        assert "a0" not in out
        # Son 2 çift içerde
        assert "q2" in out
        assert "q3" in out
        assert "a3" in out

    def test_window_size_must_be_positive(self):
        with pytest.raises(ValueError):
            SlidingWindowMemory(window_size=0)


# ----- SummaryBufferMemory -----


class TestSummaryBuffer:
    def test_short_history_no_summarize_call(self):
        llm = MagicMock()
        m = SummaryBufferMemory(llm=llm, keep_recent=2, summarize_threshold=4)
        h = _turns(("user", "q1"), ("assistant", "a1"))  # 2 turn < 4 threshold
        out = m.apply(h, "current")
        llm.generate.assert_not_called()
        assert "q1" in out

    def test_long_history_calls_llm_once(self):
        llm = MagicMock()
        llm.generate.return_value = "Konuşma algoritma hakkındaydı."
        m = SummaryBufferMemory(llm=llm, keep_recent=2, summarize_threshold=4)
        # 6 turns (3 pair) > 4 threshold
        h = []
        for i in range(3):
            h += _turns(("user", f"q{i}"), ("assistant", f"a{i}"))

        out = m.apply(h, "current")
        llm.generate.assert_called_once()
        assert "Önceki konuşma özeti:" in out
        assert "Konuşma algoritma hakkındaydı." in out
        # Son 2 turn ham görünmeli
        assert "q2" in out
        assert "a2" in out
        # Eski turn'ler ham YOK (sadece özette)
        assert "Kullanıcı: q0" not in out

    def test_empty_history(self):
        llm = MagicMock()
        m = SummaryBufferMemory(llm=llm)
        assert m.apply([], "q") == ""
        llm.generate.assert_not_called()

    def test_invalid_config(self):
        llm = MagicMock()
        with pytest.raises(ValueError):
            SummaryBufferMemory(llm=llm, keep_recent=0)
        with pytest.raises(ValueError):
            SummaryBufferMemory(llm=llm, keep_recent=5, summarize_threshold=2)


# ----- VectorRetrievalMemory -----


class TestVectorRetrieval:
    def test_empty_history(self):
        embed = MagicMock(return_value=[1, 0, 0])
        m = VectorRetrievalMemory(embed_fn=embed)
        assert m.apply([], "q") == ""
        embed.assert_not_called()

    def test_short_history_returns_all_no_embed(self):
        # top_k=2 → top_k*2=4 turn threshold
        embed = MagicMock()
        m = VectorRetrievalMemory(embed_fn=embed, top_k=2)
        h = _turns(("user", "x"), ("assistant", "y"))  # 2 turn, kısa
        out = m.apply(h, "q")
        assert "x" in out
        assert "y" in out
        # Kısa history'de embedding hiç yapılmaz
        embed.assert_not_called()

    def test_picks_most_similar_turns(self):
        # Embedding: kelime bazlı fake
        def fake_embed(text: str):
            return [
                text.lower().count("algoritma"),
                text.lower().count("python"),
                text.lower().count("yapay"),
                1.0,  # bias
            ]

        m = VectorRetrievalMemory(embed_fn=fake_embed, top_k=1)
        h = _turns(
            ("user", "algoritma nedir"),       # algoritma=1
            ("assistant", "algoritma ..."),     # algoritma=1
            ("user", "python ne"),               # python=1
            ("assistant", "python yorumlamalı"), # python=1
            ("user", "yapay zeka"),              # yapay=1
            ("assistant", "yapay zeka makine ..."), # yapay=1
        )
        # Sorgu: algoritma içerikli → algoritma turn'leri seçilmeli
        out = m.apply(h, "algoritma örnekleri")
        assert "algoritma nedir" in out
        # Python turn'leri seçilmeyebilir; en azından yapay olmamalı
        assert "yapay" not in out.lower()

    def test_embed_failure_falls_back_to_recent(self):
        def failing_embed(text: str):
            raise RuntimeError("embedder down")

        m = VectorRetrievalMemory(embed_fn=failing_embed, top_k=1)
        h = _turns(
            ("user", "old"), ("assistant", "old-a"),
            ("user", "mid"), ("assistant", "mid-a"),
            ("user", "recent"), ("assistant", "recent-a"),
        )
        out = m.apply(h, "q")
        # Recent kısmı korunmalı
        assert "recent" in out

    def test_invalid_top_k(self):
        with pytest.raises(ValueError):
            VectorRetrievalMemory(embed_fn=lambda t: [0.0], top_k=0)
