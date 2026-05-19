"""Tests for the hallucination-safe fallback default.

Regression target: when retrieval finds nothing relevant, the previous
code would let the cloud LLM produce an unconstrained "general knowledge"
answer. For private/proprietary data this leaked fabricated content into
the UI (the live bug: asking about a private CV returned a fake bio of
someone unrelated).

The fix: ``allow_general_knowledge`` defaults to False; the LLM is NOT
called and the user gets a clear "no info" message instead.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.llm_providers import LocalProvider


@pytest.fixture
def fake_rag(monkeypatch):
    """Build a TurkishRAGSystem with heavy externals stubbed."""
    from src.rag_system import TurkishRAGSystem
    from src.embeddings import EmbeddingManager

    monkeypatch.setattr(
        EmbeddingManager, "embeddings",
        property(lambda self: MagicMock(embed_query=lambda t: [0.0] * 4))
    )
    import chromadb
    monkeypatch.setattr(chromadb, "PersistentClient", lambda path: MagicMock())

    rag = TurkishRAGSystem(
        data_folder=".", model_type="local",
        api_key=None, chroma_db_path="./nope",
    )
    return rag


class TestSafeDefault:
    def test_default_does_not_call_llm(self, fake_rag):
        """The whole point of this fix: NO LLM CALL when defaulting."""
        fake_provider = MagicMock(spec=["generate", "generate_general"])
        fake_rag.llm_provider = fake_provider

        result = fake_rag._fallback_response("Kadir Yönak kimdir?", 0.05)

        # Critical assertion: the LLM was NOT consulted
        fake_provider.generate_general.assert_not_called()
        fake_provider.generate.assert_not_called()

        assert result["source"] == "insufficient_data"
        # User-facing message must explain the situation in Turkish
        assert "yeterli detay bulunamadı" in result["answer"]
        # And give actionable next steps
        assert "Yeniden İndeksle" in result["answer"]

    def test_opt_in_does_call_llm(self, fake_rag):
        """When the user explicitly opts in, the LLM is called."""
        fake_provider = MagicMock(spec=["generate", "generate_general"])
        fake_provider.generate_general.return_value = "uydurulmuş yanıt"
        fake_rag.llm_provider = fake_provider

        result = fake_rag._fallback_response(
            "Kadir Yönak kimdir?", 0.05,
            allow_general_knowledge=True,
        )

        fake_provider.generate_general.assert_called_once()
        assert result["source"] == "general_knowledge_fallback"
        # And the warning must be visible to the user
        assert "halüsinasyon" in result["answer"].lower()

    def test_local_provider_opt_in_falls_back_to_safe_message(self, fake_rag):
        """LocalProvider has no real general-knowledge mode; opt-in is still
        safe (no traceback, no garbage answer)."""
        fake_rag.llm_provider = LocalProvider()
        result = fake_rag._fallback_response(
            "Algoritma", 0.05, allow_general_knowledge=True,
        )
        # No real LLM, but the system gives a clean refusal
        assert result["source"] == "insufficient_data"
        assert "0.05" in result["answer"] or "yeterli detay" in result["answer"]
