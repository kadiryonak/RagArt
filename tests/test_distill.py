"""Tests for optional pre-chunking LLM distillation (src/distill.py).

The contract that matters: distillation must NEVER lose a document or make it
bigger. Every failure mode falls back to the original text.
"""

from __future__ import annotations

import pytest

from src.distill import (
    DISTILL_MIN_CHARS,
    distill_text,
    distill_documents,
)

try:
    from langchain_core.documents import Document
except ImportError:  # older langchain
    from langchain.schema import Document


class _LLM:
    """Fake provider: returns a canned string and records prompts/calls."""

    def __init__(self, reply="kısa öz", raises=False):
        self.reply = reply
        self.raises = raises
        self.calls = 0
        self.last_params = None

    def generate(self, prompt, **params):
        self.calls += 1
        self.last_params = params
        if self.raises:
            raise RuntimeError("provider down")
        return self.reply


# A long enough input that distillation is attempted (> DISTILL_MIN_CHARS).
_LONG = "Optimizasyon algoritmaları ve genetik yöntemler hakkında. " * 20
# A valid distillation: clearly shorter than _LONG but above the over-
# compression floor (≥15% of the original), so distill_text accepts it.
_VALID = "Damıtılmış öz: optimizasyon ve genetik algoritma temel bilgileri. " * 5


class TestDistillText:
    def test_shortens_long_text(self):
        llm = _LLM(reply=_VALID)
        out = distill_text(_LONG, llm)
        assert llm.calls == 1
        assert out == _VALID.strip()
        assert len(out) < len(_LONG)

    def test_short_text_skipped_without_llm_call(self):
        llm = _LLM()
        short = "Kısa metin."
        assert len(short) < DISTILL_MIN_CHARS
        assert distill_text(short, llm) == short
        assert llm.calls == 0  # not worth a round-trip

    def test_empty_result_falls_back_to_original(self):
        assert distill_text(_LONG, _LLM(reply="   ")) == _LONG

    def test_error_string_result_falls_back(self):
        assert distill_text(_LONG, _LLM(reply="API error: 500")) == _LONG

    def test_non_shrinking_result_falls_back(self):
        # Result longer than input → reject (distillation must shrink).
        assert distill_text(_LONG, _LLM(reply=_LONG + " extra")) == _LONG

    def test_over_compression_falls_back(self):
        # One-line "summary" of a long page drops info → keep original.
        assert distill_text(_LONG, _LLM(reply="Özet.")) == _LONG

    def test_exception_falls_back(self):
        assert distill_text(_LONG, _LLM(raises=True)) == _LONG

    def test_passes_llm_params_through(self):
        llm = _LLM(reply=_VALID)
        distill_text(_LONG, llm, temperature=0.1)
        assert llm.last_params.get("temperature") == 0.1


class TestDistillDocuments:
    def test_preserves_metadata_and_tags_changed(self):
        llm = _LLM(reply=_VALID)
        docs = [
            Document(page_content=_LONG, metadata={"source": "a.json", "page": 1}),
            Document(page_content="kısa", metadata={"source": "b.json"}),
        ]
        out = distill_documents(docs, llm, temperature=0.1)

        assert len(out) == 2
        # First was distilled → tagged + shorter; metadata preserved.
        assert out[0].metadata["source"] == "a.json"
        assert out[0].metadata["page"] == 1
        assert out[0].metadata.get("distilled") is True
        assert len(out[0].page_content) < len(_LONG)
        # Second too short → untouched, not tagged.
        assert out[1].page_content == "kısa"
        assert "distilled" not in out[1].metadata

    def test_empty_list_is_noop(self):
        assert distill_documents([], _LLM()) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
