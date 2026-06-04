"""Unit tests for the SemanticTextSplitter in src/embeddings.py."""

from __future__ import annotations

from src.embeddings import SemanticTextSplitter, _cosine_sim, _split_sentences

try:
    from langchain_core.documents import Document
except ImportError:  # older langchain
    from langchain.schema import Document


class TestSplitSentences:
    def test_splits_on_punctuation(self):
        out = _split_sentences("Birinci cümle. İkinci cümle! Üçüncü mü?")
        assert out == ["Birinci cümle.", "İkinci cümle!", "Üçüncü mü?"]

    def test_strips_and_drops_empty(self):
        assert _split_sentences("   ") == []

    def test_single_sentence(self):
        assert _split_sentences("Tek cümle var") == ["Tek cümle var"]


class TestCosineSim:
    def test_identical_vectors_is_one(self):
        assert _cosine_sim([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_is_zero(self):
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_mismatched_length_is_zero(self):
        assert _cosine_sim([1.0], [1.0, 2.0]) == 0.0

    def test_empty_is_zero(self):
        assert _cosine_sim([], []) == 0.0

    def test_zero_vector_is_zero(self):
        assert _cosine_sim([0.0, 0.0], [1.0, 1.0]) == 0.0


def _embed_factory(mapping):
    """Return an embed_fn yielding a fixed vector per sentence substring."""
    def embed(sentence):
        for key, vec in mapping.items():
            if key in sentence:
                return vec
        return [0.0, 0.0]
    return embed


class TestSemanticTextSplitter:
    def test_empty_text_returns_empty(self):
        s = SemanticTextSplitter(embed_fn=lambda t: [1.0])
        assert s.split_text("") == []

    def test_single_sentence_no_embedding_needed(self):
        calls = []
        s = SemanticTextSplitter(embed_fn=lambda t: calls.append(t) or [1.0])
        out = s.split_text("Tek cümle burada")
        assert out == ["Tek cümle burada"]
        assert calls == []  # short-circuit before embedding

    def test_similar_sentences_stay_together(self):
        # All sentences embed to the same vector → similarity 1.0 → one chunk
        s = SemanticTextSplitter(embed_fn=lambda t: [1.0, 0.0], similarity_threshold=0.5)
        out = s.split_text("Cümle bir. Cümle iki. Cümle üç.")
        assert len(out) == 1

    def test_dissimilar_sentences_split(self):
        embed = _embed_factory({"alfa": [1.0, 0.0], "beta": [0.0, 1.0]})
        s = SemanticTextSplitter(embed_fn=embed, similarity_threshold=0.5)
        out = s.split_text("alfa cümlesi. beta cümlesi.")
        assert len(out) == 2

    def test_max_chunk_size_forces_split(self):
        # Same vector (would merge) but tiny max size → must split
        s = SemanticTextSplitter(
            embed_fn=lambda t: [1.0, 0.0],
            max_chunk_size=5,
            similarity_threshold=0.0,
        )
        out = s.split_text("aaaaaa bbbbb. cccccc ddddd.")
        assert len(out) >= 2

    def test_embedding_failure_falls_back_to_single_chunk(self):
        def boom(_):
            raise RuntimeError("embed down")
        s = SemanticTextSplitter(embed_fn=boom)
        text = "Cümle bir. Cümle iki."
        assert s.split_text(text) == [text]

    def test_split_documents_adds_metadata(self):
        embed = _embed_factory({"alfa": [1.0, 0.0], "beta": [0.0, 1.0]})
        s = SemanticTextSplitter(embed_fn=embed, similarity_threshold=0.5)
        docs = [Document(page_content="alfa cümlesi. beta cümlesi.", metadata={"source": "x"})]
        out = s.split_documents(docs)
        assert len(out) == 2
        for i, d in enumerate(out):
            assert d.metadata["source"] == "x"
            assert d.metadata["chunk_index"] == i
            assert d.metadata["chunk_count"] == 2
            assert d.metadata["split_strategy"] == "semantic"
