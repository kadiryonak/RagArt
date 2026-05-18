"""Regression test for the reranker-cache stale-vector_store bug.

Reproduces the production failure mode:
    1. RAG system builds retrievers → reranker is cached
    2. Reindex runs → vector_store is replaced
    3. Cached reranker still wraps the OLD vector_store
    4. Next query hits a "Collection [UUID] does not exist" from ChromaDB

Fix verified here:
    _build_retrievers must clear _reranker_cache so that the next
    rerank=True query gets a freshly-wrapped reranker over the new
    vector_store.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Skip if we can't import the system (would happen on minimal CI without deps)
pytest.importorskip("langchain_chroma")


class _FakeVectorStore:
    """Stand-in for a Chroma instance — just needs an identity for assertion."""

    def __init__(self, tag: str):
        self.tag = tag

    def similarity_search_with_score(self, query, k):
        return []


class _FakeEmbeddings:
    def embed_query(self, text):
        return [0.0] * 4

    def embed_documents(self, texts):
        return [[0.0] * 4 for _ in texts]


def _make_rag_with_stub(monkeypatch):
    """Build a TurkishRAGSystem with the heavy externals stubbed."""
    from src.rag_system import TurkishRAGSystem
    from src.embeddings import EmbeddingManager

    # Don't load the real embedding model
    monkeypatch.setattr(
        EmbeddingManager,
        "embeddings",
        property(lambda self: _FakeEmbeddings()),
    )
    # Don't touch ChromaDB at all — chroma_client is unused for our path
    import chromadb
    monkeypatch.setattr(chromadb, "PersistentClient", lambda path: MagicMock())

    rag = TurkishRAGSystem(
        data_folder=".",
        model_type="local",
        api_key=None,
        chroma_db_path="./does-not-exist",
    )
    return rag


class TestRerankerCacheInvalidation:
    def test_cache_cleared_on_rebuild(self, monkeypatch):
        from src.rag_system import TurkishRAGSystem

        rag = _make_rag_with_stub(monkeypatch)
        # Inject docs + an initial vector_store
        from langchain_core.documents import Document
        docs = [Document(page_content="a" * 30, metadata={"source": "x.json", "item_index": 0})]
        rag.vector_store = _FakeVectorStore("v1")
        rag._build_retrievers(docs)

        # Force a reranker to be cached by selecting one
        # (We don't actually load the cross-encoder; the cache is keyed by
        # base retriever name, so just put a sentinel in.)
        rag._reranker_cache["hybrid"] = "stale-reranker-instance"
        assert "hybrid" in rag._reranker_cache

        # Now simulate a reindex: new vector_store + rebuild
        rag.vector_store = _FakeVectorStore("v2")
        rag._build_retrievers(docs)

        # Cache MUST be empty now — that's the fix
        assert rag._reranker_cache == {}, (
            "Stale reranker still cached after _build_retrievers — the very "
            "bug this test guards against."
        )

    def test_stale_collection_error_surfaces_clean_message(self, monkeypatch):
        """When the retriever raises a chroma collection error, ask() returns
        the friendly STALE_INDEX message rather than a traceback string."""
        from src.rag_system import TurkishRAGSystem

        rag = _make_rag_with_stub(monkeypatch)

        # Stub the retrieval path: pretend the only retriever raises with
        # the chromadb "Collection [...] does not exist" text.
        class StaleRetriever:
            name = "dense"
            def retrieve(self, query, k=5):
                raise RuntimeError(
                    "Error getting collection: Collection [abc-123] does not exist."
                )

        rag._dense_retriever = StaleRetriever()
        rag._hybrid_retriever = None
        rag._sparse_retriever = None
        rag.vector_store = _FakeVectorStore("v-stale")

        # Bypass relevance threshold by giving a non-empty fake response;
        # actually search() will raise STALE_INDEX before reaching that.
        result = rag.ask("herhangi bir soru")
        assert result["source"] == "stale_index"
        assert "Yeniden İndeksle" in result["answer"]
