"""Unit tests for src/vector_stores — base factory, Chroma + Qdrant adapters."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from src.vector_stores.base import (
    BaseVectorStore,
    VectorSearchResult,
    VectorStoreFactory,
)


# ── VectorSearchResult ─────────────────────────────────────────────────


class TestVectorSearchResult:
    def test_defaults_score_zero(self):
        r = VectorSearchResult(page_content="x", metadata={})
        assert r.score == 0.0

    def test_carries_fields(self):
        r = VectorSearchResult(page_content="hi", metadata={"a": 1}, score=0.7)
        assert r.page_content == "hi"
        assert r.metadata == {"a": 1}
        assert r.score == 0.7


# ── VectorStoreFactory ─────────────────────────────────────────────────


class _Dummy(BaseVectorStore):
    name = "dummy"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def upsert_documents(self, documents):  # pragma: no cover - trivial
        pass

    def similarity_search(self, query, k=5):  # pragma: no cover
        return []

    def similarity_search_with_score(self, query, k=5):  # pragma: no cover
        return []

    def count(self):
        return self.kwargs.get("n", 0)

    def delete_collection(self):  # pragma: no cover
        pass


class TestVectorStoreFactory:
    def test_register_and_create(self):
        VectorStoreFactory.register("dummy", _Dummy, label="Dummy", desc="test")
        inst = VectorStoreFactory.create("dummy", n=3)
        assert isinstance(inst, _Dummy)
        assert inst.kwargs == {"n": 3}

    def test_is_available(self):
        VectorStoreFactory.register("dummy2", _Dummy, label="D2", desc="d")
        assert VectorStoreFactory.is_available("dummy2") is True
        assert VectorStoreFactory.is_available("nope") is False

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown vector store"):
            VectorStoreFactory.create("does-not-exist")

    def test_available_returns_info_dicts(self):
        VectorStoreFactory.register("dummy3", _Dummy, label="Label3", desc="Desc3")
        infos = VectorStoreFactory.available()
        by_id = {i["id"]: i for i in infos}
        assert by_id["dummy3"]["label"] == "Label3"
        assert by_id["dummy3"]["desc"] == "Desc3"


# ── BaseVectorStore.is_empty ───────────────────────────────────────────


class TestBaseIsEmpty:
    def test_empty_when_count_zero(self):
        assert _Dummy(n=0).is_empty() is True

    def test_not_empty_when_count_positive(self):
        assert _Dummy(n=5).is_empty() is False

    def test_is_empty_swallows_errors(self):
        class Broken(_Dummy):
            def count(self):
                raise RuntimeError("db down")
        assert Broken().is_empty() is True


# ── ChromaVectorStore (mocked chromadb client) ─────────────────────────


@pytest.fixture
def chroma_store(monkeypatch):
    import chromadb
    from src.vector_stores import chroma_store as mod

    fake_client = MagicMock()
    # Fresh store: no existing collections → _open_existing returns None
    fake_client.list_collections.return_value = []
    monkeypatch.setattr(chromadb, "PersistentClient", lambda path: fake_client)

    store = mod.ChromaVectorStore(
        collection_name="ws",
        persist_path="/tmp/x",
        embedding_function=MagicMock(),
    )
    return store, fake_client


class TestChromaVectorStore:
    def test_name(self):
        from src.vector_stores.chroma_store import ChromaVectorStore
        assert ChromaVectorStore.name == "chroma"

    def test_fresh_store_has_no_langchain_store(self, chroma_store):
        store, _ = chroma_store
        assert store.langchain_store is None

    def test_search_without_store_returns_empty(self, chroma_store):
        store, _ = chroma_store
        assert store.similarity_search("q") == []
        assert store.similarity_search_with_score("q") == []

    def test_count_reads_collection(self, chroma_store):
        store, client = chroma_store
        col = MagicMock()
        col.count.return_value = 42
        client.get_collection.return_value = col
        assert store.count() == 42

    def test_count_returns_zero_on_error(self, chroma_store):
        store, client = chroma_store
        client.get_collection.side_effect = RuntimeError("missing")
        assert store.count() == 0

    def test_delete_collection_clears_store(self, chroma_store):
        store, client = chroma_store
        store.delete_collection()
        client.delete_collection.assert_called_once_with("ws")
        assert store.langchain_store is None

    def test_delete_collection_swallows_errors(self, chroma_store):
        store, client = chroma_store
        client.delete_collection.side_effect = RuntimeError("nope")
        # Must not raise
        store.delete_collection()


# ── QdrantVectorStore (fake module injection) ──────────────────────────


@pytest.fixture
def fake_qdrant(monkeypatch):
    """Inject a minimal fake qdrant_client package so the adapter imports."""
    captured = {}

    class FakeQdrantClient:
        def __init__(self, path=None):
            captured["path"] = path
            self._collections = {}
            self.upserted = []

        def get_collections(self):
            cols = [types.SimpleNamespace(name=n) for n in self._collections]
            return types.SimpleNamespace(collections=cols)

        def create_collection(self, collection_name, vectors_config):
            self._collections[collection_name] = {"count": 0, "cfg": vectors_config}

        def delete_collection(self, collection_name):
            self._collections.pop(collection_name, None)

        def upsert(self, collection_name, points):
            self.upserted.extend(points)
            self._collections.setdefault(collection_name, {"count": 0})
            self._collections[collection_name]["count"] = len(points)

        def get_collection(self, collection_name):
            n = self._collections.get(collection_name, {}).get("count", 0)
            return types.SimpleNamespace(points_count=n)

        def search(self, collection_name, query_vector, limit):
            return captured.get("search_results", [])

    class FakeVectorParams:
        def __init__(self, size, distance):
            self.size = size
            self.distance = distance

    class FakeDistance:
        COSINE = "Cosine"

    class FakePointStruct:
        def __init__(self, id, vector, payload):
            self.id = id
            self.vector = vector
            self.payload = payload

    root = types.ModuleType("qdrant_client")
    root.QdrantClient = FakeQdrantClient
    http = types.ModuleType("qdrant_client.http")
    models = types.ModuleType("qdrant_client.http.models")
    models.Distance = FakeDistance
    models.VectorParams = FakeVectorParams
    models.PointStruct = FakePointStruct
    http.models = models
    root.http = http

    monkeypatch.setitem(sys.modules, "qdrant_client", root)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", http)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", models)
    return captured


def _make_qdrant(fake_qdrant):
    from src.vector_stores.qdrant_store import QdrantVectorStore

    emb = MagicMock()
    emb.embed_query.return_value = [0.1, 0.2, 0.3]
    emb.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    return QdrantVectorStore(
        collection_name="ws",
        persist_path="/tmp/q",
        embedding_function=emb,
    ), emb


class TestQdrantConstructionGuard:
    def test_missing_client_raises_runtime_error(self):
        # qdrant_client is genuinely absent in this environment
        from src.vector_stores.qdrant_store import QdrantVectorStore
        with pytest.raises(RuntimeError, match="qdrant-client"):
            QdrantVectorStore(
                collection_name="ws",
                persist_path="/tmp/q",
                embedding_function=MagicMock(),
            )


class TestQdrantVectorStore:
    def test_name(self):
        from src.vector_stores.qdrant_store import QdrantVectorStore
        assert QdrantVectorStore.name == "qdrant"

    def test_creates_collection_on_init(self, fake_qdrant):
        store, _ = _make_qdrant(fake_qdrant)
        # Collection auto-created with embedding dimension
        assert store._client._collections["ws"]["cfg"].size == 3

    def test_empty_search_returns_empty(self, fake_qdrant):
        store, _ = _make_qdrant(fake_qdrant)
        assert store.similarity_search("q") == []
        assert store.similarity_search_with_score("q") == []

    def test_upsert_then_count(self, fake_qdrant):
        store, _ = _make_qdrant(fake_qdrant)
        doc = types.SimpleNamespace(page_content="hello", metadata={"source": "a"})
        store.upsert_documents([doc])
        assert store.count() == 1

    def test_similarity_search_maps_payload(self, fake_qdrant):
        store, _ = _make_qdrant(fake_qdrant)
        doc = types.SimpleNamespace(page_content="x", metadata={})
        store.upsert_documents([doc])
        fake_qdrant["search_results"] = [
            types.SimpleNamespace(
                payload={"page_content": "found", "metadata": {"source": "a"}},
                score=0.9,
            )
        ]
        out = store.similarity_search("q", k=1)
        assert len(out) == 1
        assert out[0].page_content == "found"
        assert out[0].score == 0.9

    def test_similarity_search_with_score_converts_distance(self, fake_qdrant):
        store, _ = _make_qdrant(fake_qdrant)
        doc = types.SimpleNamespace(page_content="x", metadata={})
        store.upsert_documents([doc])
        # cosine score 1.0 (identical) → distance 0.0
        fake_qdrant["search_results"] = [
            types.SimpleNamespace(payload={"page_content": "p", "metadata": {}}, score=1.0)
        ]
        out = store.similarity_search_with_score("q", k=1)
        assert len(out) == 1
        _doc, distance = out[0]
        assert distance == pytest.approx(0.0)

    def test_delete_collection(self, fake_qdrant):
        store, _ = _make_qdrant(fake_qdrant)
        store.delete_collection()
        assert "ws" not in store._client._collections
        assert store.count() == 0
