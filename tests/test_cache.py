"""Unit tests for the cache layer.

Covers SQLiteCache (get/set/delete/clear/TTL), EmbeddingCache
(transparent caching of embed_query / embed_documents incl. partial
hits), ResponseCache (key stability + invalidation), and SemanticCache
(threshold-based similarity hits).
"""

from __future__ import annotations

import time

import pytest

from src.cache.sqlite_cache import SQLiteCache
from src.cache.embedding_cache import EmbeddingCache
from src.cache.response_cache import ResponseCache
from src.cache.semantic_cache import SemanticCache


# ----- SQLiteCache -----


class TestSQLiteCacheCRUD:
    def test_set_then_get(self, tmp_path):
        c = SQLiteCache(str(tmp_path / "c.db"))
        c.set("k", [1, 2, 3])
        assert c.get("k") == [1, 2, 3]

    def test_miss_returns_none(self, tmp_path):
        c = SQLiteCache(str(tmp_path / "c.db"))
        assert c.get("absent") is None

    def test_overwrite(self, tmp_path):
        c = SQLiteCache(str(tmp_path / "c.db"))
        c.set("k", "v1")
        c.set("k", "v2")
        assert c.get("k") == "v2"

    def test_delete(self, tmp_path):
        c = SQLiteCache(str(tmp_path / "c.db"))
        c.set("k", 1)
        assert c.delete("k") is True
        assert c.get("k") is None
        # Deleting again returns False
        assert c.delete("k") is False

    def test_clear(self, tmp_path):
        c = SQLiteCache(str(tmp_path / "c.db"))
        c.set("a", 1)
        c.set("b", 2)
        assert c.clear() == 2
        assert c.count() == 0

    def test_count_keys(self, tmp_path):
        c = SQLiteCache(str(tmp_path / "c.db"))
        for i in range(5):
            c.set(f"k{i}", i)
        assert c.count() == 5
        assert set(c.keys()) == {f"k{i}" for i in range(5)}

    def test_invalid_table_name_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            SQLiteCache(str(tmp_path / "x.db"), table="evil; DROP TABLE x")


class TestSQLiteCacheTTL:
    def test_unlimited_ttl_never_expires(self, tmp_path):
        c = SQLiteCache(str(tmp_path / "c.db"))
        c.set("k", 1, ttl=None)
        assert c.get("k") == 1

    def test_short_ttl_expires(self, tmp_path):
        c = SQLiteCache(str(tmp_path / "c.db"))
        c.set("k", 1, ttl=0.05)
        time.sleep(0.1)
        assert c.get("k") is None  # expired, lazy GC purges
        assert c.count() == 0

    def test_zero_ttl_treated_as_unlimited(self, tmp_path):
        # ttl=0 historically a "no cache" signal — but our impl treats
        # 0 the same as None (unlimited). Document via test.
        c = SQLiteCache(str(tmp_path / "c.db"))
        c.set("k", 1, ttl=0)
        assert c.get("k") == 1


class TestSQLiteCacheStats:
    def test_hit_miss_counters(self, tmp_path):
        c = SQLiteCache(str(tmp_path / "c.db"))
        c.get("miss1")
        c.set("k", 1)
        c.get("k")
        c.get("k")
        s = c.stats()
        assert s.hits == 2
        assert s.misses == 1
        assert s.sets == 1
        assert s.hit_rate == pytest.approx(2 / 3)


# ----- EmbeddingCache -----


class _FakeEmbedder:
    """Counts calls so we can verify cache hits skip the model."""
    def __init__(self):
        self.query_calls = 0
        self.doc_calls = 0

    def embed_query(self, text):
        self.query_calls += 1
        return [float(len(text)), float(sum(map(ord, text[:8])))]

    def embed_documents(self, texts):
        self.doc_calls += 1
        return [self.embed_query(t) for t in texts]


class TestEmbeddingCache:
    def test_first_call_hits_model(self, tmp_path):
        emb = _FakeEmbedder()
        cache = EmbeddingCache(str(tmp_path / "emb.db"), embedder=emb)
        v = cache.embed_query("hello")
        assert isinstance(v, list)
        assert emb.query_calls == 1

    def test_second_call_uses_cache(self, tmp_path):
        emb = _FakeEmbedder()
        cache = EmbeddingCache(str(tmp_path / "emb.db"), embedder=emb)
        cache.embed_query("hello")
        cache.embed_query("hello")  # should be a cache hit
        assert emb.query_calls == 1

    def test_different_text_misses_cache(self, tmp_path):
        emb = _FakeEmbedder()
        cache = EmbeddingCache(str(tmp_path / "emb.db"), embedder=emb)
        cache.embed_query("hello")
        cache.embed_query("world")
        assert emb.query_calls == 2

    def test_persists_across_instances(self, tmp_path):
        # Cache survives reconstruction
        emb1 = _FakeEmbedder()
        c1 = EmbeddingCache(str(tmp_path / "emb.db"), embedder=emb1)
        c1.embed_query("persistent")
        # Second instance over the same db
        emb2 = _FakeEmbedder()
        c2 = EmbeddingCache(str(tmp_path / "emb.db"), embedder=emb2)
        c2.embed_query("persistent")
        assert emb2.query_calls == 0  # served from cache

    def test_embed_documents_partial_hit(self, tmp_path):
        emb = _FakeEmbedder()
        cache = EmbeddingCache(str(tmp_path / "emb.db"), embedder=emb)
        # Prime two of the four texts
        cache.embed_query("a")
        cache.embed_query("b")
        emb.query_calls = 0
        emb.doc_calls = 0
        # Mixed call: 2 hits + 2 misses
        out = cache.embed_documents(["a", "c", "b", "d"])
        assert len(out) == 4
        # Only the misses should have hit the model
        assert emb.doc_calls == 1
        # And that batch contained the 2 missing texts
        # (we can't easily inspect args; check call count > 0)

    def test_empty_list_no_op(self, tmp_path):
        cache = EmbeddingCache(str(tmp_path / "emb.db"), embedder=_FakeEmbedder())
        assert cache.embed_documents([]) == []


# ----- ResponseCache -----


class TestResponseCache:
    def test_same_payload_same_key(self, tmp_path):
        c = ResponseCache(str(tmp_path / "r.db"))
        k1 = c.make_key({"q": "x", "n": 1})
        k2 = c.make_key({"n": 1, "q": "x"})  # different field order
        assert k1 == k2

    def test_different_payload_different_key(self, tmp_path):
        c = ResponseCache(str(tmp_path / "r.db"))
        k1 = c.make_key({"q": "x"})
        k2 = c.make_key({"q": "y"})
        assert k1 != k2

    def test_set_then_get(self, tmp_path):
        c = ResponseCache(str(tmp_path / "r.db"))
        payload = {"question": "Algoritma nedir?", "workspace": "default"}
        c.set(payload, {"answer": "Algoritma adım adım..."})
        got = c.get(payload)
        assert got == {"answer": "Algoritma adım adım..."}

    def test_miss_when_param_differs(self, tmp_path):
        c = ResponseCache(str(tmp_path / "r.db"))
        c.set({"q": "x", "k": 5}, "first")
        assert c.get({"q": "x", "k": 10}) is None

    def test_invalidate_workspace_clears_all(self, tmp_path):
        c = ResponseCache(str(tmp_path / "r.db"))
        c.set({"q": "x"}, "ans")
        c.invalidate_workspace("default")
        assert c.get({"q": "x"}) is None


# ----- SemanticCache -----


class TestSemanticCache:
    def test_exact_match_hits(self, tmp_path):
        embed = lambda t: [1.0, 0.0] if "first" in t else [0.0, 1.0]
        c = SemanticCache(str(tmp_path / "s.db"), embed_fn=embed,
                         similarity_threshold=0.95)
        c.set("first query", "first answer")
        got, sim = c.get("first query")
        assert got == "first answer"
        assert sim == pytest.approx(1.0)

    def test_semantically_similar_hit(self, tmp_path):
        # Both "Paris population" and "Paris how many people" share dimension 0
        def embed(text):
            if "paris" in text.lower():
                return [0.95, 0.1, 0.05]
            return [0.0, 1.0, 0.0]
        c = SemanticCache(str(tmp_path / "s.db"), embed_fn=embed,
                         similarity_threshold=0.9)
        c.set("Paris'in nüfusu kaç?", "yaklaşık 2.1M")
        got, sim = c.get("Paris'te kaç kişi yaşıyor?")
        assert got == "yaklaşık 2.1M"
        assert sim >= 0.9

    def test_below_threshold_miss(self, tmp_path):
        def embed(text):
            return [1.0, 0.0] if "topic_a" in text else [0.0, 1.0]
        c = SemanticCache(str(tmp_path / "s.db"), embed_fn=embed,
                         similarity_threshold=0.9)
        c.set("topic_a question", "answer A")
        got, sim = c.get("topic_b question")
        assert got is None
        assert sim < 0.9

    def test_empty_query_returns_miss(self, tmp_path):
        c = SemanticCache(str(tmp_path / "s.db"), embed_fn=lambda t: [1.0])
        got, sim = c.get("")
        assert got is None and sim == 0.0

    def test_invalid_threshold_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            SemanticCache(str(tmp_path / "s.db"),
                         embed_fn=lambda t: [1.0],
                         similarity_threshold=1.5)

    def test_embed_failure_returns_miss(self, tmp_path):
        def bad_embed(text):
            raise RuntimeError("model down")
        c = SemanticCache(str(tmp_path / "s.db"), embed_fn=bad_embed)
        got, sim = c.get("any query")
        assert got is None and sim == 0.0
