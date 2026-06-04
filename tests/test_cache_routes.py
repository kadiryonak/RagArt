"""Unit tests for src/api/routes/cache.py — stats + clear endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def client(monkeypatch):
    import app as app_module
    from src.api import runtime

    rag = MagicMock()
    rag.embedding_cache.stats.return_value = {"hits": 1, "misses": 2}
    rag.response_cache.stats.return_value = {"hits": 3, "misses": 4}
    rag.semantic_cache.stats.return_value = {"hits": 5, "misses": 6}
    rag.embedding_cache.clear.return_value = 10
    rag.response_cache.clear.return_value = 20
    rag.semantic_cache.clear.return_value = 30

    monkeypatch.setattr(runtime, "get_rag_for", lambda _ws: rag)
    monkeypatch.setattr(runtime.system, "ready", True)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client(), runtime, rag


class TestCacheStats:
    def test_returns_all_three_layers(self, client):
        c, _, _ = client
        body = c.get("/cache/stats").get_json()
        assert body["success"] is True
        assert set(body["caches"]) == {"embedding", "response", "semantic"}
        assert body["caches"]["response"]["hits"] == 3

    def test_not_ready_503(self, client):
        c, runtime, _ = client
        runtime.system.ready = False
        assert c.get("/cache/stats").status_code == 503

    def test_error_returns_500(self, client):
        c, _, rag = client
        rag.embedding_cache.stats.side_effect = RuntimeError("boom")
        r = c.get("/cache/stats")
        assert r.status_code == 500
        assert "boom" in r.get_json()["error"]


class TestCacheClear:
    def test_clear_all_by_default(self, client):
        c, _, rag = client
        body = c.post("/cache/clear").get_json()
        assert body["success"] is True
        assert body["cleared"] == {"embedding": 10, "response": 20, "semantic": 30}

    def test_clear_single_layer(self, client):
        c, _, rag = client
        body = c.post("/cache/clear", json={"layer": "response"}).get_json()
        assert body["cleared"] == {"response": 20}
        rag.embedding_cache.clear.assert_not_called()

    def test_unknown_layer_rejected_400(self, client):
        c, _, _ = client
        r = c.post("/cache/clear", json={"layer": "bogus"})
        assert r.status_code == 400

    def test_not_ready_503(self, client):
        c, runtime, _ = client
        runtime.system.ready = False
        assert c.post("/cache/clear").status_code == 503

    def test_error_returns_500(self, client):
        c, _, rag = client
        rag.response_cache.clear.side_effect = RuntimeError("explode")
        r = c.post("/cache/clear", json={"layer": "response"})
        assert r.status_code == 500
        assert "explode" in r.get_json()["error"]
