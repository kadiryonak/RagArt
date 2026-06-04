"""Unit tests for src/api/routes/system.py — status/health/metrics/stats/etc."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def client(monkeypatch):
    import app as app_module
    from src.api import runtime

    fake_rag = MagicMock()
    fake_rag.model_type = "groq"
    fake_rag.api_key = "gsk_x"
    fake_rag.get_stats.return_value = {"model_type": "groq", "vector_store_ready": True}
    fake_rag.document_loader.get_file_info.return_value = [
        {"filename": "a.json", "document_count": 3},
        {"filename": "b.pdf", "document_count": 2},
    ]
    fake_rag.ask.return_value = {
        "answer": "kısa cevap",
        "source_documents": [1, 2],
        "source": "rag_system",
        "relevance_score": 0.8,
    }

    monkeypatch.setattr(runtime, "get_rag_for", lambda _ws: fake_rag)
    monkeypatch.setattr(runtime.rag_registry, "cached", lambda _ws: fake_rag)
    monkeypatch.setattr(runtime.system, "ready", True)
    monkeypatch.setattr(runtime.system, "status", "Ready")
    monkeypatch.setattr(runtime.system, "error", None)

    app_module.app.config["TESTING"] = True
    return app_module.app.test_client(), runtime, fake_rag


class TestStatus:
    def test_status_ready(self, client):
        c, _, _ = client
        body = c.get("/status").get_json()
        assert body["ready"] is True
        assert body["model_type"] == "groq"

    def test_status_unknown_model_when_no_rag(self, client):
        c, runtime, _ = client
        # cached() returns None but default workspace is still "ready"
        import importlib
        runtime.rag_registry.cached = lambda _ws: None
        body = c.get("/status").get_json()
        assert body["model_type"] == "unknown"


class TestHealth:
    def test_health_reports_model_and_metrics(self, client):
        c, _, _ = client
        body = c.get("/health").get_json()
        assert body["status"] == "healthy"
        assert body["model_type"] == "groq"
        assert body["api_available"] is True
        assert "uptime_seconds" in body
        assert "requests_total" in body

    def test_health_no_rag_api_unavailable(self, client):
        c, runtime, _ = client
        runtime.rag_registry.cached = lambda _ws: None
        body = c.get("/health").get_json()
        assert body["api_available"] is False
        assert body["model_type"] == "unknown"


class TestMetricsEndpoint:
    def test_metrics_returns_snapshot(self, client):
        c, _, _ = client
        body = c.get("/metrics").get_json()
        assert "requests_total" in body
        assert "latency_seconds" in body


class TestSettingsSchema:
    def test_schema_has_providers(self, client):
        c, _, _ = client
        body = c.get("/settings/schema").get_json()
        assert "prompt_strategies" in body


class TestStats:
    def test_stats_ready(self, client):
        c, _, _ = client
        body = c.get("/stats").get_json()
        assert body["model_type"] == "groq"

    def test_stats_not_ready_503(self, client):
        c, runtime, _ = client
        runtime.system.ready = False
        r = c.get("/stats")
        assert r.status_code == 503


class TestDataInfo:
    def test_data_info_aggregates_counts(self, client):
        c, _, _ = client
        body = c.get("/data-info").get_json()
        assert body["total_files"] == 2
        assert body["total_documents"] == 5
        assert body["model_type"] == "groq"

    def test_data_info_not_ready_503(self, client):
        c, runtime, _ = client
        runtime.system.ready = False
        assert c.get("/data-info").status_code == 503

    def test_data_info_handles_loader_error(self, client):
        c, _, fake_rag = client
        fake_rag.document_loader.get_file_info.side_effect = RuntimeError("boom")
        r = c.get("/data-info")
        assert r.status_code == 500
        assert "boom" in r.get_json()["error"]


class TestSelfTest:
    def test_runs_canned_questions(self, client):
        c, _, fake_rag = client
        body = c.get("/test").get_json()
        assert len(body["test_results"]) == 4
        assert all("answer" in r for r in body["test_results"])

    def test_truncates_long_answers(self, client):
        c, _, fake_rag = client
        fake_rag.ask.return_value = {
            "answer": "x" * 500,
            "source_documents": [],
            "source": "rag_system",
            "relevance_score": 0.5,
        }
        body = c.get("/test").get_json()
        assert body["test_results"][0]["answer"].endswith("...")

    def test_captures_per_question_errors(self, client):
        c, _, fake_rag = client
        fake_rag.ask.side_effect = RuntimeError("ask failed")
        body = c.get("/test").get_json()
        assert all("error" in r for r in body["test_results"])

    def test_not_ready_503(self, client):
        c, runtime, _ = client
        runtime.system.ready = False
        assert c.get("/test").status_code == 503
