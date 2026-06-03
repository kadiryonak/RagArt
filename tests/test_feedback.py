"""Tests for the feedback route + Prometheus metrics exposition."""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as app_module
    from config.settings import settings

    # Point feedback storage at a temp dir.
    monkeypatch.setattr(settings, "DATA_FOLDER", str(tmp_path), raising=False)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client(), tmp_path


class TestFeedback:
    def test_rating_only_accepted(self, client):
        c, tmp = client
        r = c.post("/feedback", json={"rating": 5})
        assert r.status_code == 200
        assert r.get_json()["success"] is True
        line = (tmp / "feedback.jsonl").read_text(encoding="utf-8").strip()
        assert json.loads(line)["rating"] == 5

    def test_message_only_accepted(self, client):
        c, tmp = client
        r = c.post("/feedback", json={"message": "Harika bir uygulama!"})
        assert r.status_code == 200
        entry = json.loads((tmp / "feedback.jsonl").read_text(encoding="utf-8").strip())
        assert entry["message"] == "Harika bir uygulama!"

    def test_empty_feedback_rejected(self, client):
        c, _ = client
        r = c.post("/feedback", json={})
        assert r.status_code == 400

    def test_out_of_range_rating_rejected(self, client):
        c, _ = client
        assert c.post("/feedback", json={"rating": 9}).status_code == 400

    def test_appends_multiple_lines(self, client):
        c, tmp = client
        c.post("/feedback", json={"rating": 4})
        c.post("/feedback", json={"rating": 2, "message": "olabilir"})
        lines = (tmp / "feedback.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_stats_aggregates(self, client):
        c, _ = client
        c.post("/feedback", json={"rating": 5})
        c.post("/feedback", json={"rating": 3, "message": "idare eder"})
        stats = c.get("/feedback/stats").get_json()
        assert stats["total"] == 2
        assert stats["average_rating"] == 4.0
        assert stats["with_message"] == 1

    def test_stats_empty(self, client):
        c, _ = client
        stats = c.get("/feedback/stats").get_json()
        assert stats["total"] == 0
        assert stats["average_rating"] is None


class TestPrometheus:
    def test_endpoint_exposition_format(self, client):
        c, _ = client
        c.get("/health")  # generate at least one request
        r = c.get("/metrics/prometheus")
        assert r.status_code == 200
        assert r.mimetype == "text/plain"
        body = r.get_data(as_text=True)
        assert "ragart_requests_total" in body
        assert "# TYPE ragart_requests_total counter" in body
        assert "ragart_latency_seconds_p95" in body

    def test_render_prometheus_unit(self):
        from src.observability import Metrics, render_prometheus
        m = Metrics()
        m.record("/ask", 200, 0.1)
        m.record("/ask", 500, 0.2)
        text = render_prometheus(m.snapshot())
        assert "ragart_requests_total 2" in text
        assert "ragart_errors_5xx_total 1" in text
        assert 'ragart_requests_by_status_total{status="2xx"} 1' in text
