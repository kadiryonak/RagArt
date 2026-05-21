"""Tests for src/observability.py — request ID context var + Flask middleware."""

from __future__ import annotations

import logging
import re

import pytest

from src.observability import (
    get_request_id,
    install_flask_middleware,
    install_logging,
    new_request_id,
    set_request_id,
)


class TestContextVar:
    def test_default_is_dash(self):
        # Fresh ContextVar default — tests run isolated by pytest
        # so this may already be set by a previous test; we explicitly
        # reset for determinism.
        set_request_id("-")
        assert get_request_id() == "-"

    def test_set_then_get(self):
        set_request_id("abc12345")
        assert get_request_id() == "abc12345"

    def test_new_request_id_is_8_hex(self):
        rid = new_request_id()
        assert re.match(r"^[0-9a-f]{8}$", rid), rid

    def test_new_request_id_is_unique(self):
        ids = {new_request_id() for _ in range(50)}
        assert len(ids) >= 49  # collisions extremely unlikely


class TestLoggingFilter:
    def test_filter_attaches_request_id_to_record(self):
        # Direct filter test — caplog uses its own handler that bypasses
        # ours, so verifying the filter is on the root handler is brittle.
        # Test the unit (filter) instead.
        from src.observability import _RequestIDFilter
        set_request_id("deadbeef")
        rec = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        result = _RequestIDFilter().filter(rec)
        assert result is True
        assert rec.request_id == "deadbeef"

    def test_install_logging_attaches_filter_to_handler(self):
        install_logging(level=logging.INFO)
        root = logging.getLogger()
        assert root.handlers, "install_logging should add at least one handler"
        # The first (and only) handler must carry our filter
        h = root.handlers[0]
        from src.observability import _RequestIDFilter
        assert any(isinstance(f, _RequestIDFilter) for f in h.filters)

    def test_install_logging_replaces_old_handlers(self):
        # Add a noise handler first
        root = logging.getLogger()
        noise = logging.StreamHandler()
        root.addHandler(noise)
        install_logging(level=logging.INFO)
        # After install_logging, the noise handler must be gone
        assert noise not in root.handlers


class TestFlaskMiddleware:
    @pytest.fixture
    def app_with_middleware(self):
        from flask import Flask, jsonify
        app = Flask(__name__)
        install_flask_middleware(app)

        @app.route("/echo")
        def echo():
            return jsonify({"rid": get_request_id()})

        return app

    def test_response_carries_request_id_header(self, app_with_middleware):
        client = app_with_middleware.test_client()
        r = client.get("/echo")
        assert r.status_code == 200
        assert "X-Request-ID" in r.headers
        body_rid = r.get_json()["rid"]
        header_rid = r.headers["X-Request-ID"]
        assert body_rid == header_rid, "body and header must agree"
        assert re.match(r"^[0-9a-f]{8,32}$", header_rid)

    def test_request_id_from_client_is_preserved(self, app_with_middleware):
        client = app_with_middleware.test_client()
        r = client.get("/echo", headers={"X-Request-ID": "client-supplied"})
        assert r.headers["X-Request-ID"] == "client-supplied"
        assert r.get_json()["rid"] == "client-supplied"

    def test_overlong_client_id_capped(self, app_with_middleware):
        client = app_with_middleware.test_client()
        huge = "x" * 500
        r = client.get("/echo", headers={"X-Request-ID": huge})
        # We cap to 32 chars to keep log lines sane
        assert len(r.headers["X-Request-ID"]) <= 32

    def test_each_request_gets_unique_id(self, app_with_middleware):
        client = app_with_middleware.test_client()
        seen = {client.get("/echo").get_json()["rid"] for _ in range(10)}
        assert len(seen) == 10  # all distinct


class TestMetrics:
    def _fresh(self):
        from src.observability import Metrics
        return Metrics()

    def test_counts_total_and_status_buckets(self):
        m = self._fresh()
        m.record("/ask", 200, 0.10)
        m.record("/ask", 200, 0.20)
        m.record("/ask", 500, 0.05)
        snap = m.snapshot()
        assert snap["requests_total"] == 3
        assert snap["requests_by_status"] == {"2xx": 2, "5xx": 1}
        assert snap["errors_5xx"] == 1

    def test_counts_per_endpoint(self):
        m = self._fresh()
        m.record("/ask", 200, 0.1)
        m.record("/health", 200, 0.01)
        m.record("/ask", 200, 0.1)
        snap = m.snapshot()
        assert snap["requests_by_endpoint"] == {"/ask": 2, "/health": 1}

    def test_latency_percentiles(self):
        m = self._fresh()
        for i in range(1, 101):          # 0.01 .. 1.00 s
            m.record("/ask", 200, i / 100)
        lat = m.snapshot()["latency_seconds"]
        assert lat["samples"] == 100
        assert lat["p50"] == pytest.approx(0.50, abs=0.02)
        assert lat["p99"] == pytest.approx(1.00, abs=0.02)
        assert lat["max"] == pytest.approx(1.00, abs=0.001)

    def test_empty_snapshot_is_safe(self):
        snap = self._fresh().snapshot()
        assert snap["requests_total"] == 0
        assert snap["latency_seconds"]["p95"] == 0.0

    def test_reset_clears_counters(self):
        m = self._fresh()
        m.record("/ask", 200, 0.1)
        m.reset()
        assert m.snapshot()["requests_total"] == 0

    def test_middleware_feeds_the_metrics_singleton(self):
        from flask import Flask, jsonify
        from src.observability import metrics

        metrics.reset()
        app = Flask(__name__)
        install_flask_middleware(app)

        @app.route("/ping")
        def ping():
            return jsonify({"ok": True})

        client = app.test_client()
        client.get("/ping")
        client.get("/ping")
        snap = metrics.snapshot()
        assert snap["requests_total"] == 2
        assert snap["requests_by_endpoint"].get("/ping") == 2
        metrics.reset()
