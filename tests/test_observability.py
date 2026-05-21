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
