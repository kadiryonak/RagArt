"""End-to-end integration tests for the RagArt HTTP API.

Every test here drives the REAL Flask app over a REAL ChromaDB index
(see conftest.live). They exercise the full request → pipeline → retrieval
→ response path, including SSE streaming, the upload → reindex → query
lifecycle, and workspace isolation.

Marked `integration` so they can be deselected in fast lanes with
`-m "not integration"`.
"""

from __future__ import annotations

import io
import json

import pytest

from tests.integration.conftest import parse_sse

pytestmark = pytest.mark.integration


# ── Health / readiness ─────────────────────────────────────────────────


class TestHealthAndStatus:
    def test_health_ready(self, live):
        client, _, _ = live
        body = client.get("/health").get_json()
        assert body["status"] == "healthy"
        assert body["system_ready"] is True
        assert body["model_type"] == "local"

    def test_status_ready(self, live):
        client, _, _ = live
        body = client.get("/status").get_json()
        assert body["ready"] is True
        assert body["workspace_id"] == "default"

    def test_metrics_endpoint(self, live):
        client, _, _ = live
        body = client.get("/metrics").get_json()
        assert "requests_total" in body
        assert "latency_seconds" in body

    def test_settings_schema(self, live):
        client, _, _ = live
        body = client.get("/settings/schema").get_json()
        assert "prompt_strategies" in body


# ── Core ask flow (real retrieval) ─────────────────────────────────────


class TestAskFlow:
    def test_ask_retrieves_relevant_source(self, live):
        client, _, _ = live
        r = client.post("/ask", json={
            "question": "Algoritma nedir ve adımları nasıl sıralı çözme yapar?",
        })
        assert r.status_code == 200
        body = r.get_json()
        assert body["success"] is True
        assert isinstance(body["answer"], str) and body["answer"]
        # Retrieval ran end-to-end: the algorithm doc should surface.
        titles = [s["title"] for s in body["sources"]]
        assert "algoritma.json" in titles
        assert body["relevance_score"] > 0.0

    def test_ask_python_question_retrieves_python_doc(self, live):
        client, _, _ = live
        r = client.post("/ask", json={
            "question": "Python programlama dili veri yapısı nedir?",
        })
        assert r.status_code == 200
        titles = [s["title"] for s in r.get_json()["sources"]]
        assert "python.json" in titles

    def test_ask_missing_question_is_400(self, live):
        client, _, _ = live
        r = client.post("/ask", json={})
        assert r.status_code == 400

    def test_ask_blank_question_is_400(self, live):
        client, _, _ = live
        r = client.post("/ask", json={"question": "   "})
        assert r.status_code == 400

    def test_unknown_workspace_header_falls_back_to_default(self, live):
        client, _, _ = live
        r = client.post(
            "/ask",
            json={"question": "Algoritma nedir?"},
            headers={"X-Workspace-Id": "does-not-exist"},
        )
        # Resolves to default rather than erroring.
        assert r.status_code == 200


# ── SSE streaming ──────────────────────────────────────────────────────


class TestStreaming:
    def test_stream_emits_sources_and_done(self, live):
        client, _, _ = live
        r = client.post("/ask/stream", json={
            "question": "Algoritma nedir, sıralı adım çözme?",
        })
        assert r.status_code == 200
        assert r.mimetype == "text/event-stream"
        events = parse_sse(r.get_data(as_text=True))
        types = [e.get("type") for e in events]
        assert "sources" in types
        assert "done" in types
        done = next(e for e in events if e.get("type") == "done")
        assert "answer" in done

    def test_stream_tokens_present(self, live):
        client, _, _ = live
        r = client.post("/ask/stream", json={"question": "Python nedir?"})
        events = parse_sse(r.get_data(as_text=True))
        # Local provider streams the full answer as at least one token event.
        assert any(e.get("type") == "token" for e in events)


# ── File lifecycle: upload → reindex → query ───────────────────────────


class TestFileLifecycle:
    def test_list_files_shows_seeds(self, live):
        client, _, _ = live
        body = client.get("/list-files").get_json()
        names = {f["filename"] for f in body["files"]}
        assert {"algoritma.json", "python.json"}.issubset(names)

    def test_upload_reindex_then_query(self, live):
        client, _, _ = live
        # 1. Upload a new, lexically-distinct document.
        content = (
            "Yapay zeka, makine öğrenme ve derin sinir ağ modellerini kapsar. "
            "Yapay zeka model öğrenme kavramıdır."
        ).encode("utf-8")
        up = client.post(
            "/upload",
            data={"file": (io.BytesIO(content), "yapayzeka.txt")},
            content_type="multipart/form-data",
        )
        assert up.status_code == 200, up.get_data(as_text=True)
        assert up.get_json()["document_count"] >= 1

        # 2. Incremental reindex picks up the new file.
        rx = client.post("/reindex", json={})
        assert rx.status_code == 200
        sync = rx.get_json()["sync"]
        assert "yapayzeka.txt" in sync["added"]

        # 3. The new content is now retrievable.
        r = client.post("/ask", json={
            "question": "Yapay zeka makine öğrenme model nedir?",
        })
        assert r.status_code == 200
        titles = [s["title"] for s in r.get_json()["sources"]]
        assert "yapayzeka.txt" in titles

    def test_short_propernoun_query_finds_uploaded_doc(self, live):
        # Regression for the "sonradan indekslediğim belgeyi bulamıyor" report:
        # a 2-word name query is classified SIMPLE. Before the fix, SIMPLE was
        # dense-only k=2 and missed the proper noun; now it uses hybrid so BM25
        # exact-matches the name, and the relevance gate uses the best doc.
        client, _, _ = live
        content = (
            "Demir Kaya bir yazılım mühendisidir. "
            "Demir, yapay zeka ve makine öğrenme alanında çalışır."
        ).encode("utf-8")
        client.post(
            "/upload",
            data={"file": (io.BytesIO(content), "demir.txt")},
            content_type="multipart/form-data",
        )
        client.post("/reindex", json={})

        r = client.post("/ask", json={"question": "Demir kimdir"})
        assert r.status_code == 200
        body = r.get_json()
        assert body["source_type"] != "insufficient_data"
        assert "demir.txt" in [s["title"] for s in body["sources"]]

    def test_upload_rejects_unsupported_type(self, live):
        client, _, _ = live
        up = client.post(
            "/upload",
            data={"file": (io.BytesIO(b"x"), "evil.exe")},
            content_type="multipart/form-data",
        )
        assert up.status_code == 400

    def test_upload_without_file_is_400(self, live):
        client, _, _ = live
        up = client.post("/upload", data={}, content_type="multipart/form-data")
        assert up.status_code == 400

    def test_serve_source_returns_file(self, live):
        client, _, _ = live
        r = client.get("/source/algoritma.json")
        assert r.status_code == 200
        assert "json" in r.mimetype

    def test_serve_source_path_traversal_blocked(self, live):
        client, _, _ = live
        r = client.get("/source/..%2f..%2fsecret.txt")
        assert r.status_code in (400, 404)


# ── Workspace isolation ────────────────────────────────────────────────


class TestWorkspaceIsolation:
    def test_create_query_isolated_workspace(self, live):
        client, wm, runtime = live
        # 1. Create a fresh workspace.
        cr = client.post("/workspaces", json={"name": "Izole Test"})
        assert cr.status_code == 201
        ws_id = cr.get_json()["workspace"]["id"]
        hdr = {"X-Workspace-Id": ws_id}

        try:
            # 2. It starts empty — no seed files leak in.
            files = client.get("/list-files", headers=hdr).get_json()["files"]
            assert files == []

            # 3. Upload + reindex its own document.
            content = "İzole workspace içinde sadece python programlama dil bilgisi var.".encode("utf-8")
            up = client.post(
                "/upload",
                data={"file": (io.BytesIO(content), "izole.txt")},
                content_type="multipart/form-data",
                headers=hdr,
            )
            assert up.status_code == 200
            rx = client.post("/reindex", json={}, headers=hdr)
            assert rx.status_code == 200

            # 4. Querying the isolated workspace returns ITS doc, not the
            #    default workspace's algoritma.json.
            r = client.post(
                "/ask",
                json={"question": "python programlama dil nedir?"},
                headers=hdr,
            )
            assert r.status_code == 200
            titles = [s["title"] for s in r.get_json()["sources"]]
            assert "izole.txt" in titles
            assert "algoritma.json" not in titles
        finally:
            # Best-effort cleanup. On Windows the cached RAG keeps the
            # workspace's SQLite cache file open, so rmtree can raise a
            # PermissionError (Linux/prod is unaffected) — don't fail the
            # isolation assertion over a platform file-lock quirk.
            runtime.invalidate_rag(ws_id)
            try:
                client.delete(f"/workspaces/{ws_id}")
            except PermissionError:
                pass  # Windows file-lock on the open SQLite cache; ignore.

    def test_delete_unqueried_workspace(self, live):
        client, _, _ = live
        # A workspace whose RAG was never built holds no open files, so
        # deletion succeeds cleanly on every platform.
        ws_id = client.post("/workspaces", json={"name": "Silinecek"}) \
            .get_json()["workspace"]["id"]
        r = client.delete(f"/workspaces/{ws_id}")
        assert r.status_code == 200
        assert r.get_json()["success"] is True

    def test_default_workspace_cannot_be_deleted(self, live):
        client, _, _ = live
        r = client.delete("/workspaces/default")
        assert r.status_code == 400

    def test_list_workspaces_includes_default(self, live):
        client, _, _ = live
        body = client.get("/workspaces").get_json()
        ids = {w["id"] for w in body["workspaces"]}
        assert "default" in ids
        assert body["default_id"] == "default"


# ── Cache layer ────────────────────────────────────────────────────────


class TestCacheLayer:
    def test_cache_stats_and_clear(self, live):
        client, _, _ = live
        # Prime the cache with a couple of asks.
        client.post("/ask", json={"question": "Algoritma nedir?"})
        client.post("/ask", json={"question": "Algoritma nedir?"})

        stats = client.get("/cache/stats")
        assert stats.status_code == 200
        assert stats.get_json()["success"] is True

        cleared = client.post("/cache/clear", json={"layer": "all"})
        assert cleared.status_code == 200
        assert cleared.get_json()["success"] is True
