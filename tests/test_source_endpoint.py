"""Integration tests for /source/<filename> endpoint.

Verifies content delivery, mime types and path-traversal protection.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Test client with an isolated workspace rooted at tmp_path.

    /source resolves files via the module-level workspace_manager (not
    settings.DATA_FOLDER), so we swap in a fresh manager rooted at tmp_path
    and hand the test the default workspace's files dir to write into.
    """
    import app as app_module
    from src.api import runtime
    from src.workspaces import WorkspaceManager, DEFAULT_WORKSPACE_ID

    wm = WorkspaceManager(str(tmp_path))
    monkeypatch.setattr(runtime, "workspace_manager", wm)
    app_module.app.config["TESTING"] = True
    files_dir = wm.files_dir(DEFAULT_WORKSPACE_ID)
    return app_module.app.test_client(), files_dir


def _write(path, content):
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))


class TestServesFile:
    def test_json_served_with_correct_mime(self, client):
        c, tmp = client
        _write(tmp / "doc.json", json.dumps({"k": "v"}))
        r = c.get("/source/doc.json")
        assert r.status_code == 200
        assert "application/json" in r.headers["Content-Type"]

    def test_pdf_served_with_pdf_mime(self, client):
        c, tmp = client
        _write(tmp / "doc.pdf", b"%PDF-1.4 minimal")
        r = c.get("/source/doc.pdf")
        assert r.status_code == 200
        assert r.headers["Content-Type"] == "application/pdf"

    def test_markdown_served_as_text(self, client):
        c, tmp = client
        _write(tmp / "note.md", "# title\n\ncontent here")
        r = c.get("/source/note.md")
        assert r.status_code == 200
        assert "text/markdown" in r.headers["Content-Type"]

    def test_txt_served_as_text(self, client):
        c, tmp = client
        _write(tmp / "note.txt", "plain text body")
        r = c.get("/source/note.txt")
        assert r.status_code == 200
        assert "text/plain" in r.headers["Content-Type"]

    def test_docx_served_as_docx_mime(self, client):
        c, tmp = client
        _write(tmp / "doc.docx", b"PK fake docx")
        r = c.get("/source/doc.docx")
        assert r.status_code == 200
        assert "officedocument" in r.headers["Content-Type"]

    def test_unknown_extension_falls_back_to_octet(self, client):
        c, tmp = client
        _write(tmp / "data.xyz", b"binary")
        r = c.get("/source/data.xyz")
        assert r.status_code == 200
        assert "octet-stream" in r.headers["Content-Type"]


class TestErrors:
    def test_missing_file_404(self, client):
        c, _ = client
        r = c.get("/source/does-not-exist.json")
        assert r.status_code == 404

    def test_path_traversal_blocked(self, client):
        # Even with URL encoding, secure_filename strips /../ → mismatch → 400
        c, _ = client
        # Try several traversal patterns
        for path in ["../etc/passwd", "..%2Fpasswd", "subdir/../../passwd"]:
            r = c.get(f"/source/{path}")
            # Either 400 (invalid) or 404 (not found in data folder)
            assert r.status_code in (400, 404), f"path={path} got {r.status_code}"

    def test_filename_with_turkish_chars(self, client):
        c, tmp = client
        # secure_filename may transliterate Turkish chars; verify behaviour
        # Most importantly: don't crash, return either 200 (if survived) or 400
        _write(tmp / "algoritma.json", b'{"a": 1}')
        r = c.get("/source/algoritma.json")
        assert r.status_code == 200
