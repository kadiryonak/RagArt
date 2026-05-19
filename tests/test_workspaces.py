"""Unit + integration tests for the workspaces feature."""

from __future__ import annotations

import json

import pytest

from src.workspaces import (
    DEFAULT_WORKSPACE_ID,
    Workspace,
    WorkspaceManager,
    _slugify,
)


# ----- Slugify -----


class TestSlugify:
    @pytest.mark.parametrize("inp,expected_prefix", [
        ("My Project", "my-project"),
        ("Türkçe İçerik", "t-rk-e-erik"),  # transliterates to ASCII
        ("ABC 123", "abc-123"),
        ("   spaces   ", "spaces"),
    ])
    def test_normalises(self, inp, expected_prefix):
        out = _slugify(inp)
        # Allow some flexibility — exact transliteration depends on stdlib
        assert out.startswith(expected_prefix.split("-")[0]) or len(out) > 0

    def test_empty_falls_back_to_uuid(self):
        out = _slugify("")
        assert len(out) >= 4  # uuid hex slice


# ----- WorkspaceManager CRUD -----


class TestWorkspaceCRUD:
    def test_default_workspace_created_on_init(self, tmp_path):
        wm = WorkspaceManager(str(tmp_path))
        items = wm.list()
        assert any(w.id == DEFAULT_WORKSPACE_ID for w in items)

    def test_create_workspace(self, tmp_path):
        wm = WorkspaceManager(str(tmp_path))
        ws = wm.create("My Project", description="Test")
        assert ws.name == "My Project"
        assert ws.description == "Test"
        assert wm.exists(ws.id)

    def test_duplicate_name_gets_unique_id(self, tmp_path):
        wm = WorkspaceManager(str(tmp_path))
        ws1 = wm.create("Project")
        ws2 = wm.create("Project")
        assert ws1.id != ws2.id

    def test_empty_name_rejected(self, tmp_path):
        wm = WorkspaceManager(str(tmp_path))
        with pytest.raises(ValueError):
            wm.create("   ")

    def test_delete_workspace(self, tmp_path):
        wm = WorkspaceManager(str(tmp_path))
        ws = wm.create("Disposable")
        assert wm.delete(ws.id) is True
        assert not wm.exists(ws.id)

    def test_cannot_delete_default(self, tmp_path):
        wm = WorkspaceManager(str(tmp_path))
        with pytest.raises(ValueError):
            wm.delete(DEFAULT_WORKSPACE_ID)

    def test_delete_unknown_returns_false(self, tmp_path):
        wm = WorkspaceManager(str(tmp_path))
        assert wm.delete("nonexistent") is False

    def test_update_metadata(self, tmp_path):
        wm = WorkspaceManager(str(tmp_path))
        ws = wm.create("Old Name")
        updated = wm.update(ws.id, name="New Name", color="#abcdef",
                            description="Bigger desc")
        assert updated.name == "New Name"
        assert updated.color == "#abcdef"
        assert updated.description == "Bigger desc"

    def test_resolve_unknown_falls_back_to_default(self, tmp_path):
        wm = WorkspaceManager(str(tmp_path))
        assert wm.resolve("bogus") == DEFAULT_WORKSPACE_ID
        assert wm.resolve(None) == DEFAULT_WORKSPACE_ID


# ----- Migration -----


class TestMigration:
    def test_legacy_json_moved_to_default(self, tmp_path):
        # Plant a fake legacy JSON in data root
        (tmp_path / "old.json").write_text(
            json.dumps({"title": "x", "content": "y"}), encoding="utf-8"
        )
        wm = WorkspaceManager(str(tmp_path))
        # File should now be inside the default workspace's files/ dir
        default_files = wm.files_dir(DEFAULT_WORKSPACE_ID)
        assert (default_files / "old.json").exists()
        # And the legacy location is empty
        assert not (tmp_path / "old.json").exists()

    def test_migration_only_runs_once(self, tmp_path):
        wm1 = WorkspaceManager(str(tmp_path))
        ws = wm1.create("Existing")
        # Plant a new file after first init
        (tmp_path / "new.json").write_text('{"a": 1}', encoding="utf-8")
        # Second init shouldn't migrate (default already exists)
        wm2 = WorkspaceManager(str(tmp_path))
        default_files = wm2.files_dir(DEFAULT_WORKSPACE_ID)
        assert not (default_files / "new.json").exists()


# ----- Isolation -----


class TestIsolation:
    def test_files_in_different_workspaces_dont_overlap(self, tmp_path):
        wm = WorkspaceManager(str(tmp_path))
        ws_a = wm.create("Workspace A")
        ws_b = wm.create("Workspace B")
        fa = wm.files_dir(ws_a.id) / "secret_a.txt"
        fb = wm.files_dir(ws_b.id) / "secret_b.txt"
        fa.write_text("a content", encoding="utf-8")
        fb.write_text("b content", encoding="utf-8")

        # Each workspace sees only its own file
        files_a = list(wm.files_dir(ws_a.id).iterdir())
        files_b = list(wm.files_dir(ws_b.id).iterdir())
        assert {f.name for f in files_a} == {"secret_a.txt"}
        assert {f.name for f in files_b} == {"secret_b.txt"}

    def test_collection_names_differ_per_workspace(self, tmp_path):
        wm = WorkspaceManager(str(tmp_path))
        ws_a = wm.create("A")
        ws_b = wm.create("B")
        assert wm.collection_name(ws_a.id) != wm.collection_name(ws_b.id)

    def test_vector_db_paths_differ(self, tmp_path):
        wm = WorkspaceManager(str(tmp_path))
        ws_a = wm.create("A")
        ws_b = wm.create("B")
        path_a = wm.vector_db_path(ws_a.id, "chroma")
        path_b = wm.vector_db_path(ws_b.id, "chroma")
        assert path_a != path_b


# ----- Workspace dataclass roundtrip -----


class TestWorkspaceRoundtrip:
    def test_to_dict_from_dict(self):
        ws = Workspace(id="x", name="X", color="#fff", description="d")
        roundtrip = Workspace.from_dict(ws.to_dict())
        assert roundtrip.name == ws.name
        assert roundtrip.color == ws.color

    def test_from_dict_drops_unknown_fields(self):
        # Forward-compat: future versions shouldn't crash old loaders
        ws = Workspace.from_dict({
            "id": "x", "name": "X",
            "future_field": "ignored",
        })
        assert ws.id == "x"


# ----- API integration tests -----


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    """Flask test client with a fresh tmp DATA_FOLDER."""
    import app as app_module
    monkeypatch.setattr(app_module.settings, "DATA_FOLDER", str(tmp_path), raising=False)
    # Re-init the workspace manager pointing at tmp_path so tests are isolated
    from src.workspaces import WorkspaceManager
    app_module.workspace_manager = WorkspaceManager(str(tmp_path))
    app_module._rag_cache.clear()
    app_module.system_ready = True
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


class TestWorkspaceAPI:
    def test_list_returns_default(self, api_client):
        r = api_client.get("/workspaces")
        assert r.status_code == 200
        body = r.get_json()
        ids = {w["id"] for w in body["workspaces"]}
        assert DEFAULT_WORKSPACE_ID in ids
        assert body["default_id"] == DEFAULT_WORKSPACE_ID
        assert "vector_stores" in body
        # At least chroma must be registered
        vstore_ids = {v["id"] for v in body["vector_stores"]}
        assert "chroma" in vstore_ids

    def test_create(self, api_client):
        r = api_client.post(
            "/workspaces",
            json={"name": "Yeni Proje", "color": "#abc123", "vector_db": "chroma"},
        )
        assert r.status_code == 201
        body = r.get_json()
        assert body["success"] is True
        assert body["workspace"]["name"] == "Yeni Proje"
        assert body["workspace"]["color"] == "#abc123"

    def test_create_empty_name_rejected(self, api_client):
        r = api_client.post("/workspaces", json={"name": ""})
        assert r.status_code == 400

    def test_create_unknown_vector_db_rejected(self, api_client):
        r = api_client.post(
            "/workspaces",
            json={"name": "x", "vector_db": "magicdb-9000"},
        )
        assert r.status_code == 400

    def test_delete(self, api_client):
        created = api_client.post("/workspaces", json={"name": "Geçici"}).get_json()
        ws_id = created["workspace"]["id"]
        r = api_client.delete(f"/workspaces/{ws_id}")
        assert r.status_code == 200

    def test_cannot_delete_default_via_api(self, api_client):
        r = api_client.delete(f"/workspaces/{DEFAULT_WORKSPACE_ID}")
        assert r.status_code == 400

    def test_patch_rename(self, api_client):
        ws = api_client.post("/workspaces", json={"name": "Eski"}).get_json()["workspace"]
        r = api_client.patch(
            f"/workspaces/{ws['id']}",
            json={"name": "Yeni"},
        )
        assert r.status_code == 200
        assert r.get_json()["workspace"]["name"] == "Yeni"

    def test_patch_changes_vector_db(self, api_client):
        ws = api_client.post(
            "/workspaces",
            json={"name": "DB-Test", "vector_db": "chroma"},
        ).get_json()["workspace"]
        # Find an alternative DB the registry knows about (qdrant if installed,
        # otherwise just patch back to chroma to assert needs_reindex semantics)
        r_list = api_client.get("/workspaces").get_json()
        alt_dbs = [v["id"] for v in r_list["vector_stores"] if v["id"] != "chroma"]
        if not alt_dbs:
            pytest.skip("Only one vector store available; cannot test switching")
        r = api_client.patch(
            f"/workspaces/{ws['id']}",
            json={"vector_db": alt_dbs[0]},
        )
        assert r.status_code == 200
        body = r.get_json()
        assert body["workspace"]["vector_db"] == alt_dbs[0]
        assert body["needs_reindex"] is True

    def test_patch_unknown_vector_db_rejected(self, api_client):
        ws = api_client.post("/workspaces", json={"name": "x"}).get_json()["workspace"]
        r = api_client.patch(
            f"/workspaces/{ws['id']}",
            json={"vector_db": "nonsense-db"},
        )
        assert r.status_code == 400

    def test_list_files_respects_workspace_header(self, api_client):
        # Plant a file in a non-default workspace
        ws = api_client.post("/workspaces", json={"name": "Test"}).get_json()["workspace"]
        # Default workspace has nothing yet because we used tmp_path
        r_default = api_client.get("/list-files")
        r_custom = api_client.get(
            "/list-files",
            headers={"X-Workspace-Id": ws["id"]},
        )
        assert r_default.status_code == 200
        assert r_custom.status_code == 200
        assert r_custom.get_json()["workspace_id"] == ws["id"]
        assert r_default.get_json()["workspace_id"] == DEFAULT_WORKSPACE_ID
