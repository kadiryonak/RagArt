"""Unit tests for RagRegistry — per-workspace RAG cache lifecycle.

build() is monkeypatched to a lightweight fake so the cache/lock/
invalidation logic can be tested without standing up a real RAG system.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.services import RagRegistry


def _registry() -> RagRegistry:
    wm = MagicMock()
    wm.resolve.side_effect = lambda x: x or "default"
    return RagRegistry(wm)


class TestRagRegistry:
    def test_get_builds_once_and_caches(self):
        reg = _registry()
        builds = []
        reg.build = lambda ws: builds.append(ws) or object()
        a = reg.get("ws1")
        b = reg.get("ws1")
        assert a is b
        assert builds == ["ws1"]  # built exactly once

    def test_get_resolves_workspace_id(self):
        reg = _registry()
        reg.build = lambda ws: object()
        reg.get(None)
        reg._wm.resolve.assert_called_with(None)

    def test_invalidate_forces_rebuild(self):
        reg = _registry()
        builds = []
        reg.build = lambda ws: builds.append(ws) or object()
        reg.get("ws1")
        reg.invalidate("ws1")
        reg.get("ws1")
        assert builds == ["ws1", "ws1"]  # rebuilt after invalidation

    def test_cached_does_not_build(self):
        reg = _registry()

        def _boom(ws):
            raise AssertionError("build() must not run for cached()")

        reg.build = _boom
        assert reg.cached("ws1") is None

    def test_cached_returns_instance_after_get(self):
        reg = _registry()
        sentinel = object()
        reg.build = lambda ws: sentinel
        reg.get("ws1")
        assert reg.cached("ws1") is sentinel

    def test_separate_workspaces_get_separate_instances(self):
        reg = _registry()
        reg.build = lambda ws: object()
        assert reg.get("ws1") is not reg.get("ws2")


class TestRagRegistryBuild:
    """build() wires settings → TurkishRAGSystem; both are stubbed here."""

    def _wm(self):
        wm = MagicMock()
        wm.resolve.side_effect = lambda x: x or "default"
        wm.get.return_value = MagicMock(vector_db="chroma")
        wm.files_dir.return_value = "/tmp/files"
        wm.vector_db_path.return_value = "/tmp/vdb"
        return wm

    def _patch_rag(self, monkeypatch):
        import src.services.rag_registry as mod
        created = {}

        def fake_rag(**kwargs):
            created.update(kwargs)
            inst = MagicMock()
            created["instance"] = inst
            return inst

        monkeypatch.setattr(mod, "TurkishRAGSystem", fake_rag)
        return mod, created

    def test_keyed_provider_with_key_used(self, monkeypatch):
        mod, created = self._patch_rag(monkeypatch)
        monkeypatch.setattr(mod.settings, "get_api_key", lambda: "gsk_live")
        monkeypatch.setattr(type(mod.settings), "MODEL_TYPE",
                            property(lambda self: "groq"))
        reg = RagRegistry(self._wm())
        rag = reg.build("ws1")
        assert created["model_type"] == "groq"
        assert created["api_key"] == "gsk_live"
        rag.initialize.assert_called_once()

    def test_keyed_provider_without_key_falls_back_to_local(self, monkeypatch):
        mod, created = self._patch_rag(monkeypatch)
        monkeypatch.setattr(mod.settings, "get_api_key", lambda: None)
        monkeypatch.setattr(type(mod.settings), "MODEL_TYPE",
                            property(lambda self: "groq"))
        reg = RagRegistry(self._wm())
        reg.build("ws1")
        # No key → silently degrade to the local provider
        assert created["model_type"] == "local"

    def test_local_provider_stays_local(self, monkeypatch):
        mod, created = self._patch_rag(monkeypatch)
        monkeypatch.setattr(mod.settings, "get_api_key", lambda: None)
        monkeypatch.setattr(type(mod.settings), "MODEL_TYPE",
                            property(lambda self: "local"))
        reg = RagRegistry(self._wm())
        reg.build("ws1")
        assert created["model_type"] == "local"

    def test_build_resolves_missing_workspace(self, monkeypatch):
        mod, created = self._patch_rag(monkeypatch)
        monkeypatch.setattr(mod.settings, "get_api_key", lambda: None)
        monkeypatch.setattr(type(mod.settings), "MODEL_TYPE",
                            property(lambda self: "local"))
        wm = self._wm()
        # First get() returns None → build() must re-resolve then re-get
        wm.get.side_effect = [None, MagicMock(vector_db="chroma")]
        reg = RagRegistry(wm)
        reg.build("ghost")
        wm.resolve.assert_called_with("ghost")
