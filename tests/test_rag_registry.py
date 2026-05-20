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
