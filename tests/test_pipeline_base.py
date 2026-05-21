"""Tests for the pipeline base — QueryRequest, QueryState, PipelineStage,
Pipeline runner. No real RAG involved; pure plumbing tests."""

from __future__ import annotations

from typing import Any

import pytest

from src.pipeline import Pipeline, PipelineStage, QueryRequest, QueryState


# ─── Test helpers ──────────────────────────────────────────────────────


class _SetField(PipelineStage):
    """A stage that drops a key/value into extra_meta."""

    def __init__(self, name, key, value):
        self.name = name
        self.key = key
        self.value = value

    def run(self, state):
        state.extra_meta[self.key] = self.value
        return state


class _Responder(PipelineStage):
    """A stage that finishes the pipeline by setting state.response."""
    name = "responder"

    def __init__(self, payload):
        self.payload = payload

    def run(self, state):
        state.response = dict(self.payload)
        return state


class _ShouldNotRun(PipelineStage):
    """Marks itself executed by setting a flag — used to verify
    short-circuit behaviour."""
    name = "should_not_run"
    invoked = False

    def run(self, state):
        type(self).invoked = True
        return state


class _Raiser(PipelineStage):
    name = "raiser"

    def run(self, state):
        raise ValueError("intentional")


# ─── QueryRequest / QueryState ─────────────────────────────────────────


class TestQueryRequest:
    def test_minimum_construction(self):
        r = QueryRequest(question="x")
        assert r.question == "x"
        assert r.k == 5  # default
        assert r.use_response_cache is True

    def test_frozen(self):
        r = QueryRequest(question="x")
        with pytest.raises(Exception):
            r.question = "different"  # type: ignore[misc]

    def test_history_default_empty_tuple(self):
        # Why tuple not list — frozen dataclass needs hashable defaults
        r = QueryRequest(question="x")
        assert r.history == ()


class TestQueryState:
    def test_initial_state(self):
        r = QueryRequest(question="x")
        s = QueryState(request=r, rag=None)
        assert s.docs == []
        assert s.context == ""
        assert s.response is None
        assert s.timings == {}

    def test_mutable(self):
        r = QueryRequest(question="x")
        s = QueryState(request=r, rag=None)
        s.context = "ctx"
        s.docs = ["a", "b"]
        assert s.context == "ctx"
        assert len(s.docs) == 2


# ─── PipelineStage timing + short-circuit ──────────────────────────────


class TestPipelineStage:
    def test_records_timing(self):
        s = _SetField("a", "key", 1)
        state = QueryState(request=QueryRequest(question="x"), rag=None)
        s(state)
        assert "a" in state.timings
        assert isinstance(state.timings["a"], float)

    def test_short_circuit_skips_when_response_set(self):
        _ShouldNotRun.invoked = False
        state = QueryState(request=QueryRequest(question="x"), rag=None)
        state.response = {"already": "set"}
        _ShouldNotRun()(state)
        assert _ShouldNotRun.invoked is False

    def test_short_circuit_still_records_no_op_timing_absence(self):
        # We deliberately *don't* record timing for skipped stages — the
        # timings dict only contains stages that actually ran.
        _ShouldNotRun.invoked = False
        state = QueryState(request=QueryRequest(question="x"), rag=None)
        state.response = {"x": 1}
        _ShouldNotRun()(state)
        assert "should_not_run" not in state.timings

    def test_exception_propagates_but_timing_recorded(self):
        state = QueryState(request=QueryRequest(question="x"), rag=None)
        with pytest.raises(ValueError):
            _Raiser()(state)
        # Even on failure, timing is recorded — useful for debugging slow
        # stages that then raise.
        assert "raiser" in state.timings


# ─── Pipeline runner ───────────────────────────────────────────────────


class TestPipeline:
    def test_runs_stages_in_order(self):
        order = []

        class _Track(PipelineStage):
            def __init__(self, n):
                self.name = n

            def run(self, state):
                order.append(self.name)
                return state

        p = Pipeline([_Track("a"), _Track("b"), _Responder({"answer": "ok"})])
        p.run(QueryRequest(question="x"), rag=None)
        assert order == ["a", "b"]  # responder doesn't append to `order`

    def test_returns_response_dict(self):
        p = Pipeline([_Responder({"answer": "hi", "source": "test"})])
        out = p.run(QueryRequest(question="x"), rag=None)
        assert out["answer"] == "hi"
        assert out["source"] == "test"

    def test_short_circuit_skips_later_stages(self):
        _ShouldNotRun.invoked = False
        p = Pipeline([
            _Responder({"answer": "fast path"}),
            _ShouldNotRun(),
        ])
        out = p.run(QueryRequest(question="x"), rag=None)
        assert out["answer"] == "fast path"
        assert _ShouldNotRun.invoked is False

    def test_empty_pipeline_rejected_at_construction(self):
        with pytest.raises(ValueError):
            Pipeline([])

    def test_no_response_raises(self):
        # If no stage sets state.response, the pipeline is misconfigured.
        # Surfacing this as a clear RuntimeError beats silently returning {}.
        p = Pipeline([_SetField("a", "key", 1)])
        with pytest.raises(RuntimeError, match="without producing a response"):
            p.run(QueryRequest(question="x"), rag=None)

    def test_timings_injected_into_response(self):
        p = Pipeline([
            _SetField("warmup", "k", 1),
            _Responder({"answer": "ok"}),
        ])
        out = p.run(QueryRequest(question="x"), rag=None)
        assert "timings" in out
        # Both the SetField stage AND the Responder show up
        assert "warmup" in out["timings"]
        assert "responder" in out["timings"]

    def test_state_carries_across_stages(self):
        # Stage 2 must see stage 1's writes.
        class _Stage2(PipelineStage):
            name = "s2"

            def run(self, state):
                state.response = {"sum": state.extra_meta["a"] + state.extra_meta["b"]}
                return state

        p = Pipeline([
            _SetField("s1a", "a", 10),
            _SetField("s1b", "b", 32),
            _Stage2(),
        ])
        out = p.run(QueryRequest(question="x"), rag=None)
        assert out["sum"] == 42
