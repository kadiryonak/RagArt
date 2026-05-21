"""Tests for the streaming path — provider generate_stream() + SSE parsing.

ask_stream() orchestration reuses the already-tested pipeline stages; the
streaming-specific surface that needs its own coverage is (a) the
provider's generate_stream and (b) the SSE wiring (see
test_settings_integration.TestAskStream).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class _FakeStreamResponse:
    """Stand-in for a `requests` streaming Response."""

    def __init__(self, status_code, lines):
        self.status_code = status_code
        self._lines = lines
        self.text = "rate limited" if status_code != 200 else ""

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)


class TestBaseProviderStreamFallback:
    def test_default_generate_stream_yields_one_chunk(self):
        from src.llm_providers import BaseLLMProvider

        class Dummy(BaseLLMProvider):
            def generate(self, prompt, **p):
                return "tam cevap"

            def generate_general(self, q, **p):
                return "x"

        # A provider without native streaming still works: one whole chunk.
        assert list(Dummy().generate_stream("soru")) == ["tam cevap"]


class TestGroqStreaming:
    def _provider(self):
        from src.llm_providers import GroqProvider
        return GroqProvider(api_key="test-key")

    def test_parses_sse_token_deltas(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"Mer"}}]}',
            'data: {"choices":[{"delta":{"content":"haba"}}]}',
            'data: {"choices":[{"delta":{}}]}',          # no content → skipped
            "data: [DONE]",
        ]
        with patch("src.llm_providers.requests.post",
                   return_value=_FakeStreamResponse(200, lines)):
            chunks = list(self._provider().generate_stream("selam"))
        assert chunks == ["Mer", "haba"]

    def test_non_200_falls_back_to_generate(self):
        prov = self._provider()
        with patch("src.llm_providers.requests.post",
                   return_value=_FakeStreamResponse(429, [])), \
             patch.object(prov, "generate", return_value="fallback cevap"):
            chunks = list(prov.generate_stream("selam"))
        # A 429 must not break streaming — fall back to the retry-aware path.
        assert chunks == ["fallback cevap"]

    def test_ignores_malformed_sse_lines(self):
        lines = [
            "garbage with no data prefix",
            "data: not-json",
            'data: {"choices":[{"delta":{"content":"OK"}}]}',
        ]
        with patch("src.llm_providers.requests.post",
                   return_value=_FakeStreamResponse(200, lines)):
            chunks = list(self._provider().generate_stream("x"))
        assert chunks == ["OK"]

    def test_connection_error_yields_error_chunk(self):
        with patch("src.llm_providers.requests.post",
                   side_effect=RuntimeError("boom")):
            chunks = list(self._provider().generate_stream("x"))
        assert len(chunks) == 1 and "error" in chunks[0].lower()
