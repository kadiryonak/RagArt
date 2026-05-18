"""Unit tests for GroqProvider — HTTP layer fully mocked."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from src.llm_providers import GroqProvider, LLMProviderFactory


def _ok_response(content: str = "Test cevap"):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    return r


class TestGroqGenerate:
    @patch("src.llm_providers.requests.post")
    def test_uses_defaults(self, mock_post):
        mock_post.return_value = _ok_response("Algoritma cevap")
        p = GroqProvider(api_key="gsk_fake")
        out = p.generate("Test prompt")
        assert out == "Algoritma cevap"
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "llama-3.3-70b-versatile"
        assert body["temperature"] == 0.1
        assert body["max_tokens"] == 800
        assert body["top_p"] == 0.9

    @patch("src.llm_providers.requests.post")
    def test_runtime_params_override_defaults(self, mock_post):
        mock_post.return_value = _ok_response()
        p = GroqProvider(api_key="gsk_fake")
        p.generate("Hi", temperature=0.7, max_tokens=200, top_p=0.5)
        body = mock_post.call_args.kwargs["json"]
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 200
        assert body["top_p"] == 0.5

    @patch("src.llm_providers.requests.post")
    def test_partial_override_keeps_other_defaults(self, mock_post):
        mock_post.return_value = _ok_response()
        p = GroqProvider(api_key="gsk_fake")
        p.generate("Hi", temperature=0.5)  # only temperature
        body = mock_post.call_args.kwargs["json"]
        assert body["temperature"] == 0.5
        assert body["max_tokens"] == 800  # default preserved
        assert body["top_p"] == 0.9       # default preserved

    @patch("src.llm_providers.requests.post")
    def test_model_per_call_override(self, mock_post):
        mock_post.return_value = _ok_response()
        p = GroqProvider(api_key="gsk_fake", model="llama-3.3-70b-versatile")
        p.generate("Hi", model="mixtral-8x7b-32768")
        body = mock_post.call_args.kwargs["json"]
        assert body["model"] == "mixtral-8x7b-32768"

    @patch("src.llm_providers.requests.post")
    def test_authorization_header(self, mock_post):
        mock_post.return_value = _ok_response()
        p = GroqProvider(api_key="gsk_secret")
        p.generate("Hi")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer gsk_secret"

    @patch("src.llm_providers.requests.post")
    def test_api_error_returns_message(self, mock_post):
        r = MagicMock()
        r.status_code = 401
        r.text = "Invalid API key"
        mock_post.return_value = r
        p = GroqProvider(api_key="gsk_bad")
        out = p.generate("Hi")
        assert "401" in out

    @patch("src.llm_providers.requests.post")
    def test_connection_error_handled(self, mock_post):
        mock_post.side_effect = ConnectionError("network down")
        p = GroqProvider(api_key="gsk_fake")
        out = p.generate("Hi")
        assert "connection" in out.lower()


class TestGroqGeneralPrompt:
    @patch("src.llm_providers.requests.post")
    def test_general_uses_turkish_template(self, mock_post):
        mock_post.return_value = _ok_response()
        p = GroqProvider(api_key="gsk_fake")
        p.generate_general("Yapay zeka nedir?")
        body = mock_post.call_args.kwargs["json"]
        prompt = body["messages"][0]["content"]
        assert "Türkçe" in prompt
        assert "Yapay zeka nedir?" in prompt


class TestRateLimitRetry:
    @patch("time.sleep", lambda *_: None)  # no actual sleep in tests
    @patch("src.llm_providers.requests.post")
    def test_429_then_200_retries_once(self, mock_post):
        # First call: 429 with "try again in X" hint; second call: 200
        rl = MagicMock()
        rl.status_code = 429
        rl.text = 'Rate limit reached. Please try again in 0.5s.'

        ok = _ok_response("Retry succeeded")
        mock_post.side_effect = [rl, ok]

        p = GroqProvider(api_key="gsk_fake")
        out = p.generate("Hi")
        assert out == "Retry succeeded"
        assert mock_post.call_count == 2

    @patch("time.sleep", lambda *_: None)
    @patch("src.llm_providers.requests.post")
    def test_429_twice_gives_up(self, mock_post):
        # Both calls hit 429
        rl = MagicMock()
        rl.status_code = 429
        rl.text = "Rate limit reached."
        mock_post.return_value = rl

        p = GroqProvider(api_key="gsk_fake")
        out = p.generate("Hi")
        assert "429" in out or "rate limit" in out.lower()
        # Only retries once
        assert mock_post.call_count == 2

    def test_parse_retry_wait(self):
        assert GroqProvider._parse_retry_wait("try again in 12.5s") == 12.5
        assert GroqProvider._parse_retry_wait("try again in 0.123s") == 0.123
        assert GroqProvider._parse_retry_wait("no hint here") is None
        assert GroqProvider._parse_retry_wait("") is None


class TestFactoryWithGroq:
    def test_create_groq_requires_key(self):
        with pytest.raises(ValueError, match="API key"):
            LLMProviderFactory.create("groq")

    def test_create_groq_with_key(self):
        p = LLMProviderFactory.create("groq", api_key="gsk_fake")
        assert isinstance(p, GroqProvider)
        assert p.api_key == "gsk_fake"

    def test_create_groq_with_custom_model(self):
        p = LLMProviderFactory.create("groq", api_key="gsk_fake", model="mixtral-8x7b-32768")
        assert p.model == "mixtral-8x7b-32768"

    def test_groq_in_available_providers(self):
        assert "groq" in LLMProviderFactory.get_available_providers()
