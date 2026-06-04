"""Supplementary unit tests for src/llm_providers.py — branches not covered
by test_llm_providers.py / test_groq_provider.py: generate_general for each
provider, error/connection paths, HuggingFace provider, the LocalProvider
contextual templating, the BaseLLMProvider.generate_stream default, and the
_merge helper."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import requests

from src.llm_providers import (
    AnthropicProvider,
    BaseLLMProvider,
    DeepSeekProvider,
    GroqProvider,
    HuggingFaceProvider,
    LLMProviderFactory,
    LocalProvider,
    OllamaProvider,
    OpenAIProvider,
    _merge,
)


def _ok(content):
    r = Mock()
    r.status_code = 200
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    return r


class _FakeSSE:
    """Mimics a streaming requests.Response, including the real charset
    pitfall: requests defaults `encoding` to ISO-8859-1 for charset-less
    text/event-stream, and iter_lines(decode_unicode=True) honours it. The
    providers must pin UTF-8 or Turkish characters become mojibake."""

    def __init__(self, utf8_lines):
        self.status_code = 200
        self.encoding = "ISO-8859-1"  # what requests picks for SSE
        self._raw = [ln.encode("utf-8") for ln in utf8_lines]

    def iter_lines(self, decode_unicode=False):
        for b in self._raw:
            yield b.decode(self.encoding) if decode_unicode else b


class TestMergeHelper:
    def test_overrides_replace_defaults(self):
        assert _merge({"a": 1, "b": 2}, {"b": 9}) == {"a": 1, "b": 9}

    def test_none_overrides_ignored(self):
        assert _merge({"a": 1}, {"a": None}) == {"a": 1}

    def test_new_keys_added(self):
        assert _merge({"a": 1}, {"c": 3}) == {"a": 1, "c": 3}


class TestGenerateGeneral:
    @patch("requests.post")
    def test_deepseek_general_wraps_question(self, mock_post):
        mock_post.return_value = _ok("genel cevap")
        out = DeepSeekProvider(api_key="k").generate_general("Soru?")
        assert out == "genel cevap"
        sent = mock_post.call_args.kwargs["json"]["messages"][0]["content"]
        assert "Soru?" in sent

    @patch("requests.post")
    def test_openai_general(self, mock_post):
        mock_post.return_value = _ok("oai")
        assert OpenAIProvider(api_key="k").generate_general("S?") == "oai"

    @patch("requests.post")
    def test_ollama_general(self, mock_post):
        r = Mock(status_code=200)
        r.json.return_value = {"response": "oll"}
        mock_post.return_value = r
        assert OllamaProvider().generate_general("S?") == "oll"


class TestOpenAIErrors:
    @patch("requests.post")
    def test_api_error_status(self, mock_post):
        mock_post.return_value = Mock(status_code=503)
        out = OpenAIProvider(api_key="k").generate("p")
        assert "OpenAI API error: 503" in out

    @patch("requests.post")
    def test_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("down")
        out = OpenAIProvider(api_key="k").generate("p")
        assert "connection error" in out.lower()

    @patch("requests.post")
    def test_model_override_passed(self, mock_post):
        mock_post.return_value = _ok("x")
        OpenAIProvider(api_key="k").generate("p", model="gpt-4o")
        assert mock_post.call_args.kwargs["json"]["model"] == "gpt-4o"


class TestOllamaErrors:
    @patch("requests.post")
    def test_api_error_status(self, mock_post):
        mock_post.return_value = Mock(status_code=500)
        assert "Ollama error: 500" in OllamaProvider().generate("p")

    @patch("requests.post")
    def test_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("nope")
        assert "connection failed" in OllamaProvider().generate("p").lower()


class TestHuggingFaceProvider:
    def test_initialization(self):
        p = HuggingFaceProvider(api_key="hf")
        assert p.api_key == "hf"
        assert "Llama" in p.model

    @patch("requests.post")
    def test_generate_success(self, mock_post):
        mock_post.return_value = _ok("  hf cevap  ")
        assert HuggingFaceProvider(api_key="hf").generate("p") == "hf cevap"

    @patch("requests.post")
    def test_no_choices_returns_message(self, mock_post):
        r = Mock(status_code=200)
        r.json.return_value = {"choices": []}
        mock_post.return_value = r
        assert HuggingFaceProvider(api_key="hf").generate("p") == "No response generated."

    @patch("requests.post")
    def test_credit_limit_402(self, mock_post):
        mock_post.return_value = Mock(status_code=402)
        out = HuggingFaceProvider(api_key="hf").generate("p")
        assert "kredi limiti" in out

    @patch("requests.post")
    def test_other_error_status(self, mock_post):
        r = Mock(status_code=500)
        r.text = "server boom"
        mock_post.return_value = r
        out = HuggingFaceProvider(api_key="hf").generate("p")
        assert "HuggingFace API error: 500" in out

    @patch("requests.post")
    def test_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("x")
        assert "HuggingFace error" in HuggingFaceProvider(api_key="hf").generate("p")

    @patch("requests.post")
    def test_generate_general(self, mock_post):
        mock_post.return_value = _ok("g")
        assert HuggingFaceProvider(api_key="hf").generate_general("S?") == "g"


class TestAnthropicProvider:
    def _ok(self, text):
        r = Mock(status_code=200)
        r.json.return_value = {"content": [{"type": "text", "text": text}]}
        return r

    def test_initialization(self):
        p = AnthropicProvider(api_key="sk-ant-xxx")
        assert p.api_key == "sk-ant-xxx"
        assert "claude" in p.model
        assert p.defaults["max_tokens"] == 1024  # required by the API

    @patch("requests.post")
    def test_generate_success(self, mock_post):
        mock_post.return_value = self._ok("Merhaba, ben Claude.")
        out = AnthropicProvider(api_key="k").generate("Selam")
        assert out == "Merhaba, ben Claude."
        # Uses x-api-key + anthropic-version headers (not Bearer)
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "k"
        assert "anthropic-version" in headers
        # max_tokens is always sent
        assert "max_tokens" in mock_post.call_args.kwargs["json"]

    @patch("requests.post")
    def test_generate_joins_text_blocks(self, mock_post):
        r = Mock(status_code=200)
        r.json.return_value = {"content": [
            {"type": "text", "text": "Parça1 "},
            {"type": "tool_use", "id": "x"},        # non-text ignored
            {"type": "text", "text": "Parça2"},
        ]}
        mock_post.return_value = r
        assert AnthropicProvider(api_key="k").generate("p") == "Parça1 Parça2"

    @patch("requests.post")
    def test_api_error_status(self, mock_post):
        r = Mock(status_code=401); r.text = "unauthorized"
        mock_post.return_value = r
        assert "Anthropic API error: 401" in AnthropicProvider(api_key="bad").generate("p")

    @patch("requests.post")
    def test_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError("down")
        assert "connection error" in AnthropicProvider(api_key="k").generate("p").lower()

    @patch("requests.post")
    def test_model_override(self, mock_post):
        mock_post.return_value = self._ok("ok")
        AnthropicProvider(api_key="k").generate("p", model="claude-opus-4-8")
        assert mock_post.call_args.kwargs["json"]["model"] == "claude-opus-4-8"

    @patch("requests.post")
    def test_generate_general(self, mock_post):
        mock_post.return_value = self._ok("genel")
        assert AnthropicProvider(api_key="k").generate_general("Soru?") == "genel"

    @patch("requests.post")
    def test_stream_parses_text_deltas(self, mock_post):
        lines = [
            'data: {"type":"message_start"}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Mer"}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"haba"}}',
            'data: {"type":"message_stop"}',
        ]
        r = Mock(status_code=200)
        r.iter_lines.return_value = iter(lines)
        mock_post.return_value = r
        chunks = list(AnthropicProvider(api_key="k").generate_stream("p"))
        assert chunks == ["Mer", "haba"]

    @patch("requests.post")
    def test_stream_falls_back_on_non_200(self, mock_post):
        bad = Mock(status_code=529)  # overloaded
        ok = self._ok("tam cevap")
        mock_post.side_effect = [bad, ok]  # stream fails → generate() fallback
        chunks = list(AnthropicProvider(api_key="k").generate_stream("p"))
        assert chunks == ["tam cevap"]

    @patch("requests.post")
    def test_stream_decodes_turkish_utf8(self, mock_post):
        # Regression: SSE without charset → requests' ISO-8859-1 default →
        # "çözmek" arrived as "Ã§Ã¶zmek". Provider must pin UTF-8.
        mock_post.return_value = _FakeSSE([
            'data: {"type":"content_block_delta","delta":'
            '{"type":"text_delta","text":"çözmek"}}',
            'data: {"type":"message_stop"}',
        ])
        chunks = list(AnthropicProvider(api_key="k").generate_stream("p"))
        assert chunks == ["çözmek"]


class TestGroqStreamEncoding:
    @patch("requests.post")
    def test_stream_decodes_turkish_utf8(self, mock_post):
        mock_post.return_value = _FakeSSE([
            'data: {"choices":[{"delta":{"content":"gerçekleştir"}}]}',
            "data: [DONE]",
        ])
        chunks = list(GroqProvider(api_key="k").generate_stream("p"))
        assert chunks == ["gerçekleştir"]


class TestLocalProviderContextual:
    def _prompt(self, context, question):
        return f"BAĞLAM:\n{context}\n\nSORU: {question}\n\nYANITIN:"

    def test_invalid_format_returns_error(self):
        assert "format" in LocalProvider().generate("no markers").lower()

    def test_empty_context_message(self):
        out = LocalProvider().generate(self._prompt("", "Algoritma nedir?"))
        assert "yeterli detay" in out

    def test_algorithm_keyword_branch(self):
        ctx = "Algoritma bir problem çözme adımıdır.\nBaşka bir satır."
        out = LocalProvider().generate(self._prompt(ctx, "Algoritma nedir?"))
        assert "Algoritma" in out

    def test_definition_branch_for_nedir(self):
        ctx = "Python: yüksek seviye bir dildir\nFlask: bir web framework"
        out = LocalProvider().generate(self._prompt(ctx, "Python nedir?"))
        assert "Tanım" in out or "Python" in out

    def test_hangi_branch_lists_items(self):
        ctx = "- birinci madde\n- ikinci madde\n- üçüncü madde"
        out = LocalProvider().generate(self._prompt(ctx, "Hangi maddeler var?"))
        assert "madde" in out.lower()

    def test_fallback_relevant_lines(self):
        ctx = "Bu yeterince uzun bir bilgi satırıdır ve anlam taşır."
        out = LocalProvider().generate(self._prompt(ctx, "Bana bunu söyle"))
        assert len(out) > 0

    def test_generate_general_mentions_api_key(self):
        assert "API" in LocalProvider().generate_general("herhangi")


class TestBaseStreamDefault:
    def test_default_stream_yields_single_chunk(self):
        class Tmp(BaseLLMProvider):
            def generate(self, prompt, **p):
                return "tek parça"

            def generate_general(self, q, **p):
                return "g"

        chunks = list(Tmp().generate_stream("p"))
        assert chunks == ["tek parça"]


class TestFactoryExtra:
    def test_create_groq_requires_key(self):
        with pytest.raises(ValueError, match="requires an API key"):
            LLMProviderFactory.create("groq")

    def test_create_huggingface_with_model(self):
        p = LLMProviderFactory.create("huggingface", api_key="hf", model="custom/model")
        assert isinstance(p, HuggingFaceProvider)
        assert p.model == "custom/model"

    def test_create_ollama_with_model(self):
        p = LLMProviderFactory.create("ollama", model="mistral:7b")
        assert isinstance(p, OllamaProvider)
        assert p.model == "mistral:7b"

    def test_get_available_includes_groq_and_hf(self):
        avail = LLMProviderFactory.get_available_providers()
        assert "groq" in avail and "huggingface" in avail

    def test_create_anthropic_requires_key(self):
        with pytest.raises(ValueError, match="requires an API key"):
            LLMProviderFactory.create("anthropic")

    def test_create_anthropic_with_key_and_model(self):
        p = LLMProviderFactory.create("anthropic", api_key="k", model="claude-opus-4-8")
        assert isinstance(p, AnthropicProvider)
        assert p.model == "claude-opus-4-8"
