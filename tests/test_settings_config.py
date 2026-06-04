"""Unit tests for config/settings.py — API key resolution per model type.

Regression guard: get_api_key() once handled deepseek/openai/huggingface
but NOT groq, so MODEL_TYPE=groq silently returned None and the server
fell back to the local model even with a valid key in .env.
"""

from __future__ import annotations

from config.settings import Settings


class TestGetApiKey:
    def test_groq_model_returns_groq_key(self, monkeypatch):
        monkeypatch.setenv("MODEL_TYPE", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_unit_test_key")
        assert Settings().get_api_key() == "gsk_unit_test_key"

    def test_openai_model_returns_openai_key(self, monkeypatch):
        monkeypatch.setenv("MODEL_TYPE", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")
        assert Settings().get_api_key() == "sk-unit-test"

    def test_deepseek_model_returns_deepseek_key(self, monkeypatch):
        monkeypatch.setenv("MODEL_TYPE", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-unit-test")
        assert Settings().get_api_key() == "ds-unit-test"

    def test_local_model_has_no_key(self, monkeypatch):
        monkeypatch.setenv("MODEL_TYPE", "local")
        assert Settings().get_api_key() is None

    def test_groq_model_without_key_returns_none(self, monkeypatch):
        monkeypatch.setenv("MODEL_TYPE", "groq")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert Settings().get_api_key() is None

    def test_huggingface_model_returns_hf_key(self, monkeypatch):
        monkeypatch.setenv("MODEL_TYPE", "huggingface")
        monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_unit_test")
        assert Settings().get_api_key() == "hf_unit_test"

    def test_anthropic_model_returns_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("MODEL_TYPE", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-unit-test")
        assert Settings().get_api_key() == "sk-ant-unit-test"

    def test_unknown_model_returns_none(self, monkeypatch):
        monkeypatch.setenv("MODEL_TYPE", "ollama")
        assert Settings().get_api_key() is None

    def test_default_model_type_is_local(self, monkeypatch):
        monkeypatch.delenv("MODEL_TYPE", raising=False)
        assert Settings().MODEL_TYPE == "local"


class TestValidateApiKey:
    def test_none_is_invalid(self):
        assert Settings.validate_api_key(None) is False

    def test_non_string_is_invalid(self):
        assert Settings.validate_api_key(12345) is False

    def test_valid_deepseek_key(self):
        assert Settings.validate_api_key("sk-" + "a" * 40, "deepseek") is True

    def test_short_deepseek_key_invalid(self):
        assert Settings.validate_api_key("sk-short", "deepseek") is False

    def test_valid_openai_key(self):
        assert Settings.validate_api_key("sk-" + "b" * 45, "openai") is True

    def test_openai_requires_longer_key(self):
        # 35 chars total < 40 → invalid for openai
        assert Settings.validate_api_key("sk-" + "c" * 32, "openai") is False

    def test_unknown_service_is_invalid(self):
        assert Settings.validate_api_key("sk-" + "d" * 40, "groq") is False

    def test_whitespace_is_stripped(self):
        assert Settings.validate_api_key("  sk-" + "e" * 40 + "  ", "deepseek") is True
