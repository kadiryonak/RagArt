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
