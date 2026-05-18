"""Integration tests for /ask BYOK header flow and /settings/schema endpoint.

Strategy: import app module, monkey-patch rag_system to a stub and force
system_ready=True, then exercise endpoints via Flask test client. No real
RAG init, no real LLM calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def client(monkeypatch):
    """Flask test client with stubbed RAG system."""
    import app as app_module

    captured: dict = {}

    def stub_ask(question, *, k=5, llm_provider=None, llm_params=None,
                 retrieval_strategy=None):
        captured["question"] = question
        captured["k"] = k
        captured["llm_provider"] = llm_provider
        captured["llm_params"] = llm_params
        captured["retrieval_strategy"] = retrieval_strategy
        return {
            "question": question,
            "answer": "stub cevap",
            "source_documents": [],
            "context_used": "",
            "source": "stub",
            "relevance_score": 1.0,
        }

    fake_rag = MagicMock()
    fake_rag.ask.side_effect = stub_ask
    fake_rag.model_type = "stub"

    monkeypatch.setattr(app_module, "rag_system", fake_rag)
    monkeypatch.setattr(app_module, "system_ready", True)

    app_module.app.config["TESTING"] = True
    return app_module.app.test_client(), captured, fake_rag


class TestSettingsSchema:
    def test_endpoint_returns_provider_list(self, client):
        c, _, _ = client
        r = c.get("/settings/schema")
        assert r.status_code == 200
        body = r.get_json()
        ids = {p["id"] for p in body["providers"]}
        assert "groq" in ids
        assert "deepseek" in ids
        assert "ollama" in ids

    def test_schema_has_param_specs(self, client):
        c, _, _ = client
        body = c.get("/settings/schema").get_json()
        ollama = next(p for p in body["providers"] if p["id"] == "ollama")
        assert "num_ctx" in ollama["params"]
        assert ollama["needs_key"] is False
        assert ollama["default_model"] == "llama3.1:8b"

    def test_schema_has_descriptions(self, client):
        c, _, _ = client
        body = c.get("/settings/schema").get_json()
        assert "temperature" in body["param_descriptions_tr"]


class TestAskWithoutHeaders:
    """No BYOK headers → server falls back to default RAG provider."""

    def test_no_provider_no_override(self, client):
        c, captured, _ = client
        r = c.post("/ask", json={"question": "Algoritma nedir?"})
        assert r.status_code == 200
        assert captured["llm_provider"] is None
        # llm_params parsed as empty dict
        assert captured["llm_params"] == {}

    def test_empty_question_rejected(self, client):
        c, _, _ = client
        r = c.post("/ask", json={"question": "  "})
        assert r.status_code == 400


class TestAskWithBYOK:
    def test_groq_provider_override(self, client):
        c, captured, _ = client
        r = c.post(
            "/ask",
            json={"question": "Test"},
            headers={
                "X-Provider": "groq",
                "X-API-Key": "gsk_test_key",
            },
        )
        assert r.status_code == 200
        provider = captured["llm_provider"]
        assert provider is not None
        # GroqProvider instance with the user's key
        from src.llm_providers import GroqProvider
        assert isinstance(provider, GroqProvider)
        assert provider.api_key == "gsk_test_key"

    def test_llm_params_threaded_through(self, client):
        c, captured, _ = client
        c.post(
            "/ask",
            json={"question": "Test"},
            headers={
                "X-Provider": "groq",
                "X-API-Key": "gsk_test",
                "X-LLM-Params": json.dumps({"temperature": 0.7, "max_tokens": 256}),
            },
        )
        assert captured["llm_params"]["temperature"] == 0.7
        assert captured["llm_params"]["max_tokens"] == 256

    def test_invalid_params_rejected(self, client):
        c, _, _ = client
        r = c.post(
            "/ask",
            json={"question": "Test"},
            headers={
                "X-Provider": "groq",
                "X-API-Key": "gsk_test",
                "X-LLM-Params": json.dumps({"temperature": 5.0}),  # out of range
            },
        )
        assert r.status_code == 400
        body = r.get_json()
        assert "Invalid LLM params" in body["error"]
        assert any("temperature" in d for d in body["details"])

    def test_missing_key_for_cloud_provider_rejected(self, client):
        c, _, _ = client
        r = c.post(
            "/ask",
            json={"question": "Test"},
            headers={"X-Provider": "groq"},  # no key!
        )
        assert r.status_code == 400
        assert "API key" in r.get_json()["error"]

    def test_unknown_provider_rejected(self, client):
        c, _, _ = client
        r = c.post(
            "/ask",
            json={"question": "Test"},
            headers={"X-Provider": "claude-secret-model", "X-API-Key": "x"},
        )
        assert r.status_code == 400

    def test_local_provider_no_key_ok(self, client):
        c, captured, _ = client
        r = c.post(
            "/ask",
            json={"question": "Test"},
            headers={"X-Provider": "local"},
        )
        assert r.status_code == 200
        from src.llm_providers import LocalProvider
        assert isinstance(captured["llm_provider"], LocalProvider)

    def test_retrieval_strategy_threaded_through(self, client):
        c, captured, _ = client
        r = c.post(
            "/ask",
            json={"question": "veri yapıları nedir?"},
            headers={"X-Retrieval-Strategy": "hybrid"},
        )
        assert r.status_code == 200
        assert captured["retrieval_strategy"] == "hybrid"

    def test_unknown_retrieval_strategy_falls_back_to_none(self, client):
        c, captured, _ = client
        r = c.post(
            "/ask",
            json={"question": "x"},
            headers={"X-Retrieval-Strategy": "magic-strategy-9000"},
        )
        assert r.status_code == 200
        assert captured["retrieval_strategy"] is None

    def test_ollama_no_key_ok(self, client):
        c, captured, _ = client
        r = c.post(
            "/ask",
            json={"question": "Test"},
            headers={
                "X-Provider": "ollama",
                "X-Model": "llama3.1:8b",
                "X-LLM-Params": json.dumps({"num_ctx": 4096, "repeat_penalty": 1.2}),
            },
        )
        assert r.status_code == 200
        from src.llm_providers import OllamaProvider
        prov = captured["llm_provider"]
        assert isinstance(prov, OllamaProvider)
        assert prov.model == "llama3.1:8b"
        assert captured["llm_params"]["num_ctx"] == 4096
