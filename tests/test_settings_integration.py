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
                 retrieval_strategy=None, rerank=False, rerank_fetch_k=20,
                 history=None, memory_strategy=None,
                 deduplicate_context=False, reorder_context=False,
                 max_context_tokens=None,
                 allow_general_knowledge_fallback=False,
                 prompt_strategy=None, custom_role=None,
                 custom_prompt_template=None):
        captured["question"] = question
        captured["k"] = k
        captured["llm_provider"] = llm_provider
        captured["llm_params"] = llm_params
        captured["retrieval_strategy"] = retrieval_strategy
        captured["rerank"] = rerank
        captured["rerank_fetch_k"] = rerank_fetch_k
        captured["history"] = history or []
        captured["memory_strategy"] = memory_strategy
        captured["deduplicate_context"] = deduplicate_context
        captured["reorder_context"] = reorder_context
        captured["max_context_tokens"] = max_context_tokens
        captured["allow_general_knowledge_fallback"] = allow_general_knowledge_fallback
        captured["prompt_strategy"] = prompt_strategy
        captured["custom_role"] = custom_role
        captured["custom_prompt_template"] = custom_prompt_template
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

    # Inject the fake into the workspace cache so any X-Workspace-Id
    # (or the default) resolves to this stub instead of trying to build
    # a real RAG (which would load the embedding model on every test).
    from src.workspaces import DEFAULT_WORKSPACE_ID
    app_module._rag_cache.clear()
    app_module._rag_cache[DEFAULT_WORKSPACE_ID] = fake_rag

    def get_rag_stub(_ws_id):
        return fake_rag

    monkeypatch.setattr(app_module, "get_rag_for", get_rag_stub)
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

    def test_rerank_header_enables_reranker(self, client):
        c, captured, _ = client
        r = c.post(
            "/ask",
            json={"question": "x"},
            headers={"X-Rerank": "true"},
        )
        assert r.status_code == 200
        assert captured["rerank"] is True

    def test_rerank_header_false_keeps_off(self, client):
        c, captured, _ = client
        r = c.post(
            "/ask",
            json={"question": "x"},
            headers={"X-Rerank": "0"},
        )
        assert r.status_code == 200
        assert captured["rerank"] is False

    def test_rerank_fetch_k_threaded(self, client):
        c, captured, _ = client
        c.post(
            "/ask",
            json={"question": "x"},
            headers={"X-Rerank": "true", "X-Rerank-Fetch-K": "50"},
        )
        assert captured["rerank_fetch_k"] == 50

    def test_rerank_fetch_k_clamped(self, client):
        c, captured, _ = client
        c.post(
            "/ask",
            json={"question": "x"},
            headers={"X-Rerank": "true", "X-Rerank-Fetch-K": "9999"},
        )
        assert captured["rerank_fetch_k"] == 200  # max clamp

    def test_memory_strategy_threaded(self, client):
        c, captured, _ = client
        r = c.post(
            "/ask",
            json={"question": "ondan bahsetmiştik"},
            headers={
                "X-Memory-Strategy": "sliding_window",
                "X-Conversation-History": json.dumps([
                    {"role": "user", "content": "algoritma nedir"},
                    {"role": "assistant", "content": "algoritma adım adım..."},
                ]),
            },
        )
        assert r.status_code == 200
        assert captured["memory_strategy"] == "sliding_window"
        assert len(captured["history"]) == 2
        from src.memory import ConversationTurn
        assert isinstance(captured["history"][0], ConversationTurn)
        assert captured["history"][0].role == "user"

    def test_unknown_memory_strategy_falls_back(self, client):
        c, captured, _ = client
        r = c.post(
            "/ask",
            json={"question": "x"},
            headers={"X-Memory-Strategy": "telepathic-memory"},
        )
        assert r.status_code == 200
        assert captured["memory_strategy"] is None

    def test_invalid_history_json_ignored(self, client):
        c, captured, _ = client
        r = c.post(
            "/ask",
            json={"question": "x"},
            headers={"X-Conversation-History": "not json"},
        )
        assert r.status_code == 200
        assert captured["history"] == []

    def test_context_dedup_header(self, client):
        c, captured, _ = client
        r = c.post("/ask", json={"question": "x"},
                   headers={"X-Context-Deduplicate": "true"})
        assert r.status_code == 200
        assert captured["deduplicate_context"] is True

    def test_context_reorder_header(self, client):
        c, captured, _ = client
        r = c.post("/ask", json={"question": "x"},
                   headers={"X-Context-Reorder": "1"})
        assert r.status_code == 200
        assert captured["reorder_context"] is True

    def test_context_max_tokens_header(self, client):
        c, captured, _ = client
        r = c.post("/ask", json={"question": "x"},
                   headers={"X-Context-Max-Tokens": "1500"})
        assert r.status_code == 200
        assert captured["max_context_tokens"] == 1500

    def test_context_max_tokens_clamped(self, client):
        c, captured, _ = client
        c.post("/ask", json={"question": "x"},
               headers={"X-Context-Max-Tokens": "99999"})
        assert captured["max_context_tokens"] == 32000

    def test_context_flags_default_off(self, client):
        c, captured, _ = client
        c.post("/ask", json={"question": "x"})
        assert captured["deduplicate_context"] is False
        assert captured["reorder_context"] is False
        assert captured["max_context_tokens"] is None

    def test_general_knowledge_fallback_default_off(self, client):
        """SAFETY: must default to False — protects against hallucination."""
        c, captured, _ = client
        c.post("/ask", json={"question": "x"})
        assert captured["allow_general_knowledge_fallback"] is False

    def test_general_knowledge_fallback_opt_in(self, client):
        c, captured, _ = client
        c.post(
            "/ask",
            json={"question": "x"},
            headers={"X-Allow-General-Knowledge": "true"},
        )
        assert captured["allow_general_knowledge_fallback"] is True

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
