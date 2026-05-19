"""Unit tests for prompt strategies + factory + integration via /ask."""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock

import pytest

from src.prompt_strategies import (
    BasePromptStrategy,
    ChainOfThoughtStrategy,
    CustomStrategy,
    DirectStrategy,
    FewShotStrategy,
    MultiQueryStrategy,
    PromptStrategyFactory,
    RoleBasedStrategy,
    StrategyContext,
)


# ----- Factory -----


class TestFactory:
    def test_registered_strategies_exist(self):
        keys = {s["id"] for s in PromptStrategyFactory.available()}
        assert {"direct", "chain_of_thought", "few_shot",
                "role_based", "custom", "multi_query"}.issubset(keys)

    def test_is_available(self):
        assert PromptStrategyFactory.is_available("direct")
        assert not PromptStrategyFactory.is_available("nonsense")

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError):
            PromptStrategyFactory.create("nonsense")

    def test_create_returns_correct_class(self):
        assert isinstance(PromptStrategyFactory.create("direct"), DirectStrategy)
        assert isinstance(PromptStrategyFactory.create("multi_query"), MultiQueryStrategy)


# ----- Direct -----


class TestDirect:
    def test_basic_prompt(self):
        s = DirectStrategy()
        p = s.build_prompt(question="Algoritma nedir?", context="ctx", memory_context="")
        assert "Algoritma nedir?" in p
        assert "ctx" in p
        assert "üçüncü tekil şahıs" in p.lower()

    def test_memory_template_used(self):
        s = DirectStrategy()
        p = s.build_prompt(question="x?", context="c", memory_context="prev turn")
        assert "ÖNCEKİ KONUŞMA" in p
        assert "prev turn" in p

    def test_no_memory_template_when_empty(self):
        s = DirectStrategy()
        p = s.build_prompt(question="x?", context="c", memory_context="")
        assert "ÖNCEKİ KONUŞMA" not in p


# ----- Chain of Thought -----


class TestCoT:
    def test_extracts_answer_section(self):
        s = ChainOfThoughtStrategy()
        raw = "MUHAKEME: Bağlamda X var.\nYANIT: Bu konu hakkında Y."
        assert s._extract_answer(raw) == "Bu konu hakkında Y."

    def test_fallback_when_no_section(self):
        s = ChainOfThoughtStrategy()
        assert s._extract_answer("Sadece düz cevap.") == "Sadece düz cevap."

    def test_execute_runs_one_llm_call_and_extracts(self):
        s = ChainOfThoughtStrategy()
        llm = MagicMock()
        llm.generate.return_value = "MUHAKEME: ...\nYANIT: nihai cevap"
        ctx = StrategyContext(llm=llm, retrieve_fn=lambda q, k: [])
        out = s.execute(ctx, question="q", context="c")
        assert out == "nihai cevap"
        assert llm.generate.call_count == 1


# ----- Few-shot -----


class TestFewShot:
    def test_includes_examples(self):
        s = FewShotStrategy()
        p = s.build_prompt(question="X?", context="Y")
        assert "Örnek 1" in p
        assert "Algoritma" in p  # default example mentions it

    def test_custom_examples(self):
        s = FewShotStrategy(examples=[("ctx", "q", "a")])
        p = s.build_prompt(question="X?", context="Y")
        assert "ctx" in p
        assert "q" in p
        assert "a" in p


# ----- Role-based -----


class TestRoleBased:
    def test_default_role(self):
        s = RoleBasedStrategy()
        p = s.build_prompt(question="q", context="c")
        assert "uzman" in p.lower()

    def test_custom_role(self):
        s = RoleBasedStrategy(role="20 yıllık hukuk danışmanı")
        p = s.build_prompt(question="q", context="c")
        assert "hukuk danışmanı" in p

    def test_role_length_capped(self):
        s = RoleBasedStrategy(role="x" * 5000)
        assert len(s.role) <= 1500


# ----- Custom -----


class TestCustom:
    def test_uses_template(self):
        s = CustomStrategy(template="Q={question} C={context}")
        p = s.build_prompt(question="abc", context="def")
        assert p == "Q=abc C=def"

    def test_fallback_when_template_empty(self):
        s = CustomStrategy(template="")
        p = s.build_prompt(question="q", context="c")
        assert "q" in p and "c" in p

    def test_unknown_placeholder_surfaces_error(self):
        s = CustomStrategy(template="Q={question} OS={os}")
        out = s.build_prompt(question="q", context="c")
        assert "Custom prompt template error" in out
        assert "{os}" in out or "'os'" in out


# ----- Multi-Query -----


class TestMultiQuery:
    def test_parses_numbered_list(self):
        s = MultiQueryStrategy(n_variants=3)
        llm = MagicMock()
        llm.generate.return_value = "1. varyant bir\n2. varyant iki\n3. varyant üç"
        ctx = StrategyContext(llm=llm, retrieve_fn=lambda q, k: [])
        variants = s.generate_query_variations("orijinal soru", ctx)
        assert variants[0] == "orijinal soru"
        assert "varyant bir" in variants
        assert "varyant iki" in variants
        assert "varyant üç" in variants

    def test_handles_llm_failure_gracefully(self):
        s = MultiQueryStrategy()
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("boom")
        ctx = StrategyContext(llm=llm, retrieve_fn=lambda q, k: [])
        # Must not raise — fall back to single query
        out = s.generate_query_variations("q", ctx)
        assert out == ["q"]

    def test_n_variants_clamped(self):
        s = MultiQueryStrategy(n_variants=99)
        assert s.n_variants == 8  # max clamp
        s = MultiQueryStrategy(n_variants=0)
        assert s.n_variants == 1

    def test_is_multi_query_flag(self):
        assert MultiQueryStrategy().is_multi_query
        assert not DirectStrategy().is_multi_query


# ----- Integration through /ask -----


@pytest.fixture
def api_client(monkeypatch):
    """Re-uses the workspace-aware fixture pattern from test_settings_integration."""
    import app as app_module
    captured = {}

    def stub_ask(question, **kwargs):
        captured.update(kwargs)
        captured["question"] = question
        return {
            "question": question, "answer": "ok",
            "source_documents": [], "context_used": "",
            "source": "stub", "relevance_score": 1.0,
        }

    fake_rag = MagicMock()
    fake_rag.ask.side_effect = stub_ask
    fake_rag.model_type = "stub"
    from src.workspaces import DEFAULT_WORKSPACE_ID
    app_module._rag_cache.clear()
    app_module._rag_cache[DEFAULT_WORKSPACE_ID] = fake_rag
    monkeypatch.setattr(app_module, "get_rag_for", lambda _: fake_rag)
    monkeypatch.setattr(app_module, "system_ready", True)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client(), captured


class TestAskHeaders:
    def test_prompt_strategy_threaded(self, api_client):
        c, captured = api_client
        r = c.post("/ask", json={"question": "x"},
                   headers={"X-Prompt-Strategy": "chain_of_thought"})
        assert r.status_code == 200
        assert captured["prompt_strategy"] == "chain_of_thought"

    def test_custom_role_base64_decoded(self, api_client):
        c, captured = api_client
        role = "Sen Türkçe konuşan bir hukuk danışmanısın."
        encoded = "b64:" + base64.b64encode(role.encode("utf-8")).decode("ascii")
        r = c.post("/ask", json={"question": "x"},
                   headers={"X-Prompt-Strategy": "role_based",
                            "X-Custom-Role": encoded})
        assert r.status_code == 200
        assert captured["custom_role"] == role

    def test_custom_template_base64_decoded(self, api_client):
        c, captured = api_client
        tmpl = "BAĞLAM: {context}\nSORU: {question}\nYANIT:"
        encoded = "b64:" + base64.b64encode(tmpl.encode("utf-8")).decode("ascii")
        r = c.post("/ask", json={"question": "x"},
                   headers={"X-Prompt-Strategy": "custom",
                            "X-Custom-Prompt": encoded})
        assert r.status_code == 200
        assert captured["custom_prompt_template"] == tmpl

    def test_defaults_when_no_header(self, api_client):
        c, captured = api_client
        c.post("/ask", json={"question": "x"})
        assert captured["prompt_strategy"] is None
        assert captured["custom_role"] is None
        assert captured["custom_prompt_template"] is None

    def test_schema_endpoint_lists_strategies(self, api_client):
        c, _ = api_client
        body = c.get("/settings/schema").get_json()
        assert "prompt_strategies" in body
        ids = {s["id"] for s in body["prompt_strategies"]}
        assert "direct" in ids
        assert "multi_query" in ids
