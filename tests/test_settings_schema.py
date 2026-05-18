"""Unit tests for settings schema, request parsing, and validation."""

from __future__ import annotations

import json

import pytest

from config.settings_schema import (
    LLMParams,
    PROVIDER_DEFAULT_MODELS,
    PROVIDER_NEEDS_KEY,
    PROVIDER_PARAMS,
    RequestSettings,
    get_settings_schema,
    parse_request_settings,
)


class TestLLMParams:
    def test_empty_by_default(self):
        p = LLMParams()
        assert p.to_dict() == {}

    def test_set_fields(self):
        p = LLMParams(temperature=0.5, top_p=0.8)
        assert p.to_dict() == {"temperature": 0.5, "top_p": 0.8}

    def test_to_dict_drops_none(self):
        p = LLMParams(temperature=0.7)
        d = p.to_dict()
        assert "top_p" not in d
        assert d == {"temperature": 0.7}

    def test_to_dict_can_keep_none(self):
        p = LLMParams(temperature=0.5)
        d = p.to_dict(drop_none=False)
        assert d["top_p"] is None

    def test_from_json_string_valid(self):
        p = LLMParams.from_json_string('{"temperature": 0.3, "num_ctx": 4096}')
        assert p.temperature == 0.3
        assert p.num_ctx == 4096

    def test_from_json_string_none(self):
        assert LLMParams.from_json_string(None).to_dict() == {}
        assert LLMParams.from_json_string("").to_dict() == {}

    def test_from_json_invalid_returns_empty(self):
        assert LLMParams.from_json_string("not json").to_dict() == {}
        assert LLMParams.from_json_string("[1,2,3]").to_dict() == {}  # not a dict

    def test_unknown_fields_ignored(self):
        p = LLMParams.from_json_string('{"temperature": 0.5, "evil_field": 999}')
        assert p.temperature == 0.5
        assert not hasattr(p, "evil_field")


class TestValidation:
    def test_valid_deepseek_params(self):
        p = LLMParams(temperature=0.5, max_tokens=500)
        assert p.validate("deepseek") == []

    def test_temperature_out_of_range(self):
        p = LLMParams(temperature=3.0)
        errors = p.validate("deepseek")
        assert len(errors) == 1
        assert "temperature" in errors[0]

    def test_negative_temperature(self):
        p = LLMParams(temperature=-0.1)
        errors = p.validate("deepseek")
        assert len(errors) == 1

    def test_unknown_param_silently_ignored(self):
        # num_ctx is Ollama-only; deepseek silently ignores it (no error)
        p = LLMParams(num_ctx=2048)
        assert p.validate("deepseek") == []

    def test_ollama_specific_validates(self):
        p = LLMParams(num_ctx=2048, repeat_penalty=1.2)
        assert p.validate("ollama") == []

    def test_int_field_rejects_float(self):
        # num_ctx must be int
        p = LLMParams(num_ctx=2048.5)  # type: ignore
        errors = p.validate("ollama")
        assert any("num_ctx" in e for e in errors)

    def test_multiple_errors(self):
        p = LLMParams(temperature=5.0, top_p=2.0)
        errors = p.validate("deepseek")
        assert len(errors) == 2


class _MockHeaders:
    def __init__(self, **kw):
        self._d = kw

    def get(self, key, default=None):
        return self._d.get(key, default)


class TestRequestParsing:
    def test_empty_headers(self):
        s = parse_request_settings(_MockHeaders())
        assert s.provider is None
        assert s.api_key is None
        assert s.model is None
        assert s.llm_params.to_dict() == {}

    def test_provider_normalized(self):
        s = parse_request_settings(_MockHeaders(**{"X-Provider": " GroQ "}))
        assert s.provider == "groq"

    def test_api_key_stripped(self):
        s = parse_request_settings(_MockHeaders(**{"X-API-Key": "  sk-abc  "}))
        assert s.api_key == "sk-abc"

    def test_empty_strings_become_none(self):
        s = parse_request_settings(_MockHeaders(**{
            "X-Provider": "",
            "X-API-Key": "   ",
            "X-Model": "",
        }))
        assert s.provider is None
        assert s.api_key is None
        assert s.model is None

    def test_llm_params_parsed(self):
        s = parse_request_settings(_MockHeaders(**{
            "X-LLM-Params": json.dumps({"temperature": 0.3, "top_p": 0.95}),
        }))
        assert s.llm_params.temperature == 0.3
        assert s.llm_params.top_p == 0.95

    def test_full_request(self):
        s = parse_request_settings(_MockHeaders(**{
            "X-Provider": "groq",
            "X-API-Key": "gsk_test",
            "X-Model": "llama-3.3-70b-versatile",
            "X-LLM-Params": json.dumps({"temperature": 0.2, "max_tokens": 1000}),
        }))
        assert s.provider == "groq"
        assert s.api_key == "gsk_test"
        assert s.model == "llama-3.3-70b-versatile"
        assert s.llm_params.temperature == 0.2
        assert s.llm_params.max_tokens == 1000


class TestSchemaShape:
    def test_all_providers_have_default_model(self):
        for pid in PROVIDER_PARAMS:
            assert pid in PROVIDER_DEFAULT_MODELS
            assert pid in PROVIDER_NEEDS_KEY

    def test_schema_exports_all_providers(self):
        schema = get_settings_schema()
        ids = {p["id"] for p in schema["providers"]}
        assert {"deepseek", "openai", "groq", "ollama", "huggingface", "local"} == ids

    def test_schema_has_descriptions(self):
        schema = get_settings_schema()
        assert "temperature" in schema["param_descriptions_tr"]
        assert "num_ctx" in schema["param_descriptions_tr"]

    def test_ollama_has_local_params(self):
        ollama_params = PROVIDER_PARAMS["ollama"]
        assert "num_ctx" in ollama_params
        assert "num_predict" in ollama_params
        assert "repeat_penalty" in ollama_params

    def test_groq_inherits_common_params(self):
        groq_params = PROVIDER_PARAMS["groq"]
        assert "temperature" in groq_params
        assert "top_p" in groq_params
        assert "max_tokens" in groq_params

    def test_schema_has_retrieval_strategies(self):
        schema = get_settings_schema()
        assert "retrieval_strategies" in schema
        ids = {s["id"] for s in schema["retrieval_strategies"]}
        assert {"auto", "dense", "sparse", "hybrid"} == ids


class TestRetrievalParsing:
    def test_valid_strategy(self):
        s = parse_request_settings(_MockHeaders(**{"X-Retrieval-Strategy": "hybrid"}))
        assert s.retrieval_strategy == "hybrid"

    def test_case_insensitive(self):
        s = parse_request_settings(_MockHeaders(**{"X-Retrieval-Strategy": "  HYBRID  "}))
        assert s.retrieval_strategy == "hybrid"

    def test_unknown_silently_dropped(self):
        s = parse_request_settings(_MockHeaders(**{"X-Retrieval-Strategy": "magic"}))
        assert s.retrieval_strategy is None

    def test_no_header_means_none(self):
        s = parse_request_settings(_MockHeaders())
        assert s.retrieval_strategy is None
