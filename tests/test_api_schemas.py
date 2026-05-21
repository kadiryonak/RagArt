"""Unit tests for src/api/schemas.py — pydantic request şemaları + parse_body.

Şemalar Flask'tan bağımsız test edilir: parse_body() ham bir dict alır,
tipli model döndürür ya da RagArt ValidationError fırlatır.
"""

from __future__ import annotations

import pytest

from src.api.schemas import (
    AskRequest,
    CacheClearRequest,
    CreateWorkspaceRequest,
    DeleteFileRequest,
    UpdateWorkspaceRequest,
    parse_body,
)
from src.exceptions import ValidationError


class TestAskRequest:
    def test_valid(self):
        body = parse_body(AskRequest, {"question": "Algoritma nedir?"})
        assert body.question == "Algoritma nedir?"

    def test_strips_whitespace(self):
        body = parse_body(AskRequest, {"question": "  merhaba  "})
        assert body.question == "merhaba"

    def test_blank_question_rejected(self):
        with pytest.raises(ValidationError):
            parse_body(AskRequest, {"question": "   "})

    def test_missing_question_rejected(self):
        with pytest.raises(ValidationError):
            parse_body(AskRequest, {})


class TestParseBodyEdgeCases:
    def test_none_treated_as_empty_body(self):
        # None gövde boş dict sayılır → zorunlu alan eksik → ValidationError
        with pytest.raises(ValidationError):
            parse_body(AskRequest, None)

    def test_non_dict_body_rejected(self):
        with pytest.raises(ValidationError):
            parse_body(AskRequest, ["not", "a", "dict"])

    def test_validation_error_carries_field_name(self):
        try:
            parse_body(CreateWorkspaceRequest, {"name": ""})
        except ValidationError as e:
            assert "name" in e.detail
        else:
            pytest.fail("ValidationError bekleniyordu")


class TestCreateWorkspaceRequest:
    def test_defaults(self):
        body = parse_body(CreateWorkspaceRequest, {"name": "Proje"})
        assert body.name == "Proje"
        assert body.vector_db == "chroma"
        assert body.description == ""
        assert body.color is None

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            parse_body(CreateWorkspaceRequest, {"name": "   "})

    def test_explicit_fields_preserved(self):
        body = parse_body(CreateWorkspaceRequest, {
            "name": "Araştırma", "color": "#abc123",
            "description": "notlar", "vector_db": "qdrant",
        })
        assert body.color == "#abc123"
        assert body.vector_db == "qdrant"


class TestUpdateWorkspaceRequest:
    def test_all_optional(self):
        body = parse_body(UpdateWorkspaceRequest, {})
        assert body.name is None
        assert body.vector_db is None

    def test_partial_update(self):
        body = parse_body(UpdateWorkspaceRequest, {"name": "Yeni Ad"})
        assert body.name == "Yeni Ad"
        assert body.color is None


class TestDeleteFileRequest:
    def test_valid(self):
        assert parse_body(DeleteFileRequest, {"filename": "a.json"}).filename == "a.json"

    def test_empty_filename_rejected(self):
        with pytest.raises(ValidationError):
            parse_body(DeleteFileRequest, {"filename": ""})


class TestCacheClearRequest:
    def test_default_layer_is_all(self):
        assert parse_body(CacheClearRequest, {}).layer == "all"

    @pytest.mark.parametrize("layer", ["all", "embedding", "response", "semantic"])
    def test_known_layers_accepted(self, layer):
        assert parse_body(CacheClearRequest, {"layer": layer}).layer == layer

    def test_unknown_layer_rejected(self):
        with pytest.raises(ValidationError):
            parse_body(CacheClearRequest, {"layer": "bogus"})
