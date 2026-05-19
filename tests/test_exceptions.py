"""Tests for the RagArt exception hierarchy.

Mostly self-validating contract checks — every subclass must define
http_status + user_message_tr; to_response() must be JSON-serialisable.
"""

from __future__ import annotations

import json
import logging

import pytest

from src.exceptions import (
    CacheError,
    ConfigError,
    DefaultWorkspaceProtectedError,
    EmptyKnowledgeBaseError,
    GuardBlockedError,
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    PathTraversalError,
    RagArtError,
    RetrievalError,
    StaleIndexError,
    ValidationError,
    WorkspaceError,
    WorkspaceNotFoundError,
)


ALL_LEAF_EXCEPTIONS = [
    WorkspaceNotFoundError,
    DefaultWorkspaceProtectedError,
    StaleIndexError,
    EmptyKnowledgeBaseError,
    LLMRateLimitError,
    LLMAuthError,
    CacheError,
    GuardBlockedError,
    PathTraversalError,
    ValidationError,
    ConfigError,
]


class TestBaseContract:
    def test_base_is_an_exception(self):
        assert issubclass(RagArtError, Exception)

    def test_default_http_status_is_500(self):
        assert RagArtError.http_status == 500

    def test_default_log_level_is_error(self):
        assert RagArtError.log_level == logging.ERROR

    def test_to_response_is_json_serialisable(self):
        e = RagArtError("internal detail")
        body = e.to_response()
        json.dumps(body, ensure_ascii=False)  # raises if not serialisable
        assert body["error_type"] == "RagArtError"
        assert "internal detail" in body["detail"]


class TestEveryLeafHasMandatoryFields:
    """Catch any future subclass that forgets to set http_status/message."""

    @pytest.mark.parametrize("cls", ALL_LEAF_EXCEPTIONS)
    def test_has_http_status(self, cls):
        assert isinstance(cls.http_status, int)
        assert 100 <= cls.http_status <= 599

    @pytest.mark.parametrize("cls", ALL_LEAF_EXCEPTIONS)
    def test_has_turkish_message(self, cls):
        msg = cls.user_message_tr
        assert isinstance(msg, str)
        assert len(msg) >= 5  # not just a placeholder

    @pytest.mark.parametrize("cls", ALL_LEAF_EXCEPTIONS)
    def test_log_level_is_valid_logging_const(self, cls):
        # logging constants are 0/10/20/30/40/50 in stdlib
        assert cls.log_level in {
            logging.DEBUG, logging.INFO, logging.WARNING,
            logging.ERROR, logging.CRITICAL,
        }

    @pytest.mark.parametrize("cls", ALL_LEAF_EXCEPTIONS)
    def test_inherits_from_ragart_error(self, cls):
        assert issubclass(cls, RagArtError)


class TestFamilies:
    """Verify the hierarchy groupings stay intact."""

    def test_workspace_family(self):
        assert issubclass(WorkspaceNotFoundError, WorkspaceError)
        assert issubclass(DefaultWorkspaceProtectedError, WorkspaceError)

    def test_retrieval_family(self):
        assert issubclass(StaleIndexError, RetrievalError)
        assert issubclass(EmptyKnowledgeBaseError, RetrievalError)

    def test_llm_family(self):
        assert issubclass(LLMRateLimitError, LLMError)
        assert issubclass(LLMAuthError, LLMError)


class TestSpecificMessages:
    """Critical messages we expect the UI to show verbatim."""

    def test_stale_index_message_mentions_reindex(self):
        # The UI shows this directly to the user — it must point them at
        # the right button name.
        msg = StaleIndexError.user_message_tr
        assert "Yeniden İndeksle" in msg

    def test_llm_rate_limit_status_is_429(self):
        # Flask-Limiter and our LLM provider both use this.
        assert LLMRateLimitError.http_status == 429

    def test_guard_blocked_is_400_not_500(self):
        # Security failures are user-induced, not server bugs.
        assert GuardBlockedError.http_status == 400


class TestRaiseCatch:
    """Catch-blocks must work as expected — type narrowing in mypy is
    a downstream goal but we can at least verify isinstance semantics."""

    def test_catch_specific(self):
        try:
            raise StaleIndexError("uuid abc")
        except StaleIndexError as e:
            assert e.detail == "uuid abc"
            assert e.http_status == 503
        except Exception:
            pytest.fail("Should have been caught by StaleIndexError clause")

    def test_catch_family(self):
        try:
            raise LLMRateLimitError()
        except LLMError as e:
            assert isinstance(e, LLMRateLimitError)

    def test_catch_root(self):
        try:
            raise GuardBlockedError()
        except RagArtError:
            pass
