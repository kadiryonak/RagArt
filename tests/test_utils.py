"""Unit tests for src/utils.py — logging setup + small helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.utils import (
    StatusEmoji,
    calculate_word_overlap,
    ensure_directory,
    get_logger,
    setup_logging,
    truncate_text,
)


class TestSetupLogging:
    def test_returns_named_logger(self):
        logger = setup_logging(level=logging.INFO)
        assert logger.name == "rag_system"
        assert logger.level == logging.INFO

    def test_clears_existing_handlers(self):
        logger = logging.getLogger("rag_system")
        logger.addHandler(logging.NullHandler())
        before = len(logger.handlers)
        setup_logging()
        # Old handlers cleared; exactly one console handler remains
        assert len(logger.handlers) == 1
        assert before >= 1

    def test_console_handler_added(self):
        logger = setup_logging()
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    def test_file_handler_added(self, tmp_path):
        log_file = tmp_path / "rag.log"
        logger = setup_logging(log_file=str(log_file))
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        # Writing actually lands in the file
        logger.info("hello")
        for h in logger.handlers:
            h.flush()
        assert log_file.exists()

    def test_custom_format_string_applied(self):
        logger = setup_logging(format_string="%(message)s")
        fmt = logger.handlers[0].formatter
        assert fmt._fmt == "%(message)s"


class TestGetLogger:
    def test_default_name(self):
        assert get_logger().name == "rag_system"

    def test_custom_name(self):
        assert get_logger("my.module").name == "my.module"

    def test_same_name_returns_same_instance(self):
        assert get_logger("x") is get_logger("x")


class TestEnsureDirectory:
    def test_creates_missing_directory(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        result = ensure_directory(str(target))
        assert isinstance(result, Path)
        assert target.is_dir()

    def test_idempotent_on_existing(self, tmp_path):
        target = tmp_path / "exists"
        target.mkdir()
        # Must not raise
        result = ensure_directory(str(target))
        assert result == target


class TestTruncateText:
    def test_short_text_unchanged(self):
        assert truncate_text("hi", max_length=200) == "hi"

    def test_exact_length_unchanged(self):
        text = "x" * 50
        assert truncate_text(text, max_length=50) == text

    def test_long_text_truncated_with_suffix(self):
        text = "x" * 300
        out = truncate_text(text, max_length=100)
        assert len(out) == 100
        assert out.endswith("...")

    def test_custom_suffix(self):
        out = truncate_text("y" * 50, max_length=10, suffix=">>")
        assert out.endswith(">>")
        assert len(out) == 10


class TestCalculateWordOverlap:
    def test_identical_texts_full_overlap(self):
        assert calculate_word_overlap("algoritma nedir", "algoritma nedir") == 1.0

    def test_no_overlap_is_zero(self):
        assert calculate_word_overlap("kedi köpek", "araba uçak") == 0.0

    def test_partial_overlap_ratio(self):
        # words1 = {a, b}; common = {a} → 0.5
        assert calculate_word_overlap("a b", "a c d") == 0.5

    def test_empty_first_text_is_zero(self):
        assert calculate_word_overlap("", "anything") == 0.0

    def test_case_insensitive(self):
        assert calculate_word_overlap("Python", "python") == 1.0


class TestStatusEmoji:
    def test_constants_are_nonempty_strings(self):
        for attr in ("SUCCESS", "ERROR", "WARNING", "INFO", "ROCKET"):
            val = getattr(StatusEmoji, attr)
            assert isinstance(val, str) and val
