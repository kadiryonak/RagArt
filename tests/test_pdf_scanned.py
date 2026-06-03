"""Tests for scanned/garbled-PDF detection in src/loaders/pdf_loader.py.

Guards the fix for the "PDF okumuyor" report: a scanned ("Scanned by
CamScanner") PDF produced glyph-soup that pypdf happily returned, and the
system indexed it as 39 junk chunks → every query scored relevance 0. The
loader now detects unreadable pages and skips them (instead of indexing
garbage), and OCRs them when the optional [ocr] extra is installed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.loaders.pdf_loader import PDFLoader, looks_like_text


# Real glyph-soup samples captured from a "Scanned by CamScanner" PDF —
# pypdf returns this for image-only pages.
_GARBAGE = [
    'Scanned by CamScanner W T2 AS a o N  $L ORf TmL i A o { & d J r EP i (',
    '1i " n - n T W p \' " . [1s \' " II Wa - lt n 61 o < l{ a \' " )l > A o L',
    'r r a \\ a r L n An N . J m s e (Q c Bm _ + l p n a a { D 5 e m c ( J e',
]


class TestLooksLikeText:
    @pytest.mark.parametrize("garbage", _GARBAGE)
    def test_rejects_scanned_glyph_soup(self, garbage):
        assert looks_like_text(garbage) is False

    def test_accepts_clean_turkish_prose(self):
        text = ("Optimizasyon, bir amaç fonksiyonunu belirli kısıtlar altında "
                "en iyi değere ulaştırma problemidir.")
        assert looks_like_text(text) is True

    def test_accepts_short_real_page(self):
        # Title-page-style short text must still pass (regression: an early
        # word-count floor wrongly rejected these).
        assert looks_like_text("Algoritma birinci sayfa Türkçe içerik.") is True

    def test_rejects_empty_and_whitespace(self):
        assert looks_like_text("") is False
        assert looks_like_text("   \n  ") is False

    def test_rejects_few_symbols(self):
        assert looks_like_text("a 1 % - / b") is False

    def test_ratio_threshold(self):
        # Mostly stray single chars → low word/token ratio → rejected.
        assert looks_like_text("kelime a b c d e f g h i j k") is False


class TestPDFLoaderScanned:
    """Loader-level behaviour when OCR is unavailable (the default install)."""

    def _fake_reader(self, monkeypatch, page_texts):
        """Patch pypdf.PdfReader so load() sees pages with the given text."""
        import pypdf

        pages = []
        for t in page_texts:
            pg = MagicMock()
            pg.extract_text.return_value = t
            pages.append(pg)
        fake = MagicMock()
        fake.pages = pages
        monkeypatch.setattr(pypdf, "PdfReader", lambda *_a, **_k: fake)
        # Force the "no OCR installed" path regardless of the test env.
        monkeypatch.setattr("src.loaders.pdf_loader.ocr_available", lambda: False)

    def test_scanned_pages_skipped_not_indexed_as_garbage(self, monkeypatch):
        self._fake_reader(monkeypatch, _GARBAGE)
        docs = PDFLoader().load(Path("scan.pdf"))
        # The whole point: zero garbage documents, not 3 junk chunks.
        assert docs == []

    def test_real_pages_still_loaded(self, monkeypatch):
        self._fake_reader(monkeypatch, [
            "Optimizasyon bir amaç fonksiyonunu kısıtlar altında en iyi değere ulaştırır.",
            "Gradyan inişi yaygın bir sayısal optimizasyon yöntemidir.",
        ])
        docs = PDFLoader().load(Path("good.pdf"))
        assert len(docs) == 2
        assert all(d.metadata["extraction"] == "text" for d in docs)
        assert docs[0].metadata["page"] == 1

    def test_mixed_pdf_keeps_only_readable_pages(self, monkeypatch):
        self._fake_reader(monkeypatch, [
            "Bu sayfa gerçek ve okunabilir Türkçe metin içeriği taşımaktadır.",
            _GARBAGE[0],  # scanned page → skipped (no OCR)
            "İkinci gerçek sayfa da yeterince anlamlı kelime barındırır burada.",
        ])
        docs = PDFLoader().load(Path("mixed.pdf"))
        assert len(docs) == 2
        assert {d.metadata["page"] for d in docs} == {1, 3}
