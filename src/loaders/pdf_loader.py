"""PDFLoader — text extraction with scanned-PDF detection + optional OCR.

Per page:
    1. Extract the embedded text layer with pypdf.
    2. If that text looks like real prose, keep it.
    3. Otherwise the page is image-only / scanned / badly-encoded (pypdf
       returns glyph-soup for these — e.g. a "Scanned by CamScanner" PDF).
       - If OCR extras are installed (pypdfium2 + easyocr), render the page
         and OCR it (Turkish + English).
       - If not, skip the page and record that the PDF needs OCR, so the
         upload/reindex layer can tell the user instead of silently indexing
         garbage (the old behaviour — which produced relevance-0 answers).

The garbage-detection (step 2) is the key fix and needs no extra deps. OCR is
opt-in:  pip install "ragart[ocr]"
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.loaders.base import BaseLoader

logger = logging.getLogger(__name__)


# ── Text-quality heuristic ─────────────────────────────────────────────

# A "word" = 3+ consecutive letters (Latin + Turkish). Glyph-soup from a bad
# scan is mostly 1-2 char tokens, digits and symbols, so it has very few of
# these and a low word/token ratio.
_WORD_RE = re.compile(r"[A-Za-zÇĞİıÖŞÜçğöşü]{3,}")


def looks_like_text(text: str, *, min_words: int = 3, min_ratio: float = 0.5) -> bool:
    """True if `text` looks like genuinely-extracted prose.

    Distinguishes a real text layer from the glyph-soup pypdf returns for
    scanned / image-only / broken-encoding PDFs. The discriminator is the
    word/token ratio: real prose is ~0.9 (almost every whitespace token is a
    word), while scanned-garbage is ~0.05 (mostly stray single chars and
    symbols). A small word-count floor rejects near-empty pages whose few
    tokens happen to be wordlike. Measured on real "Scanned by CamScanner"
    pages (ratio 0.02–0.06) vs. clean Turkish pages (ratio 0.9–1.0).
    """
    if not text or not text.strip():
        return False
    tokens = text.split()
    if not tokens:
        return False
    words = _WORD_RE.findall(text)
    ratio = len(words) / len(tokens)
    return len(words) >= min_words and ratio >= min_ratio


# ── Lazy OCR engine (optional [ocr] extra) ─────────────────────────────

_OCR_LANGS = ["tr", "en"]
_OCR_SCALE = 3          # ~216 DPI — good OCR accuracy vs. speed
_ocr_reader = None      # easyocr.Reader singleton (heavy: build once)


def ocr_available() -> bool:
    """True if the optional OCR stack (pypdfium2 + easyocr) is importable."""
    try:
        import easyocr  # noqa: F401
        import numpy  # noqa: F401
        import pypdfium2  # noqa: F401
        return True
    except Exception:
        return False


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        logger.info("Loading OCR model (easyocr, tr+en) — first run downloads weights...")
        _ocr_reader = easyocr.Reader(_OCR_LANGS, gpu=False, verbose=False)
    return _ocr_reader


def _ocr_page(pdfium_doc, idx: int) -> str:
    """Render page `idx` to an image and OCR it. Returns extracted text."""
    import numpy as np
    pil = pdfium_doc[idx].render(scale=_OCR_SCALE).to_pil()
    reader = _get_ocr_reader()
    lines = reader.readtext(np.array(pil), detail=0, paragraph=True)
    return "\n".join(lines).strip()


class PDFLoader(BaseLoader):
    name = "pdf"
    extensions = {".pdf"}

    MIN_PAGE_TEXT = 20  # below this many chars → treat the page as empty

    def load(self, file_path: Path) -> List[Document]:
        # Lazy import: pypdf missing → only this loader fails.
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError(
                "pypdf is required for PDF loading. Install with: pip install pypdf"
            ) from e

        reader = PdfReader(str(file_path))
        total = len(reader.pages)
        documents: List[Document] = []

        ocr_ok = ocr_available()
        pdfium_doc = None
        scanned_pages = 0
        ocr_pages = 0

        for idx, page in enumerate(reader.pages):
            try:
                text = (page.extract_text() or "").strip()
            except Exception:
                text = ""

            extraction = "text"
            if not looks_like_text(text):
                # Image-only / scanned / broken-encoding page.
                recovered = ""
                if ocr_ok:
                    try:
                        if pdfium_doc is None:
                            import pypdfium2 as pdfium
                            pdfium_doc = pdfium.PdfDocument(str(file_path))
                        recovered = _ocr_page(pdfium_doc, idx)
                    except Exception as e:
                        logger.warning("OCR failed on %s p%d: %s", file_path.name, idx + 1, e)
                # Accept OCR text on a looser bar (OCR is noisier than a clean
                # text layer, but partial text still helps retrieval).
                if looks_like_text(recovered, min_words=4, min_ratio=0.30):
                    text, extraction = recovered, "ocr"
                    ocr_pages += 1
                else:
                    scanned_pages += 1
                    continue

            if len(text) < self.MIN_PAGE_TEXT:
                continue

            documents.append(Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "format": "pdf",
                    "file_path": str(file_path),
                    "item_index": idx,
                    "item_count": total,
                    "page": idx + 1,
                    "extraction": extraction,
                },
            ))

        if ocr_pages:
            logger.info("%s: OCR recovered %d/%d page(s)", file_path.name, ocr_pages, total)
        if scanned_pages:
            hint = "" if ocr_ok else " — OCR için: pip install \"ragart[ocr]\""
            logger.warning(
                "%s: %d/%d sayfa taranmış/okunamadı, atlandı%s",
                file_path.name, scanned_pages, total, hint,
            )
        return documents
