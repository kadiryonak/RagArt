"""Optional pre-chunking distillation — shrink raw text with an LLM while
preserving its meaning, so the index stays dense, cheap and on-point.

WHY THIS MATTERS (the single biggest lever on a large corpus):
    The dominant cost of a RAG index is not the chunker or the vector store —
    it is the raw text. 100 MB of raw text balloons to hundreds of MB once
    metadata is attached and to tens of GB after embedding. Most of that bulk
    is filler: boilerplate, repetition, hedging, formatting cruft. Distilling
    each document — having an LLM rewrite it as a compact, fact-preserving
    version — keeps only the signal. Every later stage (embedding, search,
    context assembly) then operates on less text, so it is faster, smaller and
    more accurate, because retrieval isn't diluted by noise.

OFF BY DEFAULT, on purpose:
    Distillation costs one LLM call per document at index time and trades some
    verbatim fidelity for density. That is undesirable where exact wording is
    load-bearing (legal, medical, contracts). So callers opt in explicitly.

SAFETY:
    Distillation must NEVER lose a document or make it larger. Every result is
    guarded: an empty / error / non-shrinking / suspiciously-tiny output is
    rejected and the original text is kept. Change-detection still hashes the
    RAW text upstream, so distillation (which is non-deterministic) does not
    cause incremental reindex to re-embed unchanged files.
"""

from __future__ import annotations

from typing import Any, List

try:
    from langchain_core.documents import Document
except ImportError:  # older langchain
    from langchain.schema import Document

from src.utils import get_logger, StatusEmoji

logger = get_logger(__name__)


# Below this length distillation isn't worth a network round-trip.
DISTILL_MIN_CHARS = 400
# Reject results shorter than this fraction of the original — a 20-line page
# collapsed to one sentence is a summary, not a distillation, and has dropped
# information. Keep the original in that case.
DISTILL_MIN_RATIO = 0.15

# Provider error strings (see llm_providers) — never index these as content.
_ERROR_MARKERS = ("API error", "Connection error", "API hatası", "error:")

_PROMPT = (
    "Aşağıdaki metni, içindeki TÜM olgusal bilgiyi, terimleri, tanımları, "
    "sayıları, isimleri ve bağlamı KORUYARAK mümkün olduğunca kısalt. "
    "Gereksiz tekrarları, dolgu ifadeleri ve süslemeleri çıkar. Yeni bilgi "
    "ekleme, yorum yapma, başlık koyma — yalnızca damıtılmış düz metni döndür.\n\n"
    "METİN:\n{text}\n\nDAMITILMIŞ METİN:"
)


def distill_text(
    text: str,
    llm_provider: Any,
    *,
    min_chars: int = DISTILL_MIN_CHARS,
    min_ratio: float = DISTILL_MIN_RATIO,
    **llm_params: Any,
) -> str:
    """Return a compact, meaning-preserving version of `text`.

    Falls back to the ORIGINAL text on anything suspicious (short input, LLM
    error/exception, empty / non-shrinking / over-compressed output) so the
    caller can apply this blindly without risking lost or corrupted content.
    """
    if not text or len(text) < min_chars:
        return text

    try:
        out = llm_provider.generate(_PROMPT.format(text=text), **llm_params)
    except Exception as e:  # network / provider failure → keep original
        logger.warning(f"{StatusEmoji.WARNING} Distillation failed: {e}")
        return text

    out = (out or "").strip()
    if not out:
        return text
    low = out.lower()
    if any(marker.lower() in low for marker in _ERROR_MARKERS):
        return text
    # Must actually shrink, but not so much that it became a summary.
    if len(out) >= len(text):
        return text
    if len(out) < int(len(text) * min_ratio):
        logger.warning(
            f"{StatusEmoji.WARNING} Distillation over-compressed "
            f"({len(text)}→{len(out)} chars) — keeping original."
        )
        return text
    return out


def distill_documents(
    documents: List[Document],
    llm_provider: Any,
    **opts: Any,
) -> List[Document]:
    """Distill each document's text, preserving metadata.

    Documents whose text was actually shortened are tagged
    ``metadata["distilled"] = True``. Order is preserved.
    """
    out: List[Document] = []
    changed = 0
    total_before = total_after = 0
    for d in documents:
        before = d.page_content or ""
        after = distill_text(before, llm_provider, **opts)
        meta = dict(d.metadata)
        if after != before:
            meta["distilled"] = True
            changed += 1
        total_before += len(before)
        total_after += len(after)
        out.append(Document(page_content=after, metadata=meta))

    if documents:
        pct = (1 - total_after / total_before) * 100 if total_before else 0.0
        logger.info(
            f"{StatusEmoji.SUCCESS} Distilled {changed}/{len(documents)} "
            f"document(s) — {pct:.0f}% smaller"
        )
    return out
