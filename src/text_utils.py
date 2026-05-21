"""Turkish-aware text tokenisation — shared by BM25 retrieval and the
L3 lexical evaluator.

This lived in tests/evaluation/layers/l3_lexical.py, but src/retrievers/
sparse.py needs it too — and production code must never import from the
test tree (it breaks `pip install`, where tests/ is not shipped). The
tokenizer is genuine production logic, so it belongs in src/.
"""

from __future__ import annotations

import re
from typing import List

# Common Turkish inflectional suffixes — crude but effective. Light
# stemming keeps BM25 from treating "rapor" and "raporları" as unrelated.
_TURKISH_SUFFIXES = (
    "ları", "leri", "lar", "ler",
    "ında", "inde", "unda", "ünde",
    "dan", "den", "tan", "ten",
    "nın", "nin", "nun", "nün", "ın", "in", "un", "ün",
    "ya", "ye", "ka", "ke",
    "dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür",
)


def light_stem(word: str) -> str:
    """Strip one common Turkish suffix if doing so leaves a real stem.

    Not a true morphological analyser — just enough to reduce the unfair
    inflection penalty in BM25 / BLEU / ROUGE scoring.
    """
    for suf in _TURKISH_SUFFIXES:
        if len(word) > len(suf) + 2 and word.endswith(suf):
            return word[: -len(suf)]
    return word


def tokenize(text: str, *, stem: bool = True) -> List[str]:
    """Lower-case, drop non-alphanumerics, optionally light-stem.

    Keeps Turkish letters (çğıöşü). Used as the BM25 tokenizer and by the
    lexical (BLEU/ROUGE) evaluation layer so both score the same way.
    """
    tokens = re.findall(r"[a-zçğıöşü0-9]+", text.lower())
    if stem:
        tokens = [light_stem(t) for t in tokens]
    return tokens
