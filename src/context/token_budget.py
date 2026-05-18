"""TokenBudgetTrimmer — token sınırına göre kuyruktan kes.

NEDEN?
    Her LLM'in context limit'i var (örn. llama-3.3-70b: 128K, gpt-4o-mini:
    128K, ama input+output birlikte). Maliyet de var (input token başına
    ücret). Çok büyük context:
        - $$$
        - latency
        - lost-in-the-middle riski
    Bu yüzden bir budget belirle, kuyruktan (en az relevant) at.

TOKENIZATION
    Production'da `tiktoken` (OpenAI) veya `transformers.AutoTokenizer` —
    bizde dependency-light bir heuristic yeterli:
        approx_tokens(text) ≈ len(text) / 4   (English) veya
                              len(text) / 3   (Turkish — daha uzun
                                               kelimeler, daha az token/char)
    Kalibre edildi: Türkçe için ~3.0 karakter/token gerçekçi.

KESİM SIRASI
    Kuyruktan keser çünkü docs RELEVANCE order'dadır (öne en alakalı,
    sona en zayıf). Reorderer'dan ÖNCE çalıştırılmalı; aksi takdirde
    ortadan kesilir.
"""

from __future__ import annotations

from typing import List

from src.context.base import BaseContextProcessor
from src.retrievers.base import RetrievedDoc

# Türkçe için ~3 karakter/token, İngilizce için ~4. Karışıkta orta yol.
DEFAULT_CHARS_PER_TOKEN = 3.5


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    if not text:
        return 0
    return max(1, int(round(len(text) / chars_per_token)))


class TokenBudgetTrimmer(BaseContextProcessor):
    name = "token_budget"

    def __init__(
        self,
        *,
        max_tokens: int = 2000,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token

    def process(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        kept: List[RetrievedDoc] = []
        used = 0
        for doc in docs:
            cost = estimate_tokens(doc.page_content, self.chars_per_token)
            if used + cost > self.max_tokens and kept:
                # En az bir doc'u her zaman tut (boş context'ten iyidir)
                break
            kept.append(doc)
            used += cost
        return kept
