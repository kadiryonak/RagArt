"""Adaptive Retrieval — sorgu karmaşıklığına göre retrieval parametreleri.

NEDEN?
    "Merhaba" diyene vektör tabanı araması yapmak boş maliyet.
    "X ile Y karşılaştır, Z bağlamında" gibi zor bir soru k=3 ile
    yeterli kontekst bulamaz. Adaptive Retrieval her soruyu sınıflar
    ve k / strategy / rerank kararlarını dinamik alır.

KATEGORİLER
    GREETING  → retrieval atla, direkt selamlama
    SIMPLE    → k=2, dense-only (hızlı, tek çağrı)
    MODERATE  → k=5, hybrid (default davranış)
    COMPLEX   → k=10, hybrid+rerank (maksimum recall)

COST: 0 LLM çağrısı (pure keyword/rule tabanlı).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QueryComplexity(Enum):
    GREETING = "greeting"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass(frozen=True)
class AdaptiveConfig:
    """Bir complexity seviyesi için retrieval parametreleri."""
    k: int
    retrieval_strategy: Optional[str]  # None = default (hybrid)
    rerank: bool
    skip_retrieval: bool = False       # greeting → True


# ----- Complexity → config mapping -----

ADAPTIVE_CONFIGS = {
    QueryComplexity.GREETING: AdaptiveConfig(
        k=0, retrieval_strategy=None, rerank=False, skip_retrieval=True,
    ),
    QueryComplexity.SIMPLE: AdaptiveConfig(
        k=2, retrieval_strategy="dense", rerank=False,
    ),
    QueryComplexity.MODERATE: AdaptiveConfig(
        k=5, retrieval_strategy=None, rerank=False,  # None → hybrid if available
    ),
    QueryComplexity.COMPLEX: AdaptiveConfig(
        k=10, retrieval_strategy=None, rerank=True,
    ),
}


# ----- Rule-based classifier -----

# Greeting patterns (Turkish + universal)
_GREETING_PATTERNS = re.compile(
    r"^(merhaba|selam|hey|hi|hello|naber|nasılsın|günaydın|iyi günler|"
    r"iyi akşamlar|iyi geceler|hoş ?geldiniz|sa|selamun?\s*aleyküm)$",
    flags=re.IGNORECASE,
)

# Complexity indicators
_COMPLEX_SIGNALS = re.compile(
    r"(karşılaştır|fark[ıi]\s+(ne|nedir)|avantaj|dezavantaj|"
    r"nasıl\s+çalış|neden\s+.+\s+yerine|hangisi\s+daha|"
    r"arası(ndaki)?\s+fark|compare|versus|vs\.?|"
    r"açıkla\s+ve\s+karşılaştır|detaylı|analiz\s+et|"
    r"liste(le|yle)|sıra(la|yla)|özetle\s+ve)",
    flags=re.IGNORECASE,
)

_MULTI_QUESTION = re.compile(r"\?\s*\S", flags=re.UNICODE)

# Simple signals: very short question with a single entity/keyword
_SIMPLE_SIGNALS = re.compile(
    r"^(ne(dir)?|kim(dir)?|nedir\s*\??|ne\s+demek|tanımı?\s*(ne(dir)?)?)\s*\??$",
    flags=re.IGNORECASE,
)


class QueryClassifier:
    """Keyword/rule tabanlı sorgu karmaşıklık sınıflandırıcısı.

    LLM çağrısı YAPMAZ — 0 latency, 0 maliyet.
    """

    @staticmethod
    def classify(question: str) -> QueryComplexity:
        q = question.strip()

        # 1. Greeting check
        if _GREETING_PATTERNS.match(q):
            return QueryComplexity.GREETING

        # 2. Word count
        words = q.split()
        word_count = len(words)

        # Very short (1-3 words) and no complex signal → simple
        if word_count <= 3 and not _COMPLEX_SIGNALS.search(q):
            return QueryComplexity.SIMPLE

        # 3. Complex signals
        complex_score = 0
        if _COMPLEX_SIGNALS.search(q):
            complex_score += 2
        if _MULTI_QUESTION.search(q):
            complex_score += 1
        if word_count > 15:
            complex_score += 1
        if q.count(",") >= 2:
            complex_score += 1

        if complex_score >= 2:
            return QueryComplexity.COMPLEX

        # 4. Moderate (default)
        return QueryComplexity.MODERATE

    @staticmethod
    def get_config(complexity: QueryComplexity) -> AdaptiveConfig:
        return ADAPTIVE_CONFIGS[complexity]


# ----- Greeting response -----

_GREETING_RESPONSES = [
    "Merhaba! 👋 Size nasıl yardımcı olabilirim? Bilgi tabanındaki "
    "konular hakkında sorularınızı sorabilirsiniz.",
]


def greeting_response(question: str) -> str:
    """Selamlama sorusu için hızlı cevap."""
    return _GREETING_RESPONSES[0]
