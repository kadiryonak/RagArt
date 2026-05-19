"""Guard — prompt injection detection + groundedness scoring.

İki bağımsız bileşen:

1. InputGuard
   Sorguyu LLM'e göndermeden ÖNCE tarar. Bilinen prompt injection
   patternlerini (İngilizce + Türkçe) yakalar. Tespit ederse ask()
   erken return yapar ve uyarı mesajı döner.

2. GroundednessScorer
   LLM cevabını aldıktan SONRA, cevaptaki claim'lerin context'te
   ne kadar "dayanaklı" olduğunu ölçer. Basit word-overlap tabanlı
   (LLM çağrısı yok, 0 ek maliyet). Skor düşükse metadata'ya
   uyarı eklenir.

COST: 0 LLM çağrısı. Tamamen CPU-tabanlı, ~1ms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ── InputGuard ────────────────────────────────────────────────────────

# Known injection patterns — English
_EN_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?(your\s+)?instructions",
    r"disregard\s+(all\s+)?above",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+if",
    r"pretend\s+(you\s+are|to\s+be)",
    r"override\s+(your\s+)?system",
    r"new\s+instruction[s]?:",
    r"system\s*:\s*",
    r"\n\s*human\s*:",
    r"\n\s*assistant\s*:",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"DAN\s+mode",
]

# Known injection patterns — Turkish
_TR_INJECTION_PATTERNS = [
    r"önceki\s+talimatları?\s+unut",
    r"kuralları?\s+(değiştir|unut|görmezden\s+gel)",
    r"tüm\s+kısıtlamaları?\s+(kaldır|unut)",
    r"artık\s+sen\s+bir",
    r"gibi\s+davran",
    r"rolünü\s+değiştir",
    r"sistem\s+mesajını?\s+(değiştir|yeniden\s+yaz)",
    r"sınırlamaları?\s+kaldır",
]

# Compile all patterns into a single regex
_INJECTION_RE = re.compile(
    "|".join(_EN_INJECTION_PATTERNS + _TR_INJECTION_PATTERNS),
    flags=re.IGNORECASE | re.MULTILINE,
)

# Suspicious structural markers
_STRUCTURAL_MARKERS = re.compile(
    r"(```\s*(system|instruction|prompt)|"
    r"\[SYSTEM\]|\[INST\]|<\|im_start\|>|<\|system\|>|"
    r"<<SYS>>|<s>|</s>)",
    flags=re.IGNORECASE,
)

_INJECTION_WARNING = (
    "⚠️ Bu sorgu potansiyel bir prompt injection girişimi olarak "
    "algılandı ve güvenlik nedeniyle işlenmedi. Lütfen normal bir "
    "soru sorun."
)


@dataclass
class GuardResult:
    """InputGuard sonucu."""
    is_safe: bool
    score: float          # 0.0 = safe, 1.0 = definite injection
    reason: Optional[str] = None


class InputGuard:
    """Rule-based prompt injection detector.

    LLM çağrısı YAPMAZ. Pattern-match + heuristic tabanlı.
    """

    THRESHOLD = 0.5  # Bu ve üstü → reject

    @classmethod
    def check(cls, question: str) -> GuardResult:
        if not question or not question.strip():
            return GuardResult(is_safe=True, score=0.0)

        score = 0.0
        reasons: List[str] = []

        # Pattern matching
        injection_matches = _INJECTION_RE.findall(question)
        if injection_matches:
            score += 0.6
            reasons.append(f"injection_pattern({len(injection_matches)})")

        # Structural markers (LLM prompt format leaking)
        struct_matches = _STRUCTURAL_MARKERS.findall(question)
        if struct_matches:
            score += 0.4
            reasons.append(f"structural_markers({len(struct_matches)})")

        # Unusually long input (>500 chars is suspicious for a Q&A query)
        if len(question) > 500:
            score += 0.1
            reasons.append("long_input")

        # Multiple newlines (prompt stuffing)
        if question.count("\n") > 5:
            score += 0.2
            reasons.append("newline_stuffing")

        score = min(score, 1.0)
        is_safe = score < cls.THRESHOLD

        return GuardResult(
            is_safe=is_safe,
            score=score,
            reason="; ".join(reasons) if reasons else None,
        )

    @staticmethod
    def rejection_message() -> str:
        return _INJECTION_WARNING


# ── GroundednessScorer ────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    """Basit Türkçe-uyumlu tokenizer — lowercase + non-alpha strip."""
    return {
        w for w in re.findall(r"[a-zçğıöşü0-9]+", text.lower(), re.UNICODE)
        if len(w) > 2  # stopword-ish kısa kelimeleri atla
    }


class GroundednessScorer:
    """Cevabın context'e ne kadar dayandığını ölçer.

    Basit yaklaşım: cevaptaki content-word'lerin yüzde kaçı
    context'te de geçiyor? Yüksek overlap = grounded.

    SINIRLAR:
        - Paraphrasing yakalamaz (semantic değil, lexical)
        - Türkçe ek/kök ayrımı yok (basit overlap)
        - LLM'in eklediği genel bilgi (doğru ama context'te yok)
          düşük skor verir → false negative mümkün

    Gelişmiş versiyon: embedding-based sentence overlap (Faz 6?).
    """

    GROUNDED_THRESHOLD = 0.3   # Altında → uyarı

    @classmethod
    def score(cls, answer: str, context: str) -> float:
        """0.0–1.0 arası groundedness skoru döner."""
        if not answer or not context:
            return 0.0

        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(context)

        if not answer_tokens:
            return 0.0

        overlap = answer_tokens & context_tokens
        return len(overlap) / len(answer_tokens)

    @classmethod
    def is_grounded(cls, score: float) -> bool:
        return score >= cls.GROUNDED_THRESHOLD
