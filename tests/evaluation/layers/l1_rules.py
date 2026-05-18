"""L1 — Kural tabanlı (rule-based) değerlendirme.

NE ÖLÇER?
    Cevabın "yüzeysel sağlığını" hızlıca kontrol eder. Bu katman ANLAMSAL
    doğruluğu ölçmez — sadece "cevap kabul edilebilir formatta mı?" der.

NASIL ÇALIŞIR?
    Bağımsız check'lerin ağırlıklı toplamı. Her check 0.0–1.0 arasında bir
    skor üretir, sonuçta ortalamaları alınır. Hangi kontroller var:

    1. Uzunluk kontrolü        — çok kısa (< 20 char) veya çok uzun (> 4000)
                                  cevaplar başarısız sayılır
    2. Dil tespiti             — Türkçe soru → Türkçe cevap olmalı
                                  (Türkçeye özel karakter sayısı ile heuristic)
    3. Anahtar kelime kapsamı  — golden item'daki keywords cevapta geçiyor mu
    4. Yasaklı kalıp           — "I don't know", "yardımcı olamam" gibi tipik
                                  başarısızlık ifadeleri
    5. Halüsinasyon sinyali    — cevapta soruda olmayan ve hiçbir kaynakta
                                  geçmeyen "büyük" sayılar/isimler varsa
                                  düşürür (zayıf bir sinyal, sadece uyarı)

NE ZAMAN KULLANILIR?
    Her test koşumunda çalışır. Smoke test mantığı: sistem hiç çalışıyor mu?
    Sıfıra yakın skor → muhtemelen pipeline bozuk.

SINIRLAR
    Format doğru olsa bile cevap yanlış olabilir. L1 başarılı diye semantik
    güvence ALMAZ. L1 pass eşiği geçilirse, üst katmanlara hak kazanır.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from tests.evaluation.dataset import GoldenItem
from tests.evaluation.layers.base import BaseEvaluator, RAGOutput


# Türkçeye özgü karakter seti — basit ama etkili bir dil sinyali
_TURKISH_CHARS = set("çğıöşüÇĞIİÖŞÜ")

# Tipik "kötü" cevap kalıpları
_BAD_PATTERNS = [
    r"\bI don'?t know\b",
    r"\bcannot help\b",
    r"\bI'?m sorry\b",
    r"yard[ıi]mc[ıi] olam[aı]yor",
    r"bilgim yok",
    r"konuyla ilgili bilgim",
    r"AI dil model[iı]",
    r"as an AI",
]


def _length_score(text: str, *, min_len: int = 20, max_len: int = 4000) -> tuple[float, Dict[str, Any]]:
    n = len(text.strip())
    if n < min_len:
        return 0.0, {"length": n, "verdict": "too_short"}
    if n > max_len:
        return 0.5, {"length": n, "verdict": "too_long"}
    return 1.0, {"length": n, "verdict": "ok"}


def _turkish_language_score(text: str) -> tuple[float, Dict[str, Any]]:
    """Heuristic: Türkçeye özgü karakterler + tipik Türkçe kelime oranı."""
    if not text:
        return 0.0, {"ratio_tr_chars": 0.0}

    tr_count = sum(1 for c in text if c in _TURKISH_CHARS)
    letters = sum(1 for c in text if c.isalpha())
    ratio = (tr_count / letters) if letters else 0.0

    # Heuristic — kalibre edildi: gerçek Türkçe metin tipik %3-8 arasında
    if ratio >= 0.02:
        score = 1.0
    elif ratio >= 0.01:
        score = 0.7
    elif ratio > 0:
        score = 0.4
    else:
        score = 0.0
    return score, {"ratio_tr_chars": round(ratio, 4), "tr_char_count": tr_count}


def _keyword_coverage_score(text: str, keywords: List[str]) -> tuple[float, Dict[str, Any]]:
    if not keywords:
        return 1.0, {"keywords_total": 0, "keywords_found": 0, "missing": []}

    text_lower = text.lower()
    found, missing = [], []
    for kw in keywords:
        if kw.lower() in text_lower:
            found.append(kw)
        else:
            missing.append(kw)

    score = len(found) / len(keywords)
    return score, {
        "keywords_total": len(keywords),
        "keywords_found": len(found),
        "found": found,
        "missing": missing,
    }


def _bad_pattern_score(text: str) -> tuple[float, Dict[str, Any]]:
    hits = []
    for pattern in _BAD_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    if hits:
        return 0.0, {"bad_patterns_hit": hits}
    return 1.0, {"bad_patterns_hit": []}


class L1RulesEvaluator(BaseEvaluator):
    """L1 — kural tabanlı format/sağlık check'i."""

    name = "L1_rules"
    threshold = 0.6  # pass eşiği

    # Her check'in nihai skordaki ağırlığı (toplam 1.0)
    WEIGHTS = {
        "length": 0.20,
        "language": 0.20,
        "keywords": 0.40,
        "bad_patterns": 0.20,
    }

    def _evaluate(self, item: GoldenItem, output: RAGOutput) -> tuple[float, Dict[str, Any]]:
        answer = output.answer or ""

        # Boş cevap → diğer check'leri çalıştırmaya gerek yok; sıfır.
        # (Aksi halde "yasaklı kalıp yok" gibi kontroller boş cevaba puan verir.)
        if not answer.strip():
            return 0.0, {"reason": "empty_answer"}

        length_score, length_d = _length_score(answer)
        lang_score, lang_d = _turkish_language_score(answer)
        kw_score, kw_d = _keyword_coverage_score(answer, item.keywords)
        bp_score, bp_d = _bad_pattern_score(answer)

        total = (
            self.WEIGHTS["length"] * length_score
            + self.WEIGHTS["language"] * lang_score
            + self.WEIGHTS["keywords"] * kw_score
            + self.WEIGHTS["bad_patterns"] * bp_score
        )

        return total, {
            "length": {"score": round(length_score, 3), **length_d},
            "language": {"score": round(lang_score, 3), **lang_d},
            "keywords": {"score": round(kw_score, 3), **kw_d},
            "bad_patterns": {"score": round(bp_score, 3), **bp_d},
        }
