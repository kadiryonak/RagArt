"""L4 — LLM-as-a-Judge (Groq ücretsiz katmanı).

NE ÖLÇER?
    Üç anahtar boyut, her biri 1–5 likert ile (sonra [0,1]'e normalize):

    1. Faithfulness   — Cevap, retrieved_context'teki bilgilere SADIK mı?
                        (Halüsinasyon detektörü.)
    2. Relevance      — Cevap, soruyla ALAKALI mı? Konudan sapma var mı?
    3. Completeness   — Soruya YETERİNCE kapsamlı cevap verildi mi?

NASIL ÇALIŞIR?
    - Tek bir Groq chat completion çağrısı, JSON formatında skor ister.
    - Default model: llama-3.3-70b-versatile (Groq free tier).
    - Düşük temperature (0.1), structured output zorlanır.
    - Cevap parse edilemezse skor 0.0 + error fieldı dolar.

NE ZAMAN KULLANILIR?
    - SADECE critical=True işaretli golden item'lar üzerinde.
    - Her major branch'in baseline ve final ölçümlerinde.
    - CI'da çalışmaz (rate limit + maliyet kontrolü).

SINIRLAR
    - LLM judge'lar tutarsız olabilir; aynı örnek için skor varyansı %5-10.
    - Türkçe için Llama-3.3-70B genelde iyi, ama nadiren İngilizce cevap
      verebilir → parser bunu handle ediyor.
    - Groq rate limit: free tier dakikada birkaç istek; çok örnekte yavaş.
    - Self-bias riski: LLM judge, kendi tarzına yakın cevaplara yüksek puan
      verebilir. Bu yüzden L1-L3 ile çapraz doğrulama önemli.

CONFIG
    GROQ_API_KEY env değişkeni ile authentication.
    Key yoksa katman "skipped" durumuna düşer (hata değil) — pipeline çalışmaya
    devam eder.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import requests

from tests.evaluation.dataset import GoldenItem
from tests.evaluation.layers.base import BaseEvaluator, RAGOutput


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


_JUDGE_PROMPT = """Sen tarafsız bir RAG sistemi değerlendiricisin. Aşağıda bir SORU,
sistem tarafından üretilen CEVAP, kullanılan BAĞLAM ve insan tarafından yazılmış
REFERANS CEVAP var. Üç boyutta 1-5 arası tam sayı puan ver.

PUANLAMA REHBERİ:
- 1 = çok kötü, 3 = orta, 5 = mükemmel

BOYUTLAR:
1. faithfulness   — Cevap, BAĞLAM'daki bilgilere sadık mı? Bağlamda olmayan
                    iddialar (halüsinasyon) var mı? Bağlam yoksa REFERANS'a göre değerlendir.
2. relevance      — Cevap, SORU ile alakalı mı? Konudan sapma var mı?
3. completeness   — Cevap, REFERANS ile karşılaştırıldığında soruyu yeterince
                    kapsayıp gerekli detayları içeriyor mu?

Çıktıyı YALNIZCA aşağıdaki JSON formatında ver, başka hiçbir şey yazma:
{{"faithfulness": <int 1-5>, "relevance": <int 1-5>, "completeness": <int 1-5>, "reasoning": "<kısa bir cümle>"}}

---
SORU:
{question}

CEVAP:
{answer}

BAĞLAM:
{context}

REFERANS CEVAP:
{reference}
---
JSON:"""


def _normalize_likert(score: float) -> float:
    """1-5 → [0, 1]."""
    return max(0.0, min(1.0, (score - 1.0) / 4.0))


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """LLM çıktısından JSON nesnesini al. Bazen markdown code fence içinde gelir."""
    # Önce ham JSON dene
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    # Sonra regex ile {...} bul
    match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except ValueError:
            return None
    return None


class L4JudgeEvaluator(BaseEvaluator):
    """L4 — Groq LLM judge.

    Args:
        api_key:    Groq API anahtarı (None ise GROQ_API_KEY env'dan alınır).
        model:      Kullanılacak Groq modeli.
        timeout:    HTTP timeout saniye.
        only_critical: True ise sadece critical=True item'larda çalışır.
        client:     Test'lerde mock için inject edilen requests modülü
                    (ducktype: .post(...) → response).
    """

    name = "L4_judge"
    threshold = 0.6  # Likert 3+ ortalaması ≈ 0.5; 0.6 → 3.4/5

    WEIGHTS = {"faithfulness": 0.45, "relevance": 0.30, "completeness": 0.25}

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0,
        only_critical: bool = True,
        client: Any = None,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.timeout = timeout
        self.only_critical = only_critical
        self.client = client or requests

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _evaluate(self, item: GoldenItem, output: RAGOutput) -> tuple[float, Dict[str, Any]]:
        if not self.is_available():
            raise RuntimeError("GROQ_API_KEY not set — L4 cannot run")
        if self.only_critical and not item.critical:
            return 0.0, {"skipped": True, "reason": "not_critical"}

        prompt = _JUDGE_PROMPT.format(
            question=item.question,
            answer=output.answer or "(empty)",
            context=output.retrieved_context or "(no context)",
            reference=item.reference_answer,
        )

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self.client.post(
            GROQ_API_URL, headers=headers, json=payload, timeout=self.timeout
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Groq API {response.status_code}: {response.text[:200]}"
            )

        body = response.json()
        content = body["choices"][0]["message"]["content"]
        parsed = _extract_json(content)
        if not parsed:
            raise ValueError(f"Could not parse judge JSON: {content[:200]}")

        # Skorları parse et + normalize et
        try:
            faith = float(parsed["faithfulness"])
            rel = float(parsed["relevance"])
            comp = float(parsed["completeness"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Missing/invalid score in judge output: {e}")

        faith_n = _normalize_likert(faith)
        rel_n = _normalize_likert(rel)
        comp_n = _normalize_likert(comp)

        total = (
            self.WEIGHTS["faithfulness"] * faith_n
            + self.WEIGHTS["relevance"] * rel_n
            + self.WEIGHTS["completeness"] * comp_n
        )

        return total, {
            "faithfulness": {"raw": faith, "normalized": round(faith_n, 4)},
            "relevance": {"raw": rel, "normalized": round(rel_n, 4)},
            "completeness": {"raw": comp, "normalized": round(comp_n, 4)},
            "reasoning": parsed.get("reasoning", ""),
            "model": self.model,
        }
