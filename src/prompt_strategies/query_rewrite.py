"""Query Rewrite — LLM sorguyu retrieval için tek seferde yeniden yazar.

MULTI-QUERY ile FARKI?
    Multi-Query: N varyant üret, N retrieval yap, RRF birleştir (yüksek
    recall ama N+1 LLM çağrısı).
    Query-Rewrite: 1 yeniden yazma + orijinali de tut → 2 retrieval
    (orijinali güvenlik ağı, yeniden yazılmış kalite için). 2 LLM
    çağrısı yerine sadece 1 ek.

NE ZAMAN İYİ?
    - Konuşma dili sorgular ("şu kişi kim?", "auth hatası ne yapayım?")
    - Argo/eksik kelime
    - Kısa sorgular spesifik terim eksik

NASIL ÇALIŞIR?
    1. LLM "Bu soruyu retrieval için net, spesifik, aranabilir bir forma
       yeniden yaz" prompt'una cevap verir.
    2. rag_system orijinal + yeniden yazılmış sorguyu retrieve eder.
    3. RRF birleştirir; final cevap direct prompt ile üretilir.

KUSURLAR
    - Yeniden yazma niyeti bozarsa retrieval daha kötüleşir
      (orijinali tutuyor olmamız bu durumu absorbe eder)
    - +1 LLM çağrısı maliyeti
"""

from __future__ import annotations

import re
from typing import Any, List

from src.prompt_strategies.base import (
    BasePromptStrategy,
    PromptStrategyFactory,
    StrategyContext,
)
from src.prompt_strategies.direct import _TEMPLATE_NO_MEMORY, _TEMPLATE_WITH_MEMORY


_REWRITE_PROMPT = """Aşağıdaki soruyu retrieval (bilgi tabanı arama) için daha NET, SPESİFİK ve aranabilir bir forma yeniden yaz. Şu kurallara uy:

1. Anlamı KORU; yorum, ek bilgi, açıklama EKLEME.
2. Çıktı tek bir cümle olsun.
3. Etiket, başlık veya açıklama YAZMA. Sadece yeniden yazılmış soruyu döndür.

ORİJİNAL: {question}

YENİDEN YAZILMIŞ:"""


_PREFIX_RE = re.compile(
    r"^(yeniden\s+yaz[ıi]lm[ıi]ş?|rewritten|cevab[ıi]m|cevap)\s*:?\s*",
    flags=re.IGNORECASE,
)


class QueryRewriteStrategy(BasePromptStrategy):
    name = "query_rewrite"
    label = "Query Rewrite (sorgu iyileştirme)"
    description_tr = (
        "LLM sorguyu retrieval için 1 kez yeniden yazar; orijinal + yeniden "
        "yazılmış ile arar (RRF). Konuşma dili veya eksik anahtar kelimeli "
        "sorgularda recall artırır. +1 LLM çağrısı."
    )
    is_multi_call = True
    is_multi_query = True
    is_advanced = True

    def generate_query_variations(
        self,
        question: str,
        ctx: StrategyContext,
    ) -> List[str]:
        try:
            raw = ctx.llm.generate(
                _REWRITE_PROMPT.format(question=question),
                **ctx.llm_params,
            )
        except Exception:
            return [question]

        # Take the first non-empty line, strip any "YENİDEN YAZILMIŞ:" prefix
        rewritten = ""
        for line in (raw or "").split("\n"):
            line = line.strip()
            if not line:
                continue
            line = _PREFIX_RE.sub("", line).strip().strip("\"'")
            if line:
                rewritten = line
                break

        if not rewritten or rewritten.lower() == question.lower():
            return [question]
        return [question, rewritten]

    def build_prompt(
        self,
        *,
        question: str,
        context: str,
        memory_context: str = "",
        **kwargs: Any,
    ) -> str:
        if memory_context.strip():
            return _TEMPLATE_WITH_MEMORY.format(
                memory_context=memory_context,
                context=context,
                question=question,
            )
        return _TEMPLATE_NO_MEMORY.format(context=context, question=question)


PromptStrategyFactory.register("query_rewrite", QueryRewriteStrategy)
