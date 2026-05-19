"""HyDE — Hypothetical Document Embedding.

NEDEN?
    Soru: "Kadir Yönak kimdir?"
    Normal retrieval: Sorunun embedding'i ↔ CV paragrafı embedding'i
    arasında semantic gap büyük. Soru kısa, cevap paragraf formunda.

    HyDE çözümü: LLM'den "Bu sorunun cevabı olabilecek bir paragraf yaz"
    diyerek varsayımsal bir doküman üretir:
        "Kadir Yönak, bilgisayar mühendisliği mezunu bir yazılım
         geliştiricidir. Yapay zeka ve RAG sistemleri üzerine..."
    Bu varsayımsal dokümanın embedding'i, gerçek CV paragrafına
    çok daha yakın olur → retrieval recall dramatik artar.

MALİYET
    +1 LLM çağrısı (varsayımsal doküman üretimi).
    Retrieval sayısı: 2 (orijinal + HyDE) — fuse with RRF.

NE ZAMAN İYİ?
    - Kişi sorguları ("X kimdir?")
    - Konsept açıklama ("Y nedir?")
    - Soru ↔ cevap arasında formül/kaynak farkı büyük olduğunda

NE ZAMAN ZAYIF?
    - Soru zaten yeterince spesifik ve retrieval iyi çalışıyor
    - Latency-hassas senaryolar
"""

from __future__ import annotations

from typing import Any, List

from src.prompt_strategies.base import (
    BasePromptStrategy,
    PromptStrategyFactory,
    StrategyContext,
)
from src.prompt_strategies.direct import _TEMPLATE_NO_MEMORY, _TEMPLATE_WITH_MEMORY


_HYDE_PROMPT = """Aşağıdaki soruya kısa bir cevap paragrafı yaz. Bu cevap gerçek olmak ZORUNDA DEĞİL — sadece sorunun cevabı ne olabilir onu tahmin et. 3-5 cümle yaz.

KURAL: Sadece paragrafı yaz. Başlık, açıklama, dipnot EKLEME.

SORU: {question}

PARAGRAF:"""


class HyDEStrategy(BasePromptStrategy):
    name = "hyde"
    label = "HyDE (Varsayımsal Doküman)"
    description_tr = (
        "Soruya varsayımsal bir cevap paragrafı üretip, onun embedding'iyle "
        "arama yapar. Soru ↔ doküman semantic gap'ini kapatır. Özellikle "
        "kişi ve konsept sorgularında recall artırır. +1 LLM çağrısı."
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
            hypo_doc = ctx.llm.generate(
                _HYDE_PROMPT.format(question=question),
                **ctx.llm_params,
            )
        except Exception:
            # LLM failure → fallback to single original query
            return [question]

        # Clean up: take first paragraph, strip boilerplate
        hypo_doc = (hypo_doc or "").strip()
        if not hypo_doc:
            return [question]

        # Return original + hypothetical (RRF will fuse results)
        return [question, hypo_doc]

    def build_prompt(
        self,
        *,
        question: str,
        context: str,
        memory_context: str = "",
        **kwargs: Any,
    ) -> str:
        # Final answer prompt is the standard direct template —
        # the win came from better retrieval, not a fancier prompt.
        if memory_context.strip():
            return _TEMPLATE_WITH_MEMORY.format(
                memory_context=memory_context,
                context=context,
                question=question,
            )
        return _TEMPLATE_NO_MEMORY.format(context=context, question=question)


PromptStrategyFactory.register("hyde", HyDEStrategy)
