"""Multi-Query — orijinal soruyu N farklı şekilde yeniden yazıp her birinde
retrieval yapar, sonuçları RRF ile birleştirir, son LLM çağrısıyla cevap üretir.

NEDEN?
    Kısa/belirsiz sorularda ("auth issue") retrieval'ın recall'ı düşüktür.
    LLM aynı niyeti farklı kelimelerle ifade ederek "doğru" chunk'a
    erişme şansını artırır.

MALİYET
    1 LLM çağrısı (sorgu üretimi) + N retrieval + 1 LLM çağrısı (final
    cevap) = ~2 LLM çağrısı + N retrieval. Default N=3.

NE ZAMAN İYİ?
    - Kısa, belirsiz sorgular
    - Spesifik terim eksik ama kavram açıksa
    - Dense-only retrieval kullanıyorsanız

NE ZAMAN ZAYIF?
    - Sorgu zaten spesifik ve sözcükleri uygun
    - Latency-hassas demolar
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, List

from src.prompt_strategies.base import (
    BasePromptStrategy,
    PromptStrategyFactory,
    StrategyContext,
)
from src.prompt_strategies.direct import _TEMPLATE_NO_MEMORY, _TEMPLATE_WITH_MEMORY


_REWRITE_PROMPT = """Aşağıdaki soruyu, aynı niyeti koruyarak {n} FARKLI şekilde yeniden yaz. Her satıra bir varyant koy; numaralama kullan (1., 2., 3.). Hiçbir açıklama ekleme, sadece varyantları yaz.

ORİJİNAL SORU: {question}

VARYANTLAR:"""


class MultiQueryStrategy(BasePromptStrategy):
    name = "multi_query"
    label = "Multi-Query (N varyant)"
    description_tr = (
        "Soruyu N=3 farklı şekilde yeniden yazıp her biriyle retrieval "
        "yapar, sonuçları RRF ile birleştirir. Recall'u artırır; ~2 LLM "
        "çağrısı + N retrieval maliyeti."
    )
    is_multi_call = True
    is_multi_query = True

    DEFAULT_N_VARIANTS = 3
    K_RRF = 60

    def __init__(self, n_variants: int = DEFAULT_N_VARIANTS):
        self.n_variants = max(1, min(8, int(n_variants)))

    def generate_query_variations(
        self,
        question: str,
        ctx: StrategyContext,
    ) -> List[str]:
        prompt = _REWRITE_PROMPT.format(n=self.n_variants, question=question)
        try:
            raw = ctx.llm.generate(prompt, **ctx.llm_params)
        except Exception:
            # If the rewrite call fails, gracefully fall back to single query
            return [question]

        # Parse "1. ... 2. ... 3. ..."
        variants = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip leading "1." / "1)" / "-" etc.
            cleaned = re.sub(r"^[\d\)\.\-•]+\s*", "", line).strip()
            if cleaned and cleaned != question:
                variants.append(cleaned)
        # Always include original at position 0
        return [question] + variants[: self.n_variants]

    def build_prompt(
        self,
        *,
        question: str,
        context: str,
        memory_context: str = "",
        **kwargs: Any,
    ) -> str:
        # Final answer prompt is just the direct template — the win came
        # from richer retrieval, not a fancier final prompt.
        if memory_context.strip():
            return _TEMPLATE_WITH_MEMORY.format(
                memory_context=memory_context,
                context=context,
                question=question,
            )
        return _TEMPLATE_NO_MEMORY.format(context=context, question=question)


PromptStrategyFactory.register("multi_query", MultiQueryStrategy)
