"""Chain of Thought — LLM önce muhakeme yazar, sonra cevap.

NEDEN?
    Karmaşık çok adımlı sorularda LLM'in kendi muhakemesini görmesi
    final cevabın kalitesini artırır (Wei et al. 2022).

NE ZAMAN İYİ?
    - "Niçin / nasıl / kıyasla" tarzı analitik sorular
    - Multi-hop senaryolar
    - Sebep-sonuç gerektirenler

NE ZAMAN ZAYIF?
    - Tek-cümlelik faktöel sorular ("X kaç yılında doğdu?") — gereksiz
    - Latency hassas senaryolar (cevap 2-3x uzun)
"""

from __future__ import annotations

import re
from typing import Any

from src.prompt_strategies.base import BasePromptStrategy, PromptStrategyFactory
from src.prompt_strategies.base import StrategyContext


_TEMPLATE = """Sen Türkçe konuşan bir uzman asistanısın. SORU'yu BAĞLAM bilgileriyle çöz; önce KISA BİR MUHAKEME yaz, ardından final CEVAP'ı ver.

KURALLAR:
1. Üçüncü tekil şahıs.
2. "[Source N]" etiketleri YASAK.
3. Cevap doğal akıcı Türkçe.
4. BAĞLAM yetersizse muhakeme yazma; sadece şu cümle: "Bu konuda verilen bilgilerde yeterli detay bulunmuyor."

FORMAT (kullanıcıya GÖSTERİLECEK son cevap sadece YANIT bölümü):
MUHAKEME: <1-3 cümle, anahtar bilgileri bağlamdan topla>
YANIT: <son cevap>

BAĞLAM:
{context}

SORU: {question}

MUHAKEME:"""


class ChainOfThoughtStrategy(BasePromptStrategy):
    name = "chain_of_thought"
    label = "Chain of Thought (adım adım)"
    description_tr = (
        "LLM önce kısa bir muhakeme yazar, sonra cevap üretir. Karmaşık "
        "analitik veya çok-adımlı sorularda kaliteyi artırır."
    )

    def build_prompt(
        self,
        *,
        question: str,
        context: str,
        memory_context: str = "",
        **kwargs: Any,
    ) -> str:
        return _TEMPLATE.format(context=context, question=question)

    def execute(self, ctx: StrategyContext, *, question: str, context: str,
                memory_context: str = "", **kwargs: Any) -> str:
        # LLM "MUHAKEME: ...\nYANIT: ..." formatında dönecek;
        # kullanıcıya sadece YANIT bölümünü gösteriyoruz.
        prompt = self.build_prompt(
            question=question, context=context, memory_context=memory_context,
        )
        raw = ctx.llm.generate(prompt, **ctx.llm_params)
        return self._extract_answer(raw)

    @staticmethod
    def _extract_answer(raw: str) -> str:
        # Look for "YANIT:" tag; fall back to full text
        m = re.search(r"YANIT\s*:\s*(.+)", raw, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        return raw.strip()


PromptStrategyFactory.register("chain_of_thought", ChainOfThoughtStrategy)
