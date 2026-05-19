"""Self-Refine — LLM kendi cevabını kritik edip düzeltir.

PIPELINE
    1. INITIAL: Direct strategy ile ilk cevap üret.
    2. CRITIQUE: LLM'e "bu cevabı kritik et, eksik/yanlış yerleri yaz" der.
    3. REFINE: Kritiği göz önünde bulundurarak cevabı yeniden yaz.

Eğer critique "cevap doğru ve eksiksiz" diyorsa adım 3 atlanır (1 ekstra
çağrı tasarrufu).

NE ZAMAN İYİ?
    - Yüksek-paydaş cevaplar (yasal/tıbbi/finansal)
    - Faithfulness kritik (halüsinasyon koruması)
    - Kalite > latency tradeoff'u kabul edilebilirse

KUSURLAR
    - +1 ila +2 LLM çağrısı (latency 2-3x)
    - Maliyet ~3x
    - Bazen self-critique aşırı düzenleme yapıp orijinal cevabı bozar

ÖLÇÜLEN ETKİ (literatür)
    - Madaan et al. 2023 — Self-Refine: GPT-3.5 ile %5-15% kalite artışı
    - Reflexion (Shinn et al.) — agent task'larda %20+ ama tek-soruda
      daha mütevazi
"""

from __future__ import annotations

from typing import Any

from src.prompt_strategies.base import (
    BasePromptStrategy,
    PromptStrategyFactory,
    StrategyContext,
)
from src.prompt_strategies.direct import _TEMPLATE_NO_MEMORY, _TEMPLATE_WITH_MEMORY


_CRITIQUE_PROMPT = """Aşağıdaki SORU'ya verilen CEVAP'ı tarafsız ve kısa şekilde kritik et.

BAKACAK NOKTALAR:
- Cevap BAĞLAM'a SADIK mi? (halüsinasyon var mı?)
- Eksik bilgi var mı?
- Tutarsızlık veya çelişki var mı?
- Üçüncü tekil şahıs kuralına uyuyor mu? Tekrar var mı?

ÇIKTI FORMATI:
- Cevap tamamen doğruysa SADECE şu cümleyi yaz: "Cevap doğru ve eksiksiz."
- Aksi takdirde 1-3 madde halinde sorunları yaz.

BAĞLAM:
{context}

SORU: {question}

CEVAP:
{initial}

KRİTİK:"""


_REFINE_PROMPT = """Aşağıdaki SORU'ya verilen İLK CEVAP'ı KRİTİK'i göz önünde bulundurarak yeniden yaz.

KURALLAR:
1. Üçüncü tekil şahıs kullan.
2. "[Source N]" etiketleri YASAK.
3. Sadece düzeltilmiş cevabı yaz; meta yorum yok.
4. Akıcı doğal Türkçe.

BAĞLAM:
{context}

SORU: {question}

İLK CEVAP:
{initial}

KRİTİK:
{critique}

DÜZELTİLMİŞ CEVAP:"""


_OK_MARKERS = ("doğru ve eksiksiz", "dogru ve eksiksiz")


class SelfRefineStrategy(BasePromptStrategy):
    name = "self_refine"
    label = "Self-Refine (kritik et + düzelt)"
    description_tr = (
        "Cevap üretilir, LLM kendi cevabını kritik eder, sonra düzeltir. "
        "Kalite ve faithfulness için +%5-15. Latency 2-3x, maliyet ~3x."
    )
    is_multi_call = True
    is_advanced = True

    def build_prompt(
        self,
        *,
        question: str,
        context: str,
        memory_context: str = "",
        **kwargs: Any,
    ) -> str:
        # Used for INITIAL answer; refine orchestrated in execute()
        if memory_context.strip():
            return _TEMPLATE_WITH_MEMORY.format(
                memory_context=memory_context,
                context=context,
                question=question,
            )
        return _TEMPLATE_NO_MEMORY.format(context=context, question=question)

    def execute(
        self,
        ctx: StrategyContext,
        *,
        question: str,
        context: str,
        memory_context: str = "",
        **kwargs: Any,
    ) -> str:
        # Step 1: initial answer
        initial_prompt = self.build_prompt(
            question=question, context=context, memory_context=memory_context,
        )
        initial = ctx.llm.generate(initial_prompt, **ctx.llm_params)
        if not initial or not initial.strip():
            return initial

        # Step 2: critique
        try:
            critique = ctx.llm.generate(
                _CRITIQUE_PROMPT.format(
                    context=context, question=question, initial=initial.strip(),
                ),
                **ctx.llm_params,
            )
        except Exception:
            # If critique fails, just return the initial answer
            return initial

        critique_lower = (critique or "").lower()
        if any(m in critique_lower for m in _OK_MARKERS):
            # LLM agreed the initial answer is fine — skip the refine call.
            return initial

        # Step 3: refine using the critique
        try:
            refined = ctx.llm.generate(
                _REFINE_PROMPT.format(
                    context=context,
                    question=question,
                    initial=initial.strip(),
                    critique=(critique or "").strip(),
                ),
                **ctx.llm_params,
            )
        except Exception:
            return initial

        return (refined or "").strip() or initial


PromptStrategyFactory.register("self_refine", SelfRefineStrategy)
