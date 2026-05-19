"""Step-back prompting — soyut/genel soruyla geniş context çek.

NEDEN?
    Spesifik sorular ("Python list comprehension ile for loop hız farkı
    ne?") doğrudan arandığında dar bir sonuç kümesi döner. Step-back,
    soruyu önce daha genel bir seviyeye çıkarır ("Python list
    comprehension nedir?") ve o genel sorunun sonuçlarını da context'e
    ekler.

    Bu sayede LLM hem genel arka planı hem spesifik detayı görerek
    daha kapsamlı bir cevap verir.

MALİYET
    +1 LLM çağrısı (step-back question üretimi).
    Retrieval: 2 (orijinal + step-back).

NE ZAMAN İYİ?
    - "Neden X?" gibi reasoning gerektiren sorular
    - Karşılaştırma soruları
    - Öncül bilgi gerektiren spesifik sorular

NE ZAMAN ZAYIF?
    - Soru zaten yeterince genel
    - Basit tanım soruları ("X nedir?")
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


_STEPBACK_PROMPT = """Aşağıdaki soruyu analiz et ve onun ARKA PLANINDAKİ daha genel bir soru yaz.

ÖRNEK:
  Soru: "Python'da list comprehension ile for loop hız farkı ne?"
  Step-back: "Python'da list comprehension nedir ve nasıl çalışır?"

  Soru: "React ile Vue arasında state management farkı ne?"
  Step-back: "React ve Vue'de state management nasıl çalışır?"

KURALLAR:
1. Tek bir soru yaz.
2. Daha genel/soyut ol ama aynı konu alanında kal.
3. Açıklama, numara, etiket EKLEME. Sadece soruyu döndür.

ORİJİNAL SORU: {question}

STEP-BACK SORU:"""


_PREFIX_RE = re.compile(
    r"^(step[- ]?back\s*(soru(su)?)?|genel\s+soru)\s*:?\s*",
    flags=re.IGNORECASE,
)


class StepBackStrategy(BasePromptStrategy):
    name = "step_back"
    label = "Step-back (genel→özel)"
    description_tr = (
        "Soruyu daha genel bir seviyeye çıkarıp, hem genel hem spesifik "
        "context'le cevap verir. Reasoning ve karşılaştırma sorgularında "
        "kapsamlı cevap üretir. +1 LLM çağrısı."
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
                _STEPBACK_PROMPT.format(question=question),
                **ctx.llm_params,
            )
        except Exception:
            return [question]

        # Parse: take first non-empty line, strip prefix
        step_back = ""
        for line in (raw or "").split("\n"):
            line = line.strip()
            if not line:
                continue
            line = _PREFIX_RE.sub("", line).strip().strip("\"'")
            if line:
                step_back = line
                break

        if not step_back or step_back.lower() == question.lower():
            return [question]

        return [question, step_back]

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


PromptStrategyFactory.register("step_back", StepBackStrategy)
