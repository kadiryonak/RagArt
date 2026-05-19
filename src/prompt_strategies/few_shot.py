"""Few-shot — modelin format/üslubu kavraması için örnek Q&A çiftleri.

NEDEN?
    LLM, beklediğin cevap formatına/üslubuna örnek görerek daha
    tutarlı cevaplar üretir. Özellikle kısa-cevap, tablo, JSON gibi
    yapılandırılmış çıktılarda etkili.

NE ZAMAN İYİ?
    - Belirli bir format isteniyorsa (ör. "1-2 cümle özet")
    - Default LLM çıktısı çok uzun/dağınık
    - Domain-specific terim/tonu kazandırmak istiyorsanız

KUSURLAR
    - Token maliyeti artar (her örnek 50-100 token)
    - Yanlış örnek = yanlış format kalıcı sızıntı
"""

from __future__ import annotations

from typing import Any, List, Tuple

from src.prompt_strategies.base import BasePromptStrategy, PromptStrategyFactory


_TEMPLATE = """Sen Türkçe konuşan bir uzman asistanısın. Aşağıdaki örnekleri inceleyip aynı stilde, doğal Türkçe cevap ver.

KURALLAR:
1. Üçüncü tekil şahıs.
2. "[Source N]" etiketleri YASAK.
3. Aynı bilgiyi tekrarlama.
4. BAĞLAM yetersizse SADECE: "Bu konuda verilen bilgilerde yeterli detay bulunmuyor."

ÖRNEKLER (sadece stil için, kendi içeriklerini cevapta KULLANMA):
{examples}

GERÇEK SORU:

BAĞLAM:
{context}

SORU: {question}

YANIT:"""


DEFAULT_EXAMPLES: List[Tuple[str, str, str]] = [
    (
        "Algoritma adım adım çözüm sunan sonlu işlemler kümesidir. Bilgisayar biliminin temelidir.",
        "Algoritma nedir?",
        "Algoritma, bir problemi çözmek için tasarlanmış, başlangıç ve son durumları belirli, sonlu adımlardan oluşan işlemler kümesidir. Bilgisayar biliminin temelinde yer alır.",
    ),
    (
        "Python yorumlamalı bir programlama dilidir, 1991'de Guido van Rossum tarafından yazıldı.",
        "Python ne zaman çıktı?",
        "Python, Guido van Rossum tarafından 1991 yılında geliştirildi ve yayımlandı.",
    ),
]


def _format_examples(examples: List[Tuple[str, str, str]]) -> str:
    parts = []
    for i, (ctx, q, a) in enumerate(examples, 1):
        parts.append(
            f"--- Örnek {i} ---\n"
            f"BAĞLAM: {ctx}\nSORU: {q}\nYANIT: {a}"
        )
    return "\n\n".join(parts)


class FewShotStrategy(BasePromptStrategy):
    name = "few_shot"
    label = "Few-shot (örnekli)"
    description_tr = (
        "Prompt'a 2-3 örnek Q&A koyar, LLM bu stille cevap verir. "
        "Token maliyetini biraz artırır; format/üslubu sabitlemek için iyi."
    )

    def __init__(self, examples: List[Tuple[str, str, str]] | None = None):
        self.examples = examples or DEFAULT_EXAMPLES

    def build_prompt(
        self,
        *,
        question: str,
        context: str,
        memory_context: str = "",
        **kwargs: Any,
    ) -> str:
        return _TEMPLATE.format(
            examples=_format_examples(self.examples),
            context=context,
            question=question,
        )


PromptStrategyFactory.register("few_shot", FewShotStrategy)
