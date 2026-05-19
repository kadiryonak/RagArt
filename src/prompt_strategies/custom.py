"""Custom — kullanıcı tanımlı tam prompt template.

İleri seviye kullanıcılar UI'da kendi prompt template'ini yazabilir.
Template'te şu placeholder'lar kullanılabilir:
    {question}        — kullanıcının sorusu
    {context}         — retrieved chunk'ların formatlanmış metni
    {memory_context}  — sohbet geçmişi (varsa, boş string olabilir)

Güvenlik: Template f-string olarak format edilir. Bilinmeyen
placeholder'lar (örn. {os}) KeyError fırlatır → kullanıcıya net hata.
"""

from __future__ import annotations

from typing import Any

from src.prompt_strategies.base import BasePromptStrategy, PromptStrategyFactory


_FALLBACK_TEMPLATE = (
    "Sen Türkçe konuşan bir asistanısın.\n\n"
    "BAĞLAM:\n{context}\n\n"
    "SORU: {question}\n\nYANIT:"
)


class CustomStrategy(BasePromptStrategy):
    name = "custom"
    label = "Özel template (geliştirici)"
    description_tr = (
        "Kendi prompt template'inizi yazın. {question}, {context}, "
        "{memory_context} placeholder'larını kullanabilirsiniz."
    )
    # Tam prompt yazmak placeholder bilgisi gerektirir → geliştirici modu.
    is_advanced = True

    def __init__(self, template: str | None = None):
        # Whitelist of allowed placeholders — anything else won't format.
        self.template = (template or "").strip() or _FALLBACK_TEMPLATE

    def build_prompt(
        self,
        *,
        question: str,
        context: str,
        memory_context: str = "",
        **kwargs: Any,
    ) -> str:
        try:
            return self.template.format(
                question=question,
                context=context,
                memory_context=memory_context,
            )
        except KeyError as e:
            # Custom template referred to an unknown placeholder; surface
            # a helpful error rather than a stack trace.
            return (
                f"[Custom prompt template error] Bilinmeyen placeholder: {e}. "
                f"İzin verilenler: {{question}}, {{context}}, {{memory_context}}."
            )


PromptStrategyFactory.register("custom", CustomStrategy)
