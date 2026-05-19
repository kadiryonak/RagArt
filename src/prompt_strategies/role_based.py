"""Role-based — kullanıcı tanımlı uzman rolü.

Kullanıcı UI'dan "Sen 20 yıllık deneyimli bir hukuk danışmanısın..." gibi
bir sistem rolü tanımlar; bu rol prompt'un başına eklenir.

NE ZAMAN İYİ?
    - Domain-specific cevap (hukuk, tıp, finans, kod-review)
    - Belirli bir kişilik/ton istendiğinde
    - Profesyonel chatbot UX

DİKKAT
    - "Sen Tanrı'sın" gibi aşırı rol = jailbreak riski
    - Çok detaylı rol = token maliyeti
"""

from __future__ import annotations

from typing import Any

from src.prompt_strategies.base import BasePromptStrategy, PromptStrategyFactory


_DEFAULT_ROLE = (
    "Sen Türkçe konuşan uzman bir bilgi sistemi asistanısın. "
    "Verilen bağlamı sadakatle kullanarak doğal, akıcı ve doğru cevaplar "
    "üretirsin."
)

_TEMPLATE = """{role}

KURALLAR:
1. Üçüncü tekil şahıs kullan; birinci şahıs kalıbı bağlamdan gelse bile dönüştür.
2. "[Source N]" etiketlerini cevapta gösterme.
3. Aynı bilgiyi tekrarlama.
4. BAĞLAM yetersizse SADECE: "Bu konuda verilen bilgilerde yeterli detay bulunmuyor."

BAĞLAM:
{context}

SORU: {question}

YANIT:"""


class RoleBasedStrategy(BasePromptStrategy):
    name = "role_based"
    label = "Uzman rol (kullanıcı tanımlı)"
    description_tr = (
        "Kullanıcının verdiği rol açıklaması prompt'un başına eklenir. "
        "Domain-spesifik cevap (hukuk, tıp, kod-review) için ideal."
    )

    def __init__(self, role: str | None = None):
        # Role boş veya None ise default'a düş; aşırı uzunsa kırp.
        cleaned = (role or "").strip()
        self.role = cleaned[:1500] if cleaned else _DEFAULT_ROLE

    def build_prompt(
        self,
        *,
        question: str,
        context: str,
        memory_context: str = "",
        **kwargs: Any,
    ) -> str:
        return _TEMPLATE.format(role=self.role, context=context, question=question)


PromptStrategyFactory.register("role_based", RoleBasedStrategy)
