"""Direct strategy — improved default prompt.

İyileştirmeler vs eski TURKISH_SYSTEM_PROMPT:
    - Üçüncü tekil şahıs zorunluluğu (CV gibi 1. tekil yazılı bağlamlarda
      "Ben Junior AI Engineer adayıyım" → "Junior AI Engineer adayıdır")
    - Tekrar yasak
    - Source etiketleri YASAK
    - Soru-format'tan promtt sızıntısı yok (ipuçlarını LLM'e meta olarak
      açıklamıyor)
    - XOR cevap/red mantığı net
"""

from __future__ import annotations

from typing import Any

from src.prompt_strategies.base import BasePromptStrategy, PromptStrategyFactory


_TEMPLATE_NO_MEMORY = """Sen Türkçe konuşan bir uzman bilgi sistemi asistanısın. Görevin BAĞLAM bilgilerini kullanarak SORU'yu açık, doğal ve doğru bir Türkçe ile yanıtlamak.

YAZIM KURALLARI (çok önemli):
1. Üçüncü tekil şahıs kullan. Bağlam birinci tekil şahısta yazılmışsa ("yapıyorum", "biliyorum", "öğrenciyim") cümle yapısını dönüştür ("yapıyor", "biliyor", "öğrencidir").
2. Aynı bilgiyi iki kez yazma.
3. BAĞLAM içindeki "[Source N - ...]" gibi etiketleri ASLA cevapta gösterme.
4. Meta yorum yok ("Bağlama göre...", "Verilen bilgilerde...").
5. Akıcı, doğal Türkçe — bir bilgi kartı/özet formu, ham veri kopyası değil.

YANITLAMA MANTIĞI (XOR — birini seç):
- BAĞLAM soruyu cevaplamak için yeterliyse → doğrudan, akıcı bir cevap yaz.
- BAĞLAM yetersizse → SADECE şu cümleyi yaz, başka hiçbir şey ekleme:
  "Bu konuda verilen bilgilerde yeterli detay bulunmuyor."

BAĞLAM:
{context}

SORU: {question}

YANIT:"""


_TEMPLATE_WITH_MEMORY = """Sen Türkçe konuşan bir uzman bilgi sistemi asistanısın. Görevin BAĞLAM bilgilerini ve önceki KONUŞMA'yı kullanarak SORU'yu doğru ve doğal bir Türkçe ile yanıtlamak.

YAZIM KURALLARI (çok önemli):
1. Üçüncü tekil şahıs kullan. Bağlam birinci tekil şahısta yazılmışsa cümleyi dönüştür.
2. Aynı bilgiyi iki kez yazma.
3. "[Source N - ...]" / "BAĞLAM" / meta ifadeler YASAK.
4. Akıcı, doğal Türkçe.

KONUŞMA geçmişini sadece "o", "bu konu" gibi referansları çözmek için kullan.

YANITLAMA MANTIĞI (XOR):
- BAĞLAM yeterliyse → doğrudan cevap.
- BAĞLAM yetersizse → SADECE "Bu konuda verilen bilgilerde yeterli detay bulunmuyor."

ÖNCEKİ KONUŞMA:
{memory_context}

BAĞLAM:
{context}

SORU: {question}

YANIT:"""


class DirectStrategy(BasePromptStrategy):
    name = "direct"
    label = "Direkt cevap (önerilen)"
    description_tr = (
        "Tek LLM çağrısı, retrieval'dan gelen bağlamı doğrudan kullanır. "
        "Çoğu sorgu için en hızlı ve ekonomik strateji. Default."
    )

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


PromptStrategyFactory.register("direct", DirectStrategy)
