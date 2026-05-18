"""SummaryBufferMemory — eski turn'leri LLM ile özetle, son N'i ham tut.

NASIL ÇALIŞIR?
    History `keep_recent` turn'den uzunsa:
        eski = history[:-keep_recent]
        son  = history[-keep_recent:]
        özet = LLM.summarize(eski)
        return f"Önceki özet:\n{özet}\n\nSon mesajlar:\n{son}"

    Kısa history'de SlidingWindow gibi davranır (özet çağrısı yapmaz).

NE ZAMAN İYİ?
    - 10+ turn'lük uzun sohbetlerde eski bağlamı kaybetmemek için
    - Token budget'a duyarlı: özet hep aynı boyutta kalır

MALİYET
    Her çağrıda ÖZET LLM çağrısı yeniden yapılır (çünkü stateless).
    Production'da: client previous_summary'i tutup yollar → server
    incremental update yapar. v1'de yapmıyoruz (basitlik).
"""

from __future__ import annotations

from typing import Any, List, Optional

from src.memory.base import BaseMemory, ConversationTurn, format_turns


SUMMARY_PROMPT_TR = """Aşağıdaki sohbeti birkaç cümlede özetle. Kullanıcının niyetini, sorduğu temel konuyu ve asistanın verdiği önemli bilgileri koru. Madde işareti kullanma; akıcı düz metin yaz.

SOHBET:
{conversation}

ÖZET:"""


class SummaryBufferMemory(BaseMemory):
    name = "summary_buffer"

    def __init__(
        self,
        llm: Any,
        *,
        keep_recent: int = 4,
        summarize_threshold: int = 8,
    ):
        """
        Args:
            llm: BaseLLMProvider — özet için. .generate(prompt) çağrılır.
            keep_recent: son kaç turn ham olarak tutulsun
            summarize_threshold: history bu kadar turn'den uzunsa özet üret
                                 (kısa history'de skip → cost yok)
        """
        if keep_recent < 1:
            raise ValueError("keep_recent must be >= 1")
        if summarize_threshold < keep_recent:
            raise ValueError("summarize_threshold must be >= keep_recent")
        self.llm = llm
        self.keep_recent = keep_recent
        self.summarize_threshold = summarize_threshold

    def apply(self, history: List[ConversationTurn], query: str) -> str:
        if not history:
            return ""

        # Kısa history: özet çağrısı yapmadan tüm history'i ham döndür
        if len(history) <= self.summarize_threshold:
            return format_turns(history)

        # Uzun history: ayır + özet üret
        cutoff = len(history) - self.keep_recent
        old_turns = history[:cutoff]
        recent_turns = history[cutoff:]

        old_formatted = format_turns(old_turns)
        prompt = SUMMARY_PROMPT_TR.format(conversation=old_formatted)
        summary = self.llm.generate(prompt)

        return (
            f"Önceki konuşma özeti:\n{summary.strip()}\n\n"
            f"Son mesajlar:\n{format_turns(recent_turns)}"
        )
