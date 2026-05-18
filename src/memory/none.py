"""NoMemory — varsayılan, hafıza yok stratejisi.

Her sorgu izole; geçmişten bilgi taşımaz. Tek-turn senaryolarda doğru
varsayılan; çok-turn UX gerektiren senaryolarda kullanıcı UI'dan
başka strateji seçer.
"""

from __future__ import annotations

from typing import List

from src.memory.base import BaseMemory, ConversationTurn


class NoMemory(BaseMemory):
    name = "none"

    def apply(self, history: List[ConversationTurn], query: str) -> str:
        return ""
