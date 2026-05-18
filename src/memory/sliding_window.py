"""SlidingWindowMemory — son N turn'ü tut.

NASIL ÇALIŞIR?
    History'nin son `window_size` turn'ünü alır, formatlı string döner.
    Bir "turn" = bir kullanıcı mesajı veya bir asistan cevabı (yani
    tipik bir sohbet adımı 2 turn = user+assistant).

NE ZAMAN İYİ?
    - Kısa sohbetlerde, tüm history kullanılabilir
    - "Bunu daha basit anlat", "bir önceki cevabını detaylandır" gibi
      yakın referansları yakalar
    - Sıfır maliyet, deterministik, debug edilebilir

NE ZAMAN YETERSİZ?
    - 20+ turn'lük uzun sohbet — window kayar, eski bilgi unutulur
    - Asistanın ilk açıklamasına referans verilen geç sorularda
"""

from __future__ import annotations

from typing import List

from src.memory.base import BaseMemory, ConversationTurn, format_turns


class SlidingWindowMemory(BaseMemory):
    name = "sliding_window"

    def __init__(self, window_size: int = 5):
        """
        Args:
            window_size: kaç tur tutulacak (user+assistant ÇİFTİ olarak).
                         5 = son 5 user mesajı + onların yanıtları.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size

    def apply(self, history: List[ConversationTurn], query: str) -> str:
        if not history:
            return ""
        # Çift = user+assistant; 2*window_size raw turn
        recent = history[-(self.window_size * 2):]
        return format_turns(recent)
