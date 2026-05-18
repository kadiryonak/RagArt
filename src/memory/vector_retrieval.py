"""VectorRetrievalMemory — semantic retrieval over chat history.

NASIL ÇALIŞIR?
    Her geçmiş turn embedding'lenir; current query ile en benzer
    `top_k` turn'ü çekilir, format'lanır.

NE ZAMAN İYİ?
    - Çok uzun sohbet (50+ turn), sıralı window tüm konuyu kaybeder
    - "Önceden o algoritmadan bahsetmiştik" gibi semantic referanslar
    - Long-term memory için (bu v1'de stateless ama protokol uygun)

MALİYET
    Embedding hesaplama (history boyu kadar). Mevcut RAG'in embedder'ı
    paylaşılabilir → ek model yükü yok.

ALTERNATİF
    Production'da embedding sonuçları cache'lenir (turn_id → vector
    pickle). v1'de stateless: her çağrıda yeniden hesap.
"""

from __future__ import annotations

from typing import Callable, List

from src.memory.base import BaseMemory, ConversationTurn, format_turns


EmbedFn = Callable[[str], List[float]]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VectorRetrievalMemory(BaseMemory):
    name = "vector"

    def __init__(self, embed_fn: EmbedFn, *, top_k: int = 3):
        """
        Args:
            embed_fn: text → embedding vector. EmbeddingManager.embed_query
                      tipik.
            top_k: kaç turn döndürülecek
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.embed_fn = embed_fn
        self.top_k = top_k

    def apply(self, history: List[ConversationTurn], query: str) -> str:
        if not history:
            return ""

        # Çok kısa history → sliding window davranışı (tümünü döndür)
        if len(history) <= self.top_k * 2:
            return format_turns(history)

        try:
            query_vec = self.embed_fn(query)
        except Exception:
            # Embedding hatasında graceful fallback: son turn'leri ver
            return format_turns(history[-(self.top_k * 2):])

        scored: List[tuple[float, ConversationTurn]] = []
        for turn in history:
            try:
                turn_vec = self.embed_fn(turn.content)
                sim = _cosine(query_vec, turn_vec)
                scored.append((sim, turn))
            except Exception:
                continue

        # En relevant top_k pair (user+assistant) için 2*top_k turn al
        scored.sort(key=lambda x: -x[0])
        picked = [t for _, t in scored[: self.top_k * 2]]

        # Orijinal kronolojik sıraya geri döndür (LLM için)
        picked_set = {id(t) for t in picked}
        chronological = [t for t in history if id(t) in picked_set]
        return format_turns(chronological)
