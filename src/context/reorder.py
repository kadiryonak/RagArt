"""LostInTheMiddleReorderer — uçlara güçlü chunk'ları yerleştir.

ARAŞTIRMA (Liu et al. 2024 — Lost in the Middle):
    LLM'ler uzun context'in BAŞ ve SON kısmındaki bilgilere daha çok dikkat
    eder, ORTA kısımdaki bilgileri unutur. Bu, retrieval ile gelen
    top-relevance chunk'ları context'in ortasına koymanın kalite kaybına
    yol açtığını gösterir.

ÇÖZÜM
    Top-N relevance-sıralı doküman listesini şu deseninle yeniden sırala:
        relevance_sorted   = [d0, d1, d2, d3, d4, d5, d6, d7]   (azalan)
        reordered          = [d0, d2, d4, d6, d7, d5, d3, d1]
                              ↑          ↑    ↑          ↑
                              başta en iyi   sonda 2. en iyi

    Yani: tek-indeksli elemanlar başa, çift-indeksli sondan başlayarak
    sona — orta kısma en zayıf chunk'lar gelir.

NE ZAMAN İYİ?
    Context boyutu büyükse (5+ chunk). Tek/iki chunk'la fayda sıfır.

NE ZAMAN İYİ DEĞİL?
    LLM zaten kısa context kullanıyorsa veya sıralama önemsizse.
    O zaman NoOp tut.
"""

from __future__ import annotations

from typing import List

from src.context.base import BaseContextProcessor
from src.retrievers.base import RetrievedDoc


class LostInTheMiddleReorderer(BaseContextProcessor):
    name = "lost_in_middle"

    def __init__(self, *, min_docs: int = 3):
        """
        Args:
            min_docs: bundan az dokümanda yeniden sıralama yapılmaz
                     (etkisi yok, gereksiz hesaplama).
        """
        self.min_docs = max(2, int(min_docs))

    def process(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        if len(docs) < self.min_docs:
            return docs

        # Lost-in-the-middle pattern:
        # odd-indexed (0, 2, 4, ...) → başa
        # even-indexed (1, 3, 5, ...) → sondan başlayarak sona
        front: List[RetrievedDoc] = []
        back: List[RetrievedDoc] = []
        for i, doc in enumerate(docs):
            if i % 2 == 0:
                front.append(doc)
            else:
                back.append(doc)

        # back listesini ters çevir ki en zayıf orta noktada olsun
        back.reverse()
        return front + back
