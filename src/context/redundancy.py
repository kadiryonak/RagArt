"""RedundancyFilter — cosine similarity bazlı duplicate removal.

NEDEN?
    Retrieval (özellikle hybrid + reranker) bazen aynı dokümandan ardışık
    chunk'lar veya çok benzer paragraflar getirir. Bunların hepsi prompt'a
    girerse:
        - Token israfı (aynı bilgi tekrar)
        - LLM'in dikkati dağılır
        - Lost-in-the-middle riski büyür

NASIL?
    Sırayla docs üzerinde gez; daha önce kabul edilenlerle cosine
    similarity > threshold ise at, değilse kabul et.

    Karmaşıklık: O(N²) — N=10-20 olduğundan kabul edilebilir.

EMBEDDER
    Inject edilebilir (test için fake embedder). Üretimde
    EmbeddingManager.embed_query.
"""

from __future__ import annotations

from typing import Callable, List

from src.context.base import BaseContextProcessor
from src.retrievers.base import RetrievedDoc

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


class RedundancyFilter(BaseContextProcessor):
    name = "redundancy_filter"

    def __init__(self, embed_fn: EmbedFn, *, similarity_threshold: float = 0.92):
        """
        Args:
            embed_fn: text → embedding vector.
            similarity_threshold: 0-1 arası; bu eşiğin üstündeki çift
                                   duplicate sayılır. Default 0.92 ≈
                                   "çok benzer ama identik değil".
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold

    def process(self, query: str, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        if len(docs) < 2:
            return docs

        kept: List[RetrievedDoc] = []
        kept_vecs: List[List[float]] = []

        for doc in docs:
            try:
                vec = self.embed_fn(doc.page_content)
            except Exception:
                # Embedder hata verirse — kabul et, atlama
                kept.append(doc)
                kept_vecs.append([])
                continue

            duplicate = False
            for prev in kept_vecs:
                if prev and _cosine(vec, prev) >= self.similarity_threshold:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(doc)
                kept_vecs.append(vec)

        return kept
