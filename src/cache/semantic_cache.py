"""SemanticCache — embedding similarity tabanlı cache.

NEDEN?
    Exact ResponseCache aynı stringi yakalar ama "Paris'in nüfusu kaç?"
    ve "Paris'te kaç kişi yaşıyor?" farklı string olduğundan miss verir.
    Semantic cache her cache'lenmiş cevap için embedding tutar; yeni
    sorgu için cosine sim > threshold ise hit.

MALİYET
    - Set: 1 embedding hesaplama (zaten EmbeddingCache var → ucuz)
    - Get: tüm cache embeddingleri ile cosine (N=100 entry → ~1ms)

LIMITS
    Threshold ile yanlış pozitif olabilir → kullanıcı toleransına bağlı
    setting. Default 0.92 (aynı niyet, farklı kelime).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from src.cache.sqlite_cache import SQLiteCache


EmbedFn = Callable[[str], List[float]]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class _Entry:
    """SQLite'a serialize edilen kayıt yapısı."""
    embedding: List[float]
    response: Any
    original_query: str


class SemanticCache:
    """Embedding-similarity'ye dayalı cache."""

    DEFAULT_THRESHOLD = 0.92

    def __init__(
        self,
        db_path: str,
        embed_fn: EmbedFn,
        *,
        similarity_threshold: float = DEFAULT_THRESHOLD,
    ):
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        self._store = SQLiteCache(db_path, table="semantic_cache")
        self._embed = embed_fn
        self.threshold = similarity_threshold

    @property
    def store(self) -> SQLiteCache:
        return self._store

    @staticmethod
    def _key_for(query: str) -> str:
        # Use raw query as key for lookup-by-text; SQLite will dedupe.
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    def get(self, query: str) -> Tuple[Optional[Any], float]:
        """Returns (response, best_similarity). If None → miss."""
        if not query.strip():
            return None, 0.0
        try:
            q_emb = self._embed(query)
        except Exception:
            return None, 0.0

        best: Tuple[Optional[Any], float, Optional[str]] = (None, 0.0, None)
        for _, entry in self._store.iter_with_metadata():
            if not isinstance(entry, _Entry):
                continue
            sim = _cosine(q_emb, entry.embedding)
            if sim > best[1]:
                best = (entry.response, sim, entry.original_query)

        response, sim, _ = best
        if response is not None and sim >= self.threshold:
            # Telemetry: count this as a real cache hit at the SQLite layer
            self._store._stats.hits += 1
            return response, sim

        self._store._stats.misses += 1
        return None, sim

    def set(self, query: str, response: Any, ttl: Optional[float] = None) -> None:
        if not query.strip():
            return
        try:
            emb = self._embed(query)
        except Exception:
            return
        entry = _Entry(embedding=emb, response=response, original_query=query)
        self._store.set(self._key_for(query), entry, ttl=ttl)

    def stats(self) -> dict:
        s = self._store.stats().to_dict()
        s["name"] = "semantic"
        s["threshold"] = self.threshold
        return s

    def clear(self) -> int:
        return self._store.clear()
