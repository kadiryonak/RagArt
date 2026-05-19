"""EmbeddingCache — embed_query / embed_documents sonuçlarını cache'le.

NEDEN?
    Embedding modeli CPU'da ~50ms/text (multilingual-MiniLM). Bir
    workspace üzerinde 100 sorgu yapılırsa 100 embed çağrısı yapılıyor.
    Aynı sorular tekrarlandığında veya retrieval içinde aynı chunk'lar
    re-embed edildiğinde tasarruf büyük.

KEY
    hash(text) — SHA1 yeterli, model değişimi rare olduğundan.

TTL
    None (sınırsız). Model stabil olduğu sürece cevap aynı; model
    değişirse cache'i kullanıcı manuel clear etmeli (modeller workspace
    bazlı değil, global; pratik bir senaryo değil).
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, List, Optional

from src.cache.sqlite_cache import SQLiteCache


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Wrap an embedder; transparent cache."""

    def __init__(self, db_path: str, embedder: Any):
        """
        Args:
            db_path: SQLite dosya yolu.
            embedder: bir LangChain Embeddings benzeri obje:
                      .embed_query(text) -> List[float]
                      .embed_documents(texts) -> List[List[float]]
        """
        self._store = SQLiteCache(db_path, table="embedding_cache")
        self._embedder = embedder

    @property
    def store(self) -> SQLiteCache:
        return self._store

    def embed_query(self, text: str) -> List[float]:
        key = _hash_text(text)
        cached = self._store.get(key)
        if cached is not None:
            return cached
        vec = self._embedder.embed_query(text)
        self._store.set(key, vec)
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # Hash each, fetch what we can, batch-embed the rest, then merge
        # back into original order.
        keys = [_hash_text(t) for t in texts]
        results: List[Optional[List[float]]] = [None] * len(texts)
        misses_idx: List[int] = []
        for i, k in enumerate(keys):
            v = self._store.get(k)
            if v is not None:
                results[i] = v
            else:
                misses_idx.append(i)

        if misses_idx:
            miss_texts = [texts[i] for i in misses_idx]
            fresh = self._embedder.embed_documents(miss_texts)
            for j, idx in enumerate(misses_idx):
                results[idx] = fresh[j]
                self._store.set(keys[idx], fresh[j])

        return results  # type: ignore[return-value]

    def stats(self) -> dict:
        s = self._store.stats().to_dict()
        s["name"] = "embedding"
        return s

    def clear(self) -> int:
        return self._store.clear()
