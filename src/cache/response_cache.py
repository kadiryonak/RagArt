"""ResponseCache — full request hash → cached answer (exact match).

Cache key, sorgu cevabını değiştirebilecek HER şeyi içerir:
    question, workspace_id, provider, model,
    prompt_strategy, custom_role, custom_template,
    retrieval_strategy, rerank, k,
    deduplicate_context, reorder_context, max_context_tokens,
    memory_strategy, history hash, llm_params hash.

Aynı anahtarlı tekrar sorgu → cache hit, LLM ÇAĞRISI YAPILMAZ.

TTL: default 1 saat (LLM cevapları zamana bağlı değil ama veri tabanı
güncellenebilir; conservative default). Caller ttl=None ile sınırsız
yapabilir, ttl=0 ile cache devre dışı.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from src.cache.sqlite_cache import SQLiteCache


def _stable_json(obj: Any) -> str:
    """Deterministic JSON for hashing (keys sorted)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)


class ResponseCache:
    """Exact-match response cache."""

    DEFAULT_TTL_SECONDS = 3600  # 1 hour

    def __init__(self, db_path: str, *, default_ttl: Optional[float] = DEFAULT_TTL_SECONDS):
        self._store = SQLiteCache(db_path, table="response_cache")
        self.default_ttl = default_ttl

    @property
    def store(self) -> SQLiteCache:
        return self._store

    @staticmethod
    def make_key(payload: Dict[str, Any]) -> str:
        """Generate a stable hash from any params dict."""
        canon = _stable_json(payload)
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def get(self, payload: Dict[str, Any]) -> Optional[Any]:
        return self._store.get(self.make_key(payload))

    def set(
        self,
        payload: Dict[str, Any],
        value: Any,
        ttl: Optional[float] = "default",  # sentinel
    ) -> None:
        if ttl == "default":
            ttl = self.default_ttl
        self._store.set(self.make_key(payload), value, ttl=ttl)

    def invalidate_workspace(self, workspace_id: str) -> int:
        """Workspace reindex'i sonrası ilgili cevap cache'lerini temizle.

        Şu an basit: tümünü siler (workspace ayrımı yok). İleride
        per-workspace tabular ayırım kolayca eklenir.
        """
        return self._store.clear()

    def stats(self) -> dict:
        s = self._store.stats().to_dict()
        s["name"] = "response"
        s["default_ttl"] = self.default_ttl
        return s

    def clear(self) -> int:
        return self._store.clear()
