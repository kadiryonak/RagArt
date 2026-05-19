"""BaseCache sözleşmesi + telemetri.

Cache get/set/delete/clear + hit/miss istatistikleri. TTL opsiyonel —
0/None = sınırsız (embedding cache için ideal, çünkü model stateless).
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CacheStats:
    """Hit/miss/size telemetri. /cache/stats endpoint'inden döner."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    size_bytes: int = 0
    item_count: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "size_bytes": self.size_bytes,
            "item_count": self.item_count,
            "hit_rate": round(self.hit_rate, 4),
            **self.extra,
        }


class BaseCache(ABC):
    """Tüm cache implementasyonlarının temel sınıfı."""

    name: str = "base"

    def __init__(self):
        self._stats = CacheStats()
        self._lock = threading.Lock()

    @abstractmethod
    def _get(self, key: str) -> Optional[Any]:
        """Concrete'in fetch implementasyonu."""

    @abstractmethod
    def _set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Concrete'in store implementasyonu."""

    @abstractmethod
    def _delete(self, key: str) -> bool:
        """Concrete'in delete; True/False'a göre eviction sayar."""

    @abstractmethod
    def clear(self) -> int:
        """Tüm cache'i temizle, silinen sayısını döndür."""

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            value = self._get(key)
            if value is not None:
                self._stats.hits += 1
            else:
                self._stats.misses += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            self._set(key, value, ttl)
            self._stats.sets += 1

    def delete(self, key: str) -> bool:
        with self._lock:
            ok = self._delete(key)
            if ok:
                self._stats.evictions += 1
            return ok

    def stats(self) -> CacheStats:
        return self._stats

    def reset_stats(self) -> None:
        with self._lock:
            self._stats = CacheStats()
