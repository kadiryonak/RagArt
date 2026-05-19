"""SQLite-backed generic key-value cache.

Tek tablo: `cache_entries(key TEXT PK, value BLOB, created REAL, ttl REAL)`.
Değer pickle'lanır → herhangi bir Python objesi tutulabilir (list[float],
str, dict). chromadb zaten sqlite kullandığı için ek dependency yok.

TTL semantiği:
    ttl = None / 0 → sınırsız (silinmez)
    ttl > 0       → created + ttl < now → expired, GET sırasında silinir

Lazy GC: süresi geçmiş kayıtları sadece okurken siliyoruz; periyodik
vacuum yok. Boyut sorun olursa clear() veya delete_expired() çağırılır.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from src.cache.base import BaseCache


class SQLiteCache(BaseCache):
    name = "sqlite"

    def __init__(self, db_path: str, *, table: str = "cache_entries"):
        super().__init__()
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # Table name validated at construction since we interpolate it raw
        if not table.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {table}")
        self.table = table
        # check_same_thread=False because Flask handles requests across
        # threads; we serialise our own ops via self._lock.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_table()

    def _init_table(self) -> None:
        with self._conn:
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ("
                "key TEXT PRIMARY KEY, "
                "value BLOB NOT NULL, "
                "created REAL NOT NULL, "
                "ttl REAL"
                ")"
            )

    @staticmethod
    def _is_expired(created: float, ttl: Optional[float], now: float) -> bool:
        if ttl is None or ttl <= 0:
            return False
        return (created + ttl) <= now

    def _get(self, key: str) -> Optional[Any]:
        row = self._conn.execute(
            f"SELECT value, created, ttl FROM {self.table} WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        value_blob, created, ttl = row
        if self._is_expired(created, ttl, time.time()):
            # Lazy GC for this entry
            with self._conn:
                self._conn.execute(
                    f"DELETE FROM {self.table} WHERE key = ?", (key,)
                )
            return None
        try:
            return pickle.loads(value_blob)
        except Exception:
            # Corrupt entry — purge it
            with self._conn:
                self._conn.execute(
                    f"DELETE FROM {self.table} WHERE key = ?", (key,)
                )
            return None

    def _set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        blob = pickle.dumps(value)
        with self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {self.table} "
                "(key, value, created, ttl) VALUES (?, ?, ?, ?)",
                (key, blob, time.time(), ttl),
            )

    def _delete(self, key: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                f"DELETE FROM {self.table} WHERE key = ?", (key,)
            )
            return cur.rowcount > 0

    def clear(self) -> int:
        with self._lock:
            with self._conn:
                cur = self._conn.execute(f"DELETE FROM {self.table}")
                n = cur.rowcount
            self._stats = type(self._stats)()  # reset stats too
            return n

    def keys(self) -> List[str]:
        return [
            r[0] for r in
            self._conn.execute(f"SELECT key FROM {self.table}").fetchall()
        ]

    def count(self) -> int:
        return self._conn.execute(
            f"SELECT COUNT(*) FROM {self.table}"
        ).fetchone()[0]

    def size_bytes(self) -> int:
        """Total size of the SQLite file on disk (approximation)."""
        try:
            return os.path.getsize(self.db_path)
        except OSError:
            return 0

    def stats(self):
        s = super().stats()
        s.item_count = self.count()
        s.size_bytes = self.size_bytes()
        return s

    def iter_with_metadata(self) -> List[Tuple[str, Any]]:
        """Tüm girişleri (key, value) olarak döndür — semantic cache için."""
        out = []
        now = time.time()
        for key, blob, created, ttl in self._conn.execute(
            f"SELECT key, value, created, ttl FROM {self.table}"
        ).fetchall():
            if self._is_expired(created, ttl, now):
                continue
            try:
                out.append((key, pickle.loads(blob)))
            except Exception:
                continue
        return out

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
