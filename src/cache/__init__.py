"""Caching layer — embedding + response + semantic caches.

Üç katman:
    EmbeddingCache  — embed_query/documents çağrılarını cache'ler. Aynı
                       metin tekrar embed edilmez. Tüm workspace'ler
                       paylaşır (embedding modeli stateless).
    ResponseCache   — (question + workspace + provider + model + strategy
                       + params) hash'i ile cevabı cache'ler. Exact match.
    SemanticCache   — Aynı anlamlı farklı cümleler için. Cache hit
                       kriteri: query embedding'i cache'lenmiş
                       sorgulardan birine cos>threshold.

Hepsi SQLite tabanlı (chromadb zaten sqlite kullanıyor; ek bağımlılık
yok). Cache dosyası per-workspace olabilir veya global — config'e bağlı.
"""

from src.cache.base import BaseCache, CacheStats
from src.cache.sqlite_cache import SQLiteCache
from src.cache.embedding_cache import EmbeddingCache
from src.cache.response_cache import ResponseCache
from src.cache.semantic_cache import SemanticCache

__all__ = [
    "BaseCache",
    "CacheStats",
    "SQLiteCache",
    "EmbeddingCache",
    "ResponseCache",
    "SemanticCache",
]
