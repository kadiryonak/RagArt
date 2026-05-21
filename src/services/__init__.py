"""Service layer — route handler'ların çağırdığı iş mantığı.

Faz B'nin amacı app.py'yi inceltmek: HTTP endpoint'leri parsing +
delegasyona indirgeyip, asıl işi test edilebilir servislere taşımak.

    - RagRegistry : workspace başına TurkishRAGSystem yaşam döngüsü
                    (lazy build, thread-safe cache, invalidation)
"""

from src.services.rag_registry import RagRegistry

__all__ = ["RagRegistry"]
