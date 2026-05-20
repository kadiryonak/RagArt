"""RagRegistry — workspace başına RAG instance yaşam döngüsü.

NEDEN
    app.py'de bu mantık dağınık module-level global'lerdi: _rag_cache,
    _rag_init_lock, _build_rag_for_workspace, get_rag_for, invalidate_rag.
    Test edilemez, mock'lanamaz, app import'una bağlıydı.

    RagRegistry hepsini tek bir nesnede toplar — route handler artık
    sadece registry.get(ws_id) çağırır; cache + lock + lazy build
    detayları burada kapsüllenir.

TASARIM
    Her workspace NotebookLM tarzı izole bir bilgi tabanıdır: kendi dosya
    klasörü + kendi vektör DB'si. Embedder modeli süreç içinde paylaşılır
    ama vector store ayrıdır — bu yüzden workspace başına bir RAG instance
    lazy kurulur ve cache'lenir.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from config.settings import settings
from src.rag_system import TurkishRAGSystem
from src.utils import StatusEmoji, get_logger
from src.workspaces import WorkspaceManager

logger = get_logger(__name__)

# API key gerektiren provider'lar — biri seçili ama key yoksa local'e düşeriz.
_KEYED_PROVIDERS = ("deepseek", "openai", "groq", "huggingface")


class RagRegistry:
    """Workspace id → TurkishRAGSystem; lazy kurulum + thread-safe cache."""

    def __init__(self, workspace_manager: WorkspaceManager):
        self._wm = workspace_manager
        self._cache: Dict[str, TurkishRAGSystem] = {}
        self._lock = threading.Lock()

    def build(self, workspace_id: str) -> TurkishRAGSystem:
        """Bir workspace için taze RAG instance kur (cache'e yazmaz).

        API key gerektiren bir model seçilmiş ama key yoksa sessizce
        local modele düşülür — sunucu key olmadan da ayağa kalkmalı.
        """
        ws = self._wm.get(workspace_id)
        if ws is None:
            workspace_id = self._wm.resolve(workspace_id)
            ws = self._wm.get(workspace_id)

        api_key = settings.get_api_key()
        model_type = settings.MODEL_TYPE
        if model_type in _KEYED_PROVIDERS and not api_key:
            logger.warning(
                f"{StatusEmoji.WARNING} No API key found, falling back to local model"
            )
            model_type = "local"

        persist_path = self._wm.vector_db_path(workspace_id, ws.vector_db)
        rag = TurkishRAGSystem(
            data_folder=str(self._wm.files_dir(workspace_id)),
            model_type=model_type,
            api_key=api_key,
            chroma_db_path=str(persist_path),
        )
        rag.initialize()
        return rag

    def get(self, workspace_id: str) -> TurkishRAGSystem:
        """Cache'lenmiş RAG'i döndür; yoksa lock altında lazy kur.

        Çift kontrol (lock öncesi + sonrası): aynı anda gelen iki istek
        aynı workspace'i iki kez build etmesin.
        """
        workspace_id = self._wm.resolve(workspace_id)
        cached = self._cache.get(workspace_id)
        if cached is not None:
            return cached
        with self._lock:
            if workspace_id not in self._cache:
                self._cache[workspace_id] = self.build(workspace_id)
        return self._cache[workspace_id]

    def invalidate(self, workspace_id: str) -> None:
        """Sonraki get() çağrısı bu workspace'i yeniden kursun."""
        self._cache.pop(workspace_id, None)

    def cached(self, workspace_id: str) -> Optional[TurkishRAGSystem]:
        """Cache'te varsa RAG'i döndür, yoksa None — build TETİKLEMEZ.

        /status, /health gibi "hazır mı?" sorgularında kullanılır:
        bu endpoint'ler ağır bir build'i tetiklememeli.
        """
        return self._cache.get(workspace_id)
