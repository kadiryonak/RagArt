"""RagArt exception hierarchy.

Tüm domain hataları RagArtError'dan türer. Her exception:
    - http_status      : Flask handler hangi HTTP kodunu döndürmeli?
    - user_message_tr  : son kullanıcıya gösterilecek Türkçe mesaj
    - log_level        : loglanırken hangi seviyede yazılmalı?

NEDEN?
    Bu refactor öncesi kod base'de "stale collection" gibi durumlar
    `RuntimeError("STALE_INDEX")` gibi magic string sentinel'larla
    işaretleniyordu. except clausları str(e) içinde substring arıyordu.
    Bu yaklaşım:
        - Type system'i bypass eder (mypy yakalayamaz)
        - Sessizce yutulan exception riski yüksek
        - Yeni hata tipi eklemek için her except'ı dolaşmak gerekir

    Hierarchy ile:
        - except StaleIndexError as e: → tip kontrolü mevcut
        - HTTP layer tek noktada hatadan response'a map eder
        - Yeni hata sadece yeni bir class — mevcut kod değişmez

KULLANIM
    try:
        rag.retrieve(query)
    except StaleIndexError as e:
        # Friendly Turkish message hazır; HTTP kodu da hazır
        return jsonify({"error": e.user_message_tr}), e.http_status
    except RagArtError as e:
        # Catch-all aile reddi
        ...
"""

from __future__ import annotations

import logging
from typing import Optional


class RagArtError(Exception):
    """Tüm RagArt domain hatalarının temel sınıfı.

    Her alt sınıf üç şeyi tanımlar:
        - http_status      : Flask'ın döndüreceği HTTP kodu
        - user_message_tr  : UI'a gösterilecek nazik Türkçe mesaj
        - log_level        : Bu hata loglanırken hangi seviyede yazılmalı
                             (warning vs error — gürültü kontrolü)

    Constructor argument'i opsiyonel `detail` — geliştirici/log için
    serbest metin (kullanıcıya yansımaz).
    """

    http_status: int = 500
    user_message_tr: str = "Beklenmeyen bir hata oluştu."
    log_level: int = logging.ERROR

    def __init__(self, detail: Optional[str] = None):
        super().__init__(detail or self.user_message_tr)
        self.detail = detail or ""

    def to_response(self) -> dict:
        """HTTP response için JSON-serializable dict."""
        return {
            "error": self.user_message_tr,
            "detail": self.detail,
            "error_type": type(self).__name__,
        }


# ─── Workspace ─────────────────────────────────────────────────────────


class WorkspaceError(RagArtError):
    """Workspace tabanlı hatalar."""


class WorkspaceNotFoundError(WorkspaceError):
    http_status = 404
    user_message_tr = "Çalışma alanı bulunamadı."
    log_level = logging.WARNING


class DefaultWorkspaceProtectedError(WorkspaceError):
    http_status = 400
    user_message_tr = "Varsayılan çalışma alanı silinemez."
    log_level = logging.WARNING


# ─── Retrieval / Vector Store ──────────────────────────────────────────


class RetrievalError(RagArtError):
    """Retrieval / vector store hatalarının kök sınıfı."""


class StaleIndexError(RetrievalError):
    """ChromaDB koleksiyonu silinmiş ama bellek referansı eski (reindex
    sonrası rerank cache'in stale kaldığı senaryo)."""
    http_status = 503
    user_message_tr = (
        "Vektör tabanı eski koleksiyona referans veriyor. "
        "Lütfen 'Bilgi Tabanını Yeniden İndeksle' butonuna basın."
    )
    log_level = logging.WARNING


class EmptyKnowledgeBaseError(RetrievalError):
    """Workspace'te hiç dosya yok veya hepsinden 0 chunk çıkarılmış."""
    http_status = 400
    user_message_tr = (
        "Bilgi tabanı boş. Önce 'Dosyaları Yönet' sekmesinden bir dosya "
        "yükleyip ardından yeniden indeksleyin."
    )
    log_level = logging.INFO


# ─── LLM Provider ──────────────────────────────────────────────────────


class LLMError(RagArtError):
    """LLM provider'dan dönen hataların kök sınıfı."""


class LLMRateLimitError(LLMError):
    http_status = 429
    user_message_tr = (
        "LLM sağlayıcı geçici olarak rate limit uyguluyor; "
        "biraz sonra tekrar deneyin."
    )
    log_level = logging.WARNING


class LLMAuthError(LLMError):
    http_status = 401
    user_message_tr = (
        "LLM API anahtarı geçersiz veya eksik. "
        "Ayarlar sekmesinden anahtarı kontrol edin."
    )
    log_level = logging.WARNING


# ─── Cache ─────────────────────────────────────────────────────────────


class CacheError(RagArtError):
    """Cache layer hatalarının kök sınıfı (bozuk pickle, disk dolu vb.)."""
    log_level = logging.WARNING


# ─── Guard / Security ──────────────────────────────────────────────────


class GuardBlockedError(RagArtError):
    """Input guard (prompt injection detector) sorguyu reddetti."""
    http_status = 400
    user_message_tr = "Soru güvenlik kontrolünden geçemedi."
    log_level = logging.WARNING


class PathTraversalError(RagArtError):
    """secure_filename'in atlayabileceği bir traversal denemesi yakalandı."""
    http_status = 400
    user_message_tr = "Dosya yolu geçersiz."
    log_level = logging.WARNING


# ─── Validation ────────────────────────────────────────────────────────


class ValidationError(RagArtError):
    """Request parametresi şema kontrolünden geçemedi."""
    http_status = 400
    user_message_tr = "İstek geçersiz parametre içeriyor."
    log_level = logging.INFO


# ─── Configuration ─────────────────────────────────────────────────────


class ConfigError(RagArtError):
    """Sunucu-tarafı konfig hatası (env var eksik, dosya bulunamadı vb.)."""
    http_status = 500
    user_message_tr = "Sunucu yapılandırma hatası."
