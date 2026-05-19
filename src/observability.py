"""Observability — request ID + structured logging hooks.

ANA FİKİR
    Her HTTP isteğine 8-karakter request_id atanır. Bu ID:
        - Tüm log mesajlarına otomatik eklenir
        - Response'da X-Request-ID header'ı olarak döner
        - Tail'da grep'lenebilir → "kullanıcı şu sorguda hata aldı" durumunda
          tek satırlık aramayla tüm pipeline log'u görülebilir

KULLANIM
    # Flask startup'ında:
    from src.observability import install_logging, install_flask_middleware
    install_logging()
    install_flask_middleware(app)

    # Daha sonra her logger.info(...) çağrısı otomatik olarak
    # `[req-XXXX] timestamp ... mesaj` formatında çıktı verir.

TASARIM KARARI
    contextvars.ContextVar kullanıyoruz — Flask'ın `g` objesi yerine.
    Sebep: ContextVar thread-safe + async-safe + Flask dışında da çalışır
    (pipeline stage'lerinden, background thread'lerden, vb.).
"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from typing import Any, Optional


# Tüm modüller bu context var üzerinden okur/yazar.
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "ragart_request_id", default="-"
)


def get_request_id() -> str:
    """Aktif request'in ID'sini döndür. Request dışında '-' döner."""
    return _request_id_var.get()


def set_request_id(rid: str) -> None:
    """Yeni request başında çağrılır."""
    _request_id_var.set(rid)


def new_request_id() -> str:
    """Kısa, log-friendly ID üret (uuid4 ilk 8 karakteri)."""
    return uuid.uuid4().hex[:8]


# ─── Logging filter ───────────────────────────────────────────────────


class _RequestIDFilter(logging.Filter):
    """Her LogRecord'a request_id field'ı ekler."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


_DEFAULT_FORMAT = (
    "%(asctime)s [%(request_id)s] %(levelname)-7s "
    "%(name)s — %(message)s"
)


def install_logging(
    level: int = logging.INFO,
    *,
    format_string: Optional[str] = None,
    quiet_loggers: Optional[list] = None,
) -> None:
    """Root logger'a request_id filter + standart formatter ekle.

    Mevcut handler'ları temizler — uvicorn/Flask development server'ın
    iki kez log basmasını engeller.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Drop existing handlers — we own logging from this point.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_string or _DEFAULT_FORMAT))
    handler.addFilter(_RequestIDFilter())
    root.addHandler(handler)

    # Quiet some noisy third-parties (chromadb, httpx) unless DEBUG.
    for noisy in (quiet_loggers or [
        "chromadb", "httpx", "httpcore", "urllib3",
        "sentence_transformers", "transformers",
    ]):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


# ─── Flask middleware ──────────────────────────────────────────────────


def install_flask_middleware(app: Any) -> None:
    """Flask app'e before_request + after_request hook'ları ekle.

    İncoming request'te:
        - X-Request-ID header'ı varsa onu kullan, yoksa yeni üret
        - ContextVar'a set et — tüm pipeline boyunca erişilebilir
    Outgoing response'ta:
        - X-Request-ID response header'ı olarak yansıt — istemci debug'da
          aynı ID ile arar

    Flask import'unu modül seviyesinde yapmamak için runtime'da import
    ediyoruz (observability modülü Flask'sız da kullanılabilsin diye).
    """
    from flask import request

    @app.before_request
    def _assign_request_id():
        rid = (request.headers.get("X-Request-ID") or new_request_id()).strip()
        # Cap length to avoid huge IDs in logs
        rid = rid[:32] or new_request_id()
        set_request_id(rid)

    @app.after_request
    def _propagate_request_id(response):
        response.headers["X-Request-ID"] = get_request_id()
        return response
