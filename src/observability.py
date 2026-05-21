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
import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, Optional


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


# ─── Metrics ───────────────────────────────────────────────────────────


class Metrics:
    """Process-wide HTTP request metrics — thread-safe, in-memory.

    Deliberately dependency-free (no Prometheus client): a showcase repo
    should run with `pip install -r requirements.txt` and nothing else.
    /metrics serves a JSON snapshot — counts, latency percentiles, uptime.

    Latency samples are kept in a bounded deque so memory stays flat under
    long-running load; percentiles are computed over that recent window.
    """

    _MAX_SAMPLES = 2000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.time()
        self._total = 0
        self._by_status: Dict[str, int] = {}     # "2xx" → n
        self._by_endpoint: Dict[str, int] = {}   # route rule → n
        self._errors_5xx = 0
        self._latencies: deque = deque(maxlen=self._MAX_SAMPLES)

    def record(self, endpoint: str, status_code: int, latency_s: float) -> None:
        """Tek bir tamamlanmış HTTP isteğini kaydet."""
        bucket = f"{status_code // 100}xx"
        with self._lock:
            self._total += 1
            self._by_status[bucket] = self._by_status.get(bucket, 0) + 1
            if endpoint:
                self._by_endpoint[endpoint] = self._by_endpoint.get(endpoint, 0) + 1
            if status_code >= 500:
                self._errors_5xx += 1
            self._latencies.append(latency_s)

    def reset(self) -> None:
        """Sayaçları sıfırla — testler + manuel debug için."""
        with self._lock:
            self._started = time.time()
            self._total = 0
            self._by_status.clear()
            self._by_endpoint.clear()
            self._errors_5xx = 0
            self._latencies.clear()

    def snapshot(self) -> Dict[str, Any]:
        """/metrics için JSON-serializable anlık görüntü."""
        with self._lock:
            lat = sorted(self._latencies)
            total, errors = self._total, self._errors_5xx
            by_status = dict(self._by_status)
            by_endpoint = dict(self._by_endpoint)
            uptime = time.time() - self._started

        def pct(p: float) -> float:
            if not lat:
                return 0.0
            idx = min(len(lat) - 1, int(round(p / 100 * (len(lat) - 1))))
            return round(lat[idx], 4)

        return {
            "uptime_seconds": round(uptime, 1),
            "requests_total": total,
            "requests_by_status": by_status,
            "requests_by_endpoint": by_endpoint,
            "errors_5xx": errors,
            "latency_seconds": {
                "samples": len(lat),
                "avg": round(sum(lat) / len(lat), 4) if lat else 0.0,
                "p50": pct(50),
                "p95": pct(95),
                "p99": pct(99),
                "max": round(lat[-1], 4) if lat else 0.0,
            },
        }


# Process-wide singleton — install_flask_middleware feeds it, /metrics reads it.
metrics = Metrics()


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
    from flask import g, request

    @app.before_request
    def _assign_request_id():
        rid = (request.headers.get("X-Request-ID") or new_request_id()).strip()
        # Cap length to avoid huge IDs in logs
        rid = rid[:32] or new_request_id()
        set_request_id(rid)
        # Start the latency timer for the metrics collector.
        g._ragart_t0 = time.perf_counter()

    @app.after_request
    def _propagate_request_id(response):
        response.headers["X-Request-ID"] = get_request_id()
        # Record request metrics. Use the matched route rule (e.g.
        # /workspaces/<ws_id>) not the raw path, to keep cardinality low.
        t0 = g.pop("_ragart_t0", None)
        if t0 is not None:
            endpoint = request.url_rule.rule if request.url_rule else "(unmatched)"
            metrics.record(endpoint, response.status_code, time.perf_counter() - t0)
        return response
