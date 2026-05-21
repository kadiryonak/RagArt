"""Merkezi exception → HTTP response map'i.

Faz A'da RagArtError hiyerarşisi kuruldu (her sınıf http_status /
user_message_tr / log_level taşır) ama HTTP layer hâlâ her endpoint'te
elle `return jsonify({"error": ...}), code` yapıyordu.

register_error_handlers(app) bunu tek noktaya toplar:

    raise WorkspaceNotFoundError("ws=foo")             # route içinde
    → 404 {"error": "Çalışma alanı bulunamadı.", ...}  # otomatik

Böylece route handler "happy path"e odaklanır; hata gösterimi + loglama
tek yerde, tutarlı.
"""

from __future__ import annotations

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from src.exceptions import RagArtError
from src.utils import get_logger

logger = get_logger(__name__)


def register_error_handlers(app: Flask) -> None:
    """RagArtError + yaygın HTTP hatalarını JSON response'a bağla."""

    @app.errorhandler(RagArtError)
    def _handle_ragart(e: RagArtError):
        # log_level exception'ın kendi sınıfından gelir — gürültü kontrolü:
        # ValidationError INFO, StaleIndexError WARNING, ConfigError ERROR.
        logger.log(e.log_level, "%s: %s", type(e).__name__, e.detail or e)
        return jsonify(e.to_response()), e.http_status

    @app.errorhandler(413)
    def _handle_payload_too_large(_e):
        # MAX_CONTENT_LENGTH aşıldığında werkzeug 413 üretir.
        return jsonify({
            "error": "Dosya çok büyük (en fazla 16 MB).",
            "error_type": "RequestEntityTooLarge",
        }), 413

    @app.errorhandler(HTTPException)
    def _handle_http_exception(e: HTTPException):
        # werkzeug'un kendi 400/404/405'leri — HTML hata sayfası yerine
        # API ile tutarlı JSON döndür.
        return jsonify({
            "error": e.description or e.name,
            "error_type": e.name,
        }), e.code or 500
