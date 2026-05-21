"""Flask application for the Turkish RAG System.

Thin entry point. The factory wires blueprints + middleware + error
handlers and warm-starts the default workspace. The endpoints live in
src/api/routes/, shared runtime objects in src/api/runtime.py, request
schemas + the exception→HTTP mapper in src/api/, and the business logic
in src/services/ and src/rag_system.py.
"""

from __future__ import annotations

import threading

from flask import Flask
from flask_cors import CORS

from config.settings import settings
from src.api import runtime
from src.api.errors import register_error_handlers
from src.api.routes import register_blueprints
from src.observability import install_flask_middleware, install_logging
from src.utils import StatusEmoji, get_logger

# Request-id-aware logging — replaces the legacy setup_logging.
install_logging()
logger = get_logger(__name__)

_MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # 16 MB upload cap


def create_app() -> Flask:
    """Build and configure the Flask app.

    Application factory: tests can build an isolated app, production uses
    the module-level `app` created just below.
    """
    flask_app = Flask(__name__)
    flask_app.config["MAX_CONTENT_LENGTH"] = _MAX_UPLOAD_BYTES
    CORS(flask_app)
    install_flask_middleware(flask_app)   # X-Request-ID + request metrics
    register_error_handlers(flask_app)    # RagArtError → consistent JSON
    register_blueprints(flask_app)        # all /… endpoints
    return flask_app


app = create_app()


def _start_background_init() -> None:
    """Warm-start the default workspace's RAG off the request path."""
    threading.Thread(
        target=runtime.initialize_default_workspace, daemon=True
    ).start()


_start_background_init()


if __name__ == "__main__":
    print(f"{StatusEmoji.ROCKET} Starting Flask server on :{settings.PORT}")
    app.run(debug=settings.DEBUG, host=settings.HOST, port=settings.PORT)
