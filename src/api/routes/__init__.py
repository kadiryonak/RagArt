"""API route blueprints.

Each module groups one domain's endpoints. register_blueprints() wires
them all onto the Flask app — app.py's factory calls it once.
"""

from flask import Flask

from src.api.routes.cache import bp as cache_bp
from src.api.routes.chat import bp as chat_bp
from src.api.routes.files import bp as files_bp
from src.api.routes.system import bp as system_bp
from src.api.routes.workspaces import bp as workspaces_bp

_BLUEPRINTS = (system_bp, chat_bp, files_bp, workspaces_bp, cache_bp)


def register_blueprints(app: Flask) -> None:
    """Register every API blueprint on the app."""
    for bp in _BLUEPRINTS:
        app.register_blueprint(bp)


__all__ = ["register_blueprints"]
