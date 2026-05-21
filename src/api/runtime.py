"""Process-wide runtime objects shared by every API blueprint.

app.py used to keep these as module globals, but route handlers now live
in separate blueprint modules under src/api/routes/ and can't share
those. They are created once here and used as `runtime.X` everywhere.

IMPORTANT (tests): access these as attributes of the `runtime` module
(`runtime.workspace_manager`, `runtime.system.ready`) — never
`from runtime import workspace_manager`, which would freeze the binding
and make monkeypatching invisible to the blueprints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from flask import request

from config.settings import settings
from src.document_loader import create_sample_data
from src.loaders import get_default_registry
from src.rag_system import TurkishRAGSystem
from src.services import RagRegistry
from src.utils import StatusEmoji, get_logger
from src.workspaces import DEFAULT_WORKSPACE_ID, WorkspaceManager

logger = get_logger(__name__)


@dataclass
class SystemState:
    """Mutable boot/health state — flipped by the init thread and /reindex.

    A single shared object (not module globals) so every blueprint sees
    the same mutations without needing `global` declarations.
    """

    ready: bool = False
    status: str = "Initializing..."
    error: Optional[str] = None


# ── Singletons (created once at import) ────────────────────────────────
workspace_manager = WorkspaceManager(settings.DATA_FOLDER)
rag_registry = RagRegistry(workspace_manager)
system = SystemState()

# Multi-format upload allow-list, driven by the loader registry — adding
# a loader auto-enables its extension.
ALLOWED_EXTENSIONS = {
    ext.lstrip(".") for ext in get_default_registry().supported_extensions
}

# Seed the default workspace so a fresh clone has something to query.
_default_files_dir = workspace_manager.files_dir(DEFAULT_WORKSPACE_ID)
if not any(_default_files_dir.iterdir()):
    create_sample_data(str(_default_files_dir))


def allowed_file(filename: str) -> bool:
    """True if the file extension has a registered loader."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_rag_for(workspace_id: str) -> TurkishRAGSystem:
    """Cached RAG for a workspace (lazy build under lock)."""
    return rag_registry.get(workspace_id)


def invalidate_rag(workspace_id: str) -> None:
    """Force the next get_rag_for() to rebuild the cache entry."""
    rag_registry.invalidate(workspace_id)


def current_workspace_id() -> str:
    """Active workspace id from the X-Workspace-Id header (or default)."""
    return workspace_manager.resolve(request.headers.get("X-Workspace-Id"))


def initialize_default_workspace() -> None:
    """Warm-start the default workspace's RAG (run in a background thread)."""
    try:
        system.status = "Checking data files..."
        logger.info(f"{StatusEmoji.ROCKET} Initializing default workspace RAG...")
        get_rag_for(DEFAULT_WORKSPACE_ID)
        system.status = "Ready"
        system.ready = True
        logger.info(f"{StatusEmoji.SUCCESS} Default workspace ready")
    except Exception as e:
        system.error = str(e)
        system.status = f"Error: {e}"
        system.ready = False
        logger.error(f"{StatusEmoji.ERROR} Initialization error: {e}")
