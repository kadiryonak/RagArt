"""File / knowledge-base routes: upload, delete, list, reindex, serve source."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Blueprint, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from src.api import runtime
from src.api.schemas import DeleteFileRequest, parse_body
from src.loaders import get_default_registry
from src.utils import StatusEmoji, get_logger

logger = get_logger(__name__)

bp = Blueprint("files", __name__)


@bp.route("/upload", methods=["POST"])
def upload_file():
    """Upload a single file into the active workspace's knowledge base."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not runtime.allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {sorted(runtime.ALLOWED_EXTENSIONS)}"
        }), 400

    filepath = None
    ws_id = runtime.current_workspace_id()
    try:
        filename = secure_filename(file.filename)
        ws_files_dir = runtime.workspace_manager.files_dir(ws_id)
        filepath = str(ws_files_dir / filename)
        file.save(filepath)

        # Format-specific validation: try to load with the matching loader.
        # If parsing fails, delete the file and report — the user shouldn't
        # end up with a corrupted file in their knowledge base.
        registry = get_default_registry()
        loader = registry.get_loader(Path(filepath))
        if loader is None:
            os.remove(filepath)
            return jsonify({"error": "No loader registered for this extension"}), 400

        try:
            docs = loader.load(Path(filepath))
        except Exception as e:
            os.remove(filepath)
            return jsonify({
                "error": f"Could not parse {filename}: {type(e).__name__}: {e}"
            }), 400

        if not docs:
            logger.warning(
                f"{StatusEmoji.WARNING} {filename} produced 0 documents "
                "(empty or unreadable)"
            )

        logger.info(
            f"{StatusEmoji.SUCCESS} File uploaded to ws={ws_id}: {filename} "
            f"(format={loader.name}, {len(docs)} documents)"
        )
        runtime.workspace_manager.touch(ws_id)

        warning = None
        if not docs:
            warning = (
                f"'{filename}' başarıyla yüklendi ama PDF/DOCX/MD/TXT'den "
                "metin çıkarılamadı (taranmış / boş içerik olabilir). "
                "Bu dosya reindex'e dahil olsa bile retrieval için boş kalır."
            )

        return jsonify({
            "success": True,
            "filename": filename,
            "format": loader.name,
            "document_count": len(docs),
            "workspace_id": ws_id,
            "warning": warning,
            "message": (
                f"File '{filename}' uploaded successfully. "
                "Use /reindex to update the knowledge base."
            ),
        })

    except Exception as e:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        logger.error(f"{StatusEmoji.ERROR} Upload error: {e}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@bp.route("/delete-file", methods=["POST"])
def delete_file():
    """Delete a file from the active workspace's knowledge base."""
    body = parse_body(DeleteFileRequest, request.get_json(silent=True))

    # Secure the filename to prevent path traversal.
    filename = secure_filename(body.filename)
    ws_id = runtime.current_workspace_id()
    filepath = str(runtime.workspace_manager.files_dir(ws_id) / filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        os.remove(filepath)
        runtime.workspace_manager.touch(ws_id)
        logger.info(f"{StatusEmoji.SUCCESS} File deleted from ws={ws_id}: {filename}")
        return jsonify({
            "success": True,
            "message": f"File '{filename}' deleted. Use /reindex to update the knowledge base.",
        })
    except Exception as e:
        logger.error(f"{StatusEmoji.ERROR} Delete error: {e}")
        return jsonify({"error": f"Delete failed: {str(e)}"}), 500


@bp.route("/reindex", methods=["POST"])
def reindex_documents():
    """Update the vector store for the active workspace.

    Body JSON (optional):
        {"full": true}  → full rebuild (re-embeds every file). Needed when
                           a file was edited in place (same name).
        {} or omitted   → incremental sync: only new files are embedded,
                           removed files are dropped. Cheap as the KB grows.
    """
    data = request.get_json(silent=True) or {}
    full = bool(data.get("full"))
    ws_id = runtime.current_workspace_id()
    try:
        runtime.system.status = f"Reindexing workspace '{ws_id}'..."
        runtime.system.ready = False

        logger.info(
            f"{StatusEmoji.LOADING} Reindexing ws={ws_id} "
            f"({'full' if full else 'incremental'})..."
        )
        rag = runtime.get_rag_for(ws_id)
        if full:
            rag.create_vector_store()
            sync = {"mode": "full", "added": [], "removed": [], "added_chunks": 0}
        else:
            sync = rag.sync_index()
        runtime.workspace_manager.touch(ws_id)

        # Invalidate response & semantic caches — knowledge base changed,
        # old answers may be stale. Embedding cache is kept (embeddings
        # are model-level, not data-level).
        rag.response_cache.clear()
        rag.semantic_cache.clear()

        runtime.system.status = "Ready"
        runtime.system.ready = True

        logger.info(f"{StatusEmoji.SUCCESS} Reindexing complete for ws={ws_id}!")

        return jsonify({
            "success": True,
            "message": "Knowledge base reindexed successfully.",
            "sync": sync,
            "document_count": rag.document_loader.document_count,
            "workspace_id": ws_id,
        })

    except Exception as e:
        runtime.system.status = f"Error: {str(e)}"
        logger.error(f"{StatusEmoji.ERROR} Reindex error: {e}")
        return jsonify({"error": f"Reindex failed: {str(e)}"}), 500


@bp.route("/list-files")
def list_files():
    """List all supported files in the active workspace."""
    try:
        ws_id = runtime.current_workspace_id()
        files_dir = runtime.workspace_manager.files_dir(ws_id)
        if not files_dir.exists():
            return jsonify({"files": [], "total": 0})

        # Delegate to the document loader's get_file_info — it already
        # walks every supported extension and is unit tested.
        from src.document_loader import JSONDocumentLoader
        loader = JSONDocumentLoader(str(files_dir))
        files = loader.get_file_info()

        return jsonify({
            "files": files,
            "total": len(files),
            "supported_extensions": sorted(runtime.ALLOWED_EXTENSIONS),
            "workspace_id": ws_id,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/source/<path:filename>")
def serve_source(filename):
    """Serve a raw source file so the UI can open the original document.

    Path-traversal safe via secure_filename: any rewrite is refused.
    """
    safe = secure_filename(filename)
    if not safe or safe != filename:
        return jsonify({"error": "Invalid filename"}), 400

    ws_id = request.args.get("ws") or runtime.current_workspace_id()
    ws_id = runtime.workspace_manager.resolve(ws_id)
    folder = str(runtime.workspace_manager.files_dir(ws_id))
    target = os.path.join(folder, safe)
    if not os.path.exists(target) or not os.path.isfile(target):
        return jsonify({"error": "Not found"}), 404

    ext = os.path.splitext(safe)[1].lower()
    mime = {
        ".pdf": "application/pdf",
        ".json": "application/json; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".markdown": "text/markdown; charset=utf-8",
        ".txt": "text/plain; charset=utf-8",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }.get(ext, "application/octet-stream")

    return send_from_directory(folder, safe, mimetype=mime)
