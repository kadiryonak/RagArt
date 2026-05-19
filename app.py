"""
Flask web application for the Turkish RAG System.

This module provides a REST API and web interface for interacting
with the RAG system.
"""

import os
import json
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.rag_system import TurkishRAGSystem
from src.document_loader import create_sample_data
from src.llm_providers import LLMProviderFactory
from src.memory import ConversationTurn
from src.utils import get_logger, StatusEmoji, setup_logging
from src.workspaces import WorkspaceManager, DEFAULT_WORKSPACE_ID
from src.vector_stores import VectorStoreFactory
from config.settings import settings
from config.settings_schema import (
    get_settings_schema,
    parse_request_settings,
)

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# File upload configuration — multi-format
# Driven by the LoaderRegistry so adding a new loader auto-enables uploads.
from src.loaders import get_default_registry as _get_registry
ALLOWED_EXTENSIONS = {ext.lstrip(".") for ext in _get_registry().supported_extensions}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ----- Global state — workspaces + lazy RAG cache -----
#
# Workspaces are NotebookLM-style isolated knowledge bases. Each workspace
# has its own data folder and ChromaDB collection. We keep one RAG
# instance per workspace (lazily created on first use) so the embedder
# model can be shared while the vector store is separate.

workspace_manager = WorkspaceManager(settings.DATA_FOLDER)
_rag_cache: dict = {}
_rag_init_lock = threading.Lock()

# Sample data seeding: ensure the default workspace has SOMETHING
_default_files_dir = workspace_manager.files_dir(DEFAULT_WORKSPACE_ID)
if not any(_default_files_dir.iterdir()):
    create_sample_data(str(_default_files_dir))

system_ready = False
system_status = "Initializing..."
initialization_error = None


def _build_rag_for_workspace(workspace_id: str) -> TurkishRAGSystem:
    """Build (or retrieve) a fresh RAG instance scoped to a workspace."""
    ws = workspace_manager.get(workspace_id)
    if ws is None:
        workspace_id = workspace_manager.resolve(workspace_id)
        ws = workspace_manager.get(workspace_id)

    api_key = settings.get_api_key()
    model_type = settings.MODEL_TYPE
    if model_type in ("deepseek", "openai", "groq", "huggingface") and not api_key:
        logger.warning(
            f"{StatusEmoji.WARNING} No API key found, falling back to local model"
        )
        model_type = "local"

    persist_path = workspace_manager.vector_db_path(workspace_id, ws.vector_db)
    rag = TurkishRAGSystem(
        data_folder=str(workspace_manager.files_dir(workspace_id)),
        model_type=model_type,
        api_key=api_key,
        chroma_db_path=str(persist_path),
    )
    rag.initialize()
    return rag


def get_rag_for(workspace_id: str) -> TurkishRAGSystem:
    """Return the cached RAG for a workspace, building lazily under lock."""
    workspace_id = workspace_manager.resolve(workspace_id)
    if workspace_id in _rag_cache:
        return _rag_cache[workspace_id]
    with _rag_init_lock:
        if workspace_id not in _rag_cache:
            _rag_cache[workspace_id] = _build_rag_for_workspace(workspace_id)
    return _rag_cache[workspace_id]


def invalidate_rag(workspace_id: str) -> None:
    """Force the next get_rag_for() call to rebuild the cache entry."""
    _rag_cache.pop(workspace_id, None)


def _current_workspace_id() -> str:
    """Pull the active workspace id from the request header (or default)."""
    return workspace_manager.resolve(request.headers.get("X-Workspace-Id"))


def initialize_default_workspace() -> None:
    """Warm-start the default workspace's RAG in the background."""
    global system_ready, system_status, initialization_error
    try:
        system_status = "Checking data files..."
        logger.info(f"{StatusEmoji.ROCKET} Initializing default workspace RAG...")
        get_rag_for(DEFAULT_WORKSPACE_ID)
        system_status = "Ready"
        system_ready = True
        logger.info(f"{StatusEmoji.SUCCESS} Default workspace ready")
    except Exception as e:
        initialization_error = str(e)
        system_status = f"Error: {str(e)}"
        system_ready = False
        logger.error(f"{StatusEmoji.ERROR} Initialization error: {e}")


init_thread = threading.Thread(target=initialize_default_workspace, daemon=True)
init_thread.start()


@app.route("/")
def index():
    """Serve the web interface."""
    templates_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(templates_dir, "templates/index.html")


@app.route("/status")
def get_status():
    """Get system status (for the current workspace if header is set)."""
    ws_id = workspace_manager.resolve(request.headers.get("X-Workspace-Id"))
    rag = _rag_cache.get(ws_id)
    return jsonify({
        "ready": system_ready and (rag is not None or ws_id == DEFAULT_WORKSPACE_ID),
        "status": system_status,
        "error": initialization_error,
        "model_type": rag.model_type if rag else "unknown",
        "workspace_id": ws_id,
    })


@app.route("/ask", methods=["POST"])
def ask_question():
    """Handle question answering requests.

    BYOK: clients may pass X-Provider / X-API-Key / X-Model / X-LLM-Params
    headers. When set, those override the server-side defaults for this
    single request — the server never persists the key.
    """
    if not system_ready:
        return jsonify({
            "error": "System not ready",
            "status": system_status
        }), 503

    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        # Parse per-request settings from headers (BYOK)
        req_settings = parse_request_settings(request.headers)
        llm_override = None
        if req_settings.provider:
            errors = req_settings.llm_params.validate(req_settings.provider)
            if errors:
                return jsonify({
                    "error": "Invalid LLM params",
                    "details": errors,
                }), 400
            try:
                llm_override = LLMProviderFactory.create(
                    req_settings.provider,
                    api_key=req_settings.api_key,
                    model=req_settings.model,
                )
            except ValueError as e:
                return jsonify({"error": str(e)}), 400

        logger.info(f"{StatusEmoji.SEARCH} Question received: {question}")

        history = [ConversationTurn.from_dict(t) for t in req_settings.history]

        # Resolve the workspace this request targets — lazy-create RAG.
        ws_id = _current_workspace_id()
        try:
            rag = get_rag_for(ws_id)
        except Exception as e:
            return jsonify({"error": f"Workspace init failed: {e}"}), 500

        result = rag.ask(
            question,
            k=settings.TOP_K_DOCUMENTS,
            llm_provider=llm_override,
            llm_params=req_settings.llm_params.to_dict(),
            retrieval_strategy=req_settings.retrieval_strategy,
            rerank=req_settings.rerank,
            rerank_fetch_k=req_settings.rerank_fetch_k,
            history=history,
            memory_strategy=req_settings.memory_strategy,
            deduplicate_context=req_settings.deduplicate_context,
            reorder_context=req_settings.reorder_context,
            max_context_tokens=req_settings.max_context_tokens,
            allow_general_knowledge_fallback=req_settings.allow_general_knowledge_fallback,
        )
        
        # Check for errors
        if result.get("source") == "error":
            return jsonify({
                "error": result.get("answer", "Unknown error"),
                "question": question
            }), 500
        
        # Return successful response
        response_data = {
            "success": True,
            "question": result["question"],
            "answer": result["answer"],
            "sources": [
                {
                    "title": doc["source"],
                    "content": doc["content"],
                    "metadata": doc["metadata"]
                }
                for doc in result["source_documents"]
            ],
            "context_used": result.get("context_used", ""),
            "source_type": result.get("source", "unknown"),
            "relevance_score": result.get("relevance_score", 0.0)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"{StatusEmoji.ERROR} Question handling error: {e}")
        return jsonify({
            "error": f"Error processing question: {str(e)}"
        }), 500


@app.route("/test")
def test_system():
    """Run system tests against the active workspace."""
    if not system_ready:
        return jsonify({"error": "System not ready"}), 503

    rag = get_rag_for(_current_workspace_id())
    test_questions = [
        "Algoritma nedir?",
        "Python hakkında ne biliyorsun?",
        "Yapay zeka ile ilgili bilgiler neler?",
        "Veri yapıları nedir?",
    ]

    results = []
    for question in test_questions:
        try:
            result = rag.ask(question)
            results.append({
                "question": question,
                "answer": result["answer"][:200] + "..." if len(result["answer"]) > 200 else result["answer"],
                "sources_count": len(result["source_documents"]),
                "source_type": result.get("source", "unknown"),
                "relevance_score": result.get("relevance_score", 0.0)
            })
        except Exception as e:
            results.append({
                "question": question,
                "error": str(e)
            })
    
    return jsonify({"test_results": results})


@app.route("/data-info")
def get_data_info():
    """Get information about loaded data in the active workspace."""
    if not system_ready:
        return jsonify({"error": "System not ready"}), 503

    try:
        rag = get_rag_for(_current_workspace_id())
        file_info = rag.document_loader.get_file_info()

        total_documents = sum(
            f.get("document_count", 0) for f in file_info
            if "error" not in f
        )

        return jsonify({
            "files": file_info,
            "total_files": len(file_info),
            "total_documents": total_documents,
            "model_type": rag.model_type,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/source/<path:filename>")
def serve_source(filename):
    """Serve a raw source file from the data folder.

    Used by the UI so that clicking a source-document chip opens the
    original file (PDFs with #page=N let the browser jump to the right
    page). Path-traversal safe via secure_filename.
    """
    safe = secure_filename(filename)
    if not safe or safe != filename:
        # secure_filename strips or rewrites — refuse the rewritten form
        # to make traversal attempts loud rather than silent
        return jsonify({"error": "Invalid filename"}), 400

    folder = os.path.abspath(settings.DATA_FOLDER)
    target = os.path.join(folder, safe)
    if not os.path.exists(target) or not os.path.isfile(target):
        return jsonify({"error": "Not found"}), 404

    ext = os.path.splitext(safe)[1].lower()
    mime = {
        ".pdf":     "application/pdf",
        ".json":    "application/json; charset=utf-8",
        ".md":      "text/markdown; charset=utf-8",
        ".markdown": "text/markdown; charset=utf-8",
        ".txt":     "text/plain; charset=utf-8",
        ".docx":    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }.get(ext, "application/octet-stream")

    return send_from_directory(folder, safe, mimetype=mime)


@app.route("/settings/schema")
def settings_schema():
    """Return the settings schema used by the frontend to build the UI.

    No secrets here — pure metadata (provider list, param specs, defaults,
    human-readable descriptions). The frontend reads this once on load to
    render dropdowns and sliders.
    """
    return jsonify(get_settings_schema())


@app.route("/health")
def health_check():
    """Health check endpoint."""
    ws_id = _current_workspace_id()
    rag = _rag_cache.get(ws_id)
    return jsonify({
        "status": "healthy",
        "system_ready": system_ready,
        "model_type": rag.model_type if rag else "unknown",
        "api_available": bool(rag and rag.api_key) if rag else False,
        "active_workspace": ws_id,
    })


@app.route("/stats")
def get_stats():
    """Get system statistics for the active workspace."""
    if not system_ready:
        return jsonify({"error": "System not ready"}), 503

    rag = get_rag_for(_current_workspace_id())
    return jsonify(rag.get_stats())


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload for knowledge base."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        }), 400
    
    filepath = None
    ws_id = _current_workspace_id()
    try:
        filename = secure_filename(file.filename)
        ws_files_dir = workspace_manager.files_dir(ws_id)
        filepath = str(ws_files_dir / filename)
        file.save(filepath)

        # Format-specific validation: try to load with the matching loader.
        # If parsing fails, delete the file and report the error — the user
        # shouldn't end up with a corrupted file in their knowledge base.
        from pathlib import Path
        from src.loaders import get_default_registry
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
            # Empty extract — still keep the file but warn (e.g. scanned PDF)
            logger.warning(
                f"{StatusEmoji.WARNING} {filename} produced 0 documents "
                "(empty or unreadable)"
            )

        logger.info(
            f"{StatusEmoji.SUCCESS} File uploaded to ws={ws_id}: {filename} "
            f"(format={loader.name}, {len(docs)} documents)"
        )
        workspace_manager.touch(ws_id)

        warning = None
        if not docs:
            warning = (
                f"'{filename}' başarıyla yüklendi ama PDF/DOCX/MD/TXT'den "
                "metin çıkarılamadı (taranmış / boş içerik olabilir). "
                "Bu dosya reindex'e dahil olsa bile retrieval için boş "
                "kalır."
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


@app.route("/delete-file", methods=["POST"])
def delete_file():
    """Delete a file from the knowledge base."""
    data = request.get_json()
    filename = data.get("filename", "")
    
    if not filename:
        return jsonify({"error": "No filename provided"}), 400
    
    # Secure the filename to prevent path traversal
    filename = secure_filename(filename)
    ws_id = _current_workspace_id()
    filepath = str(workspace_manager.files_dir(ws_id) / filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404

    try:
        os.remove(filepath)
        workspace_manager.touch(ws_id)
        logger.info(f"{StatusEmoji.SUCCESS} File deleted from ws={ws_id}: {filename}")
        
        return jsonify({
            "success": True,
            "message": f"File '{filename}' deleted. Use /reindex to update the knowledge base."
        })
        
    except Exception as e:
        logger.error(f"{StatusEmoji.ERROR} Delete error: {e}")
        return jsonify({"error": f"Delete failed: {str(e)}"}), 500


@app.route("/reindex", methods=["POST"])
def reindex_documents():
    """Rebuild the vector store for the active workspace only."""
    global system_ready, system_status

    ws_id = _current_workspace_id()
    try:
        system_status = f"Reindexing workspace '{ws_id}'..."
        system_ready = False

        logger.info(f"{StatusEmoji.LOADING} Reindexing ws={ws_id}...")
        rag = get_rag_for(ws_id)
        rag.create_vector_store()
        workspace_manager.touch(ws_id)

        system_status = "Ready"
        system_ready = True

        logger.info(f"{StatusEmoji.SUCCESS} Reindexing complete for ws={ws_id}!")

        return jsonify({
            "success": True,
            "message": "Knowledge base reindexed successfully.",
            "document_count": rag.document_loader.document_count,
            "workspace_id": ws_id,
        })

    except Exception as e:
        system_status = f"Error: {str(e)}"
        logger.error(f"{StatusEmoji.ERROR} Reindex error: {e}")
        return jsonify({"error": f"Reindex failed: {str(e)}"}), 500


@app.route("/list-files")
def list_files():
    """List all supported files in the active workspace."""
    try:
        ws_id = _current_workspace_id()
        files_dir = workspace_manager.files_dir(ws_id)
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
            "supported_extensions": sorted(ALLOWED_EXTENSIONS),
            "workspace_id": ws_id,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Workspace endpoints ----------

@app.route("/workspaces", methods=["GET"])
def list_workspaces():
    """List all workspaces for the landing page."""
    return jsonify({
        "workspaces": [ws.to_dict() for ws in workspace_manager.list()],
        "default_id": DEFAULT_WORKSPACE_ID,
        "vector_stores": VectorStoreFactory.available(),
    })


@app.route("/workspaces", methods=["POST"])
def create_workspace():
    """Create a new workspace from JSON body: {name, color?, description?, vector_db?}."""
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Workspace name required"}), 400

    vector_db = data.get("vector_db", "chroma")
    if not VectorStoreFactory.is_available(vector_db):
        return jsonify({
            "error": f"Unknown vector_db '{vector_db}'",
            "available": [s["id"] for s in VectorStoreFactory.available()],
        }), 400

    try:
        ws = workspace_manager.create(
            name=name,
            color=data.get("color"),
            description=data.get("description", ""),
            vector_db=vector_db,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"success": True, "workspace": ws.to_dict()}), 201


@app.route("/workspaces/<ws_id>", methods=["DELETE"])
def delete_workspace(ws_id):
    """Delete a workspace (and its files). Default workspace is protected."""
    try:
        ok = workspace_manager.delete(ws_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not ok:
        return jsonify({"error": "Workspace not found"}), 404
    invalidate_rag(ws_id)
    return jsonify({"success": True, "workspace_id": ws_id})


@app.route("/workspaces/<ws_id>", methods=["PATCH"])
def update_workspace(ws_id):
    """Rename / recolor / redescribe a workspace."""
    data = request.get_json(silent=True) or {}
    ws = workspace_manager.update(
        ws_id,
        name=data.get("name"),
        color=data.get("color"),
        description=data.get("description"),
    )
    if ws is None:
        return jsonify({"error": "Workspace not found"}), 404
    return jsonify({"success": True, "workspace": ws.to_dict()})


def create_app():
    """Application factory for testing."""
    return app


if __name__ == "__main__":
    print(f"{StatusEmoji.ROCKET} Starting Flask server...")
    print(f"Web interface: http://localhost:{settings.PORT}")
    print(f"\nAPI Endpoints:")
    print(f"  - GET  /status     : System status")
    print(f"  - POST /ask        : Ask a question")
    print(f"  - GET  /test       : Run test questions")
    print(f"  - GET  /data-info  : Data information")
    print(f"  - GET  /health     : Health check")
    print(f"  - GET  /stats      : System statistics")
    
    app.run(
        debug=settings.DEBUG,
        host=settings.HOST,
        port=settings.PORT
    )
