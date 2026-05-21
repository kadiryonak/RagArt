"""System / introspection routes: index, status, health, metrics, stats,
settings schema, data-info, self-test."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from config.settings_schema import get_settings_schema
from src.api import runtime
from src.observability import metrics
from src.workspaces import DEFAULT_WORKSPACE_ID

bp = Blueprint("system", __name__)


@bp.route("/")
def index():
    """Serve the single-page web interface."""
    # root_path is the directory of app.py (project root) — resolve the
    # template relative to it so this works regardless of CWD.
    return send_from_directory(current_app.root_path, "templates/index.html")


@bp.route("/status")
def get_status():
    """System status for the current workspace (if the header is set)."""
    ws_id = runtime.current_workspace_id()
    rag = runtime.rag_registry.cached(ws_id)
    return jsonify({
        "ready": runtime.system.ready and (
            rag is not None or ws_id == DEFAULT_WORKSPACE_ID
        ),
        "status": runtime.system.status,
        "error": runtime.system.error,
        "model_type": rag.model_type if rag else "unknown",
        "workspace_id": ws_id,
    })


@bp.route("/health")
def health_check():
    """Liveness + readiness probe with a bit of runtime context."""
    ws_id = runtime.current_workspace_id()
    rag = runtime.rag_registry.cached(ws_id)
    snap = metrics.snapshot()
    return jsonify({
        "status": "healthy",
        "system_ready": runtime.system.ready,
        "system_status": runtime.system.status,
        "model_type": rag.model_type if rag else "unknown",
        "api_available": bool(rag and rag.api_key) if rag else False,
        "active_workspace": ws_id,
        "uptime_seconds": snap["uptime_seconds"],
        "requests_total": snap["requests_total"],
    })


@bp.route("/metrics")
def metrics_endpoint():
    """Process metrics: request counts, latency percentiles, errors, uptime.

    Plain JSON (no Prometheus dependency) — keeps the showcase repo to a
    single `pip install`. Point a dashboard at it or just curl it.
    """
    return jsonify(metrics.snapshot())


@bp.route("/settings/schema")
def settings_schema():
    """Settings schema the frontend reads to build its dropdowns/sliders.

    No secrets — pure metadata (providers, param specs, defaults).
    """
    return jsonify(get_settings_schema())


@bp.route("/stats")
def get_stats():
    """System statistics for the active workspace."""
    if not runtime.system.ready:
        return jsonify({"error": "System not ready"}), 503

    rag = runtime.get_rag_for(runtime.current_workspace_id())
    return jsonify(rag.get_stats())


@bp.route("/data-info")
def get_data_info():
    """Information about the loaded data in the active workspace."""
    if not runtime.system.ready:
        return jsonify({"error": "System not ready"}), 503

    try:
        rag = runtime.get_rag_for(runtime.current_workspace_id())
        file_info = rag.document_loader.get_file_info()
        total_documents = sum(
            f.get("document_count", 0) for f in file_info if "error" not in f
        )
        return jsonify({
            "files": file_info,
            "total_files": len(file_info),
            "total_documents": total_documents,
            "model_type": rag.model_type,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/test")
def test_system():
    """Run a handful of canned questions against the active workspace."""
    if not runtime.system.ready:
        return jsonify({"error": "System not ready"}), 503

    rag = runtime.get_rag_for(runtime.current_workspace_id())
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
            answer = result["answer"]
            results.append({
                "question": question,
                "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                "sources_count": len(result["source_documents"]),
                "source_type": result.get("source", "unknown"),
                "relevance_score": result.get("relevance_score", 0.0),
            })
        except Exception as e:
            results.append({"question": question, "error": str(e)})

    return jsonify({"test_results": results})
