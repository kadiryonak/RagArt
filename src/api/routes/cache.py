"""Cache routes: hit/miss stats + clearing one or all cache layers."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from src.api import runtime
from src.api.schemas import CacheClearRequest, parse_body
from src.utils import StatusEmoji, get_logger

logger = get_logger(__name__)

bp = Blueprint("cache", __name__)


@bp.route("/cache/stats")
def cache_stats():
    """Hit/miss statistics for all three cache layers."""
    if not runtime.system.ready:
        return jsonify({"error": "System not ready"}), 503

    try:
        rag = runtime.get_rag_for(runtime.current_workspace_id())
        return jsonify({
            "success": True,
            "caches": {
                "embedding": rag.embedding_cache.stats(),
                "response": rag.response_cache.stats(),
                "semantic": rag.semantic_cache.stats(),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/cache/clear", methods=["POST"])
def cache_clear():
    """Clear one or all cache layers.

    Body JSON (optional): {"layer": "response" | "semantic" | "embedding"}
    — omitted clears all.
    """
    if not runtime.system.ready:
        return jsonify({"error": "System not ready"}), 503

    # layer is validated by the schema (Literal) — an unknown layer is
    # rejected as a 400 before any cache work happens.
    body = parse_body(CacheClearRequest, request.get_json(silent=True))
    layer = body.layer

    try:
        rag = runtime.get_rag_for(runtime.current_workspace_id())

        cleared = {}
        if layer in ("all", "embedding"):
            cleared["embedding"] = rag.embedding_cache.clear()
        if layer in ("all", "response"):
            cleared["response"] = rag.response_cache.clear()
        if layer in ("all", "semantic"):
            cleared["semantic"] = rag.semantic_cache.clear()

        logger.info(f"{StatusEmoji.SUCCESS} Cache cleared: {cleared}")
        return jsonify({"success": True, "cleared": cleared})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
