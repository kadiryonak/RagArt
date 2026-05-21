"""Workspace CRUD routes: list / create / delete / update."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from src.api import runtime
from src.api.schemas import CreateWorkspaceRequest, UpdateWorkspaceRequest, parse_body
from src.vector_stores import VectorStoreFactory
from src.workspaces import DEFAULT_WORKSPACE_ID

bp = Blueprint("workspaces", __name__)


@bp.route("/workspaces", methods=["GET"])
def list_workspaces():
    """List all workspaces for the landing page."""
    return jsonify({
        "workspaces": [ws.to_dict() for ws in runtime.workspace_manager.list()],
        "default_id": DEFAULT_WORKSPACE_ID,
        "vector_stores": VectorStoreFactory.available(),
    })


@bp.route("/workspaces", methods=["POST"])
def create_workspace():
    """Create a workspace from JSON: {name, color?, description?, vector_db?}."""
    body = parse_body(CreateWorkspaceRequest, request.get_json(silent=True))

    if not VectorStoreFactory.is_available(body.vector_db):
        return jsonify({
            "error": f"Unknown vector_db '{body.vector_db}'",
            "available": [s["id"] for s in VectorStoreFactory.available()],
        }), 400

    try:
        ws = runtime.workspace_manager.create(
            name=body.name,
            color=body.color,
            description=body.description,
            vector_db=body.vector_db,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"success": True, "workspace": ws.to_dict()}), 201


@bp.route("/workspaces/<ws_id>", methods=["DELETE"])
def delete_workspace(ws_id):
    """Delete a workspace (and its files). The default workspace is protected."""
    try:
        ok = runtime.workspace_manager.delete(ws_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not ok:
        return jsonify({"error": "Workspace not found"}), 404
    runtime.invalidate_rag(ws_id)
    return jsonify({"success": True, "workspace_id": ws_id})


@bp.route("/workspaces/<ws_id>", methods=["PATCH"])
def update_workspace(ws_id):
    """Rename / recolor / redescribe / switch vector DB for a workspace.

    Changing vector_db invalidates the cached RAG so the next request
    rebuilds against the new DB. The user still needs to reindex — the
    old DB's vectors don't migrate.
    """
    body = parse_body(UpdateWorkspaceRequest, request.get_json(silent=True))

    new_vector_db = body.vector_db
    if new_vector_db is not None and not VectorStoreFactory.is_available(new_vector_db):
        return jsonify({
            "error": f"Unknown vector_db '{new_vector_db}'",
            "available": [s["id"] for s in VectorStoreFactory.available()],
        }), 400

    ws = runtime.workspace_manager.update(
        ws_id,
        name=body.name,
        color=body.color,
        description=body.description,
        vector_db=new_vector_db,
    )
    if ws is None:
        return jsonify({"error": "Workspace not found"}), 404

    # Drop the cached RAG so the next request picks up the new vector DB.
    if new_vector_db is not None:
        runtime.invalidate_rag(ws_id)

    return jsonify({
        "success": True,
        "workspace": ws.to_dict(),
        "needs_reindex": new_vector_db is not None,
    })
