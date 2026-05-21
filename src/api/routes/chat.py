"""Chat route: POST /ask — the question-answering endpoint."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from config.settings import settings
from config.settings_schema import parse_request_settings
from src.api import runtime
from src.api.schemas import AskRequest, parse_body
from src.llm_providers import LLMProviderFactory
from src.memory import ConversationTurn
from src.utils import StatusEmoji, get_logger

logger = get_logger(__name__)

bp = Blueprint("chat", __name__)


@bp.route("/ask", methods=["POST"])
def ask_question():
    """Answer a question using the RAG pipeline.

    BYOK: clients may pass X-Provider / X-API-Key / X-Model / X-LLM-Params
    headers. When set, those override the server-side defaults for this
    single request — the server never persists the key.
    """
    if not runtime.system.ready:
        return jsonify({
            "error": "System not ready",
            "status": runtime.system.status,
        }), 503

    # Validate the body before the try block: a bad request must surface
    # as a clean 400 via the central handler, not get swallowed as a 500.
    body = parse_body(AskRequest, request.get_json(silent=True))
    question = body.question

    try:
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
        ws_id = runtime.current_workspace_id()
        try:
            rag = runtime.get_rag_for(ws_id)
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
            selected_files=req_settings.selected_files,
            history=history,
            memory_strategy=req_settings.memory_strategy,
            deduplicate_context=req_settings.deduplicate_context,
            reorder_context=req_settings.reorder_context,
            max_context_tokens=req_settings.max_context_tokens,
            allow_general_knowledge_fallback=req_settings.allow_general_knowledge_fallback,
            prompt_strategy=req_settings.prompt_strategy,
            custom_role=req_settings.custom_role,
            custom_prompt_template=req_settings.custom_prompt_template,
        )

        if result.get("source") == "error":
            return jsonify({
                "error": result.get("answer", "Unknown error"),
                "question": question,
            }), 500

        return jsonify({
            "success": True,
            "question": result["question"],
            "answer": result["answer"],
            "sources": [
                {
                    "title": doc["source"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                }
                for doc in result["source_documents"]
            ],
            "context_used": result.get("context_used", ""),
            "source_type": result.get("source", "unknown"),
            "relevance_score": result.get("relevance_score", 0.0),
        })

    except Exception as e:
        logger.error(f"{StatusEmoji.ERROR} Question handling error: {e}")
        return jsonify({
            "error": f"Error processing question: {str(e)}",
        }), 500
