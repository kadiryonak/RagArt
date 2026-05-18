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

# File upload configuration
ALLOWED_EXTENSIONS = {'json'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global state
rag_system = None
system_ready = False
system_status = "Initializing..."
initialization_error = None


def initialize_rag_system() -> None:
    """Initialize the RAG system in the background."""
    global rag_system, system_ready, system_status, initialization_error
    
    try:
        system_status = "Checking data files..."
        logger.info(f"{StatusEmoji.ROCKET} Initializing RAG system...")
        
        # Ensure data folder exists
        if not os.path.exists(settings.DATA_FOLDER):
            os.makedirs(settings.DATA_FOLDER)
            create_sample_data(settings.DATA_FOLDER)
        
        # Check for JSON files
        json_files = [f for f in os.listdir(settings.DATA_FOLDER) if f.endswith('.json')]
        if not json_files:
            create_sample_data(settings.DATA_FOLDER)
        
        system_status = "Creating vector store..."
        
        # Determine API key and model type
        api_key = settings.get_api_key()
        model_type = settings.MODEL_TYPE
        
        if model_type in ("deepseek", "openai") and not api_key:
            logger.warning(f"{StatusEmoji.WARNING} No API key found, falling back to local model")
            model_type = "local"
        
        # Create RAG system
        rag_system = TurkishRAGSystem(
            data_folder=settings.DATA_FOLDER,
            model_type=model_type,
            api_key=api_key,
            chroma_db_path=settings.CHROMA_DB_PATH
        )
        
        system_status = "Processing documents..."
        rag_system.initialize()
        
        system_status = "Ready"
        system_ready = True
        logger.info(f"{StatusEmoji.SUCCESS} RAG system ready! Model: {model_type}")
        
    except Exception as e:
        initialization_error = str(e)
        system_status = f"Error: {str(e)}"
        system_ready = False
        logger.error(f"{StatusEmoji.ERROR} Initialization error: {e}")


# Start initialization in background thread
init_thread = threading.Thread(target=initialize_rag_system, daemon=True)
init_thread.start()


@app.route("/")
def index():
    """Serve the web interface."""
    templates_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(templates_dir, "templates/index.html")


@app.route("/status")
def get_status():
    """Get system status."""
    return jsonify({
        "ready": system_ready,
        "status": system_status,
        "error": initialization_error,
        "model_type": rag_system.model_type if rag_system else "unknown"
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

        result = rag_system.ask(
            question,
            k=settings.TOP_K_DOCUMENTS,
            llm_provider=llm_override,
            llm_params=req_settings.llm_params.to_dict(),
            retrieval_strategy=req_settings.retrieval_strategy,
            rerank=req_settings.rerank,
            rerank_fetch_k=req_settings.rerank_fetch_k,
            history=history,
            memory_strategy=req_settings.memory_strategy,
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
    """Run system tests."""
    if not system_ready:
        return jsonify({"error": "System not ready"}), 503
    
    test_questions = [
        "Algoritma nedir?",
        "Python hakkında ne biliyorsun?",
        "Yapay zeka ile ilgili bilgiler neler?",
        "Veri yapıları nedir?"
    ]
    
    results = []
    for question in test_questions:
        try:
            result = rag_system.ask(question)
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
    """Get information about loaded data."""
    if not system_ready:
        return jsonify({"error": "System not ready"}), 503
    
    try:
        file_info = rag_system.document_loader.get_file_info()
        
        total_documents = sum(
            f.get("document_count", 0) for f in file_info 
            if "error" not in f
        )
        
        return jsonify({
            "files": file_info,
            "total_files": len(file_info),
            "total_documents": total_documents,
            "model_type": rag_system.model_type
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    return jsonify({
        "status": "healthy",
        "system_ready": system_ready,
        "model_type": rag_system.model_type if rag_system else "unknown",
        "api_available": bool(rag_system and rag_system.api_key) if rag_system else False
    })


@app.route("/stats")
def get_stats():
    """Get system statistics."""
    if not system_ready:
        return jsonify({"error": "System not ready"}), 503
    
    return jsonify(rag_system.get_stats())


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload for knowledge base."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Only JSON files are allowed"}), 400
    
    filepath = None
    try:
        # Secure the filename
        filename = secure_filename(file.filename)

        # Ensure data folder exists
        os.makedirs(settings.DATA_FOLDER, exist_ok=True)

        # Save the file
        filepath = os.path.join(settings.DATA_FOLDER, filename)
        file.save(filepath)

        # Validate JSON content
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Count documents
        doc_count = len(data) if isinstance(data, list) else 1

        logger.info(f"{StatusEmoji.SUCCESS} File uploaded: {filename} ({doc_count} documents)")

        return jsonify({
            "success": True,
            "filename": filename,
            "document_count": doc_count,
            "message": f"File '{filename}' uploaded successfully. Use /reindex to update the knowledge base."
        })

    except json.JSONDecodeError:
        # Remove invalid file
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": "Invalid JSON file format"}), 400

    except Exception as e:
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
    filepath = os.path.join(settings.DATA_FOLDER, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        os.remove(filepath)
        logger.info(f"{StatusEmoji.SUCCESS} File deleted: {filename}")
        
        return jsonify({
            "success": True,
            "message": f"File '{filename}' deleted. Use /reindex to update the knowledge base."
        })
        
    except Exception as e:
        logger.error(f"{StatusEmoji.ERROR} Delete error: {e}")
        return jsonify({"error": f"Delete failed: {str(e)}"}), 500


@app.route("/reindex", methods=["POST"])
def reindex_documents():
    """Rebuild the vector store with current documents."""
    global rag_system, system_ready, system_status
    
    try:
        system_status = "Reindexing documents..."
        system_ready = False
        
        logger.info(f"{StatusEmoji.LOADING} Reindexing documents...")
        
        # Reinitialize the vector store
        rag_system.create_vector_store()
        
        system_status = "Ready"
        system_ready = True
        
        logger.info(f"{StatusEmoji.SUCCESS} Reindexing complete!")
        
        return jsonify({
            "success": True,
            "message": "Knowledge base reindexed successfully.",
            "document_count": rag_system.document_loader.document_count
        })
        
    except Exception as e:
        system_status = f"Error: {str(e)}"
        logger.error(f"{StatusEmoji.ERROR} Reindex error: {e}")
        return jsonify({"error": f"Reindex failed: {str(e)}"}), 500


@app.route("/list-files")
def list_files():
    """List all files in the knowledge base."""
    try:
        if not os.path.exists(settings.DATA_FOLDER):
            return jsonify({"files": [], "total": 0})
        
        files = []
        for filename in os.listdir(settings.DATA_FOLDER):
            if filename.endswith(".json"):
                filepath = os.path.join(settings.DATA_FOLDER, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    doc_count = len(data) if isinstance(data, list) else 1
                    size_kb = round(os.path.getsize(filepath) / 1024, 2)
                    
                    files.append({
                        "filename": filename,
                        "document_count": doc_count,
                        "size_kb": size_kb
                    })
                except Exception:
                    files.append({
                        "filename": filename,
                        "error": "Could not read file"
                    })
        
        return jsonify({
            "files": files,
            "total": len(files)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
