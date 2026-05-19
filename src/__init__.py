"""
Turkish RAG System - A Retrieval-Augmented Generation system optimized for Turkish language.

This package provides a complete RAG solution with:
- Document loading from JSON files
- Vector store management with ChromaDB
- Multiple LLM provider support (DeepSeek, OpenAI, Ollama, Local)
- Semantic search and relevance scoring
"""

__version__ = "1.0.0"
__author__ = "RAG Team"

# Lazy imports — avoid pulling heavy deps (chromadb, langchain) when
# only lightweight submodules (e.g. src.cache) are needed.
__all__ = [
    "TurkishRAGSystem",
    "JSONDocumentLoader",
    "EmbeddingManager",
    "LLMProviderFactory",
]


def __getattr__(name: str):
    if name == "TurkishRAGSystem":
        from src.rag_system import TurkishRAGSystem
        return TurkishRAGSystem
    if name == "JSONDocumentLoader":
        from src.document_loader import JSONDocumentLoader
        return JSONDocumentLoader
    if name == "EmbeddingManager":
        from src.embeddings import EmbeddingManager
        return EmbeddingManager
    if name == "LLMProviderFactory":
        from src.llm_providers import LLMProviderFactory
        return LLMProviderFactory
    raise AttributeError(f"module 'src' has no attribute {name!r}")

