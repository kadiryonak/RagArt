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

from src.rag_system import TurkishRAGSystem
from src.document_loader import JSONDocumentLoader
from src.embeddings import EmbeddingManager
from src.llm_providers import LLMProviderFactory

__all__ = [
    "TurkishRAGSystem",
    "JSONDocumentLoader", 
    "EmbeddingManager",
    "LLMProviderFactory",
]
