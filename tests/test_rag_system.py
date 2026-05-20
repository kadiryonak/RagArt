"""
Tests for the main RAG system.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTurkishRAGSystem:
    """Test cases for TurkishRAGSystem class."""
    
    @pytest.fixture
    def temp_data_folder(self):
        """Create a temporary data folder with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_data = [
                {
                    "id": 1,
                    "topic": "Algorithm",
                    "definition": "Algoritma, belirli bir problemi çözmek için tasarlanmış adımların sıralı bir listesidir.",
                    "details": "Algoritma bilgisayar biliminin temel taşlarından biridir."
                },
                {
                    "id": 2,
                    "topic": "Data Structures",
                    "definition": "Veri yapıları, verileri organize etme yöntemleridir.",
                    "details": "Array, linked list, stack, queue gibi türleri vardır."
                }
            ]
            
            test_file = Path(tmpdir) / "test_knowledge.json"
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False)
            
            yield tmpdir
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock embeddings to avoid loading real models."""
        with patch("src.embeddings.HuggingFaceEmbeddings") as mock_hf:
            mock_instance = Mock()
            mock_instance.embed_query.return_value = [0.1] * 384
            mock_instance.embed_documents.return_value = [[0.1] * 384]
            mock_hf.return_value = mock_instance
            yield mock_hf
    
    @pytest.fixture
    def mock_chroma(self):
        """Mock ChromaDB to avoid file system operations."""
        with patch("chromadb.PersistentClient") as mock_client:
            mock_collection = Mock()
            mock_client.return_value.delete_collection = Mock()
            yield mock_client
    
    def test_initialization(self, temp_data_folder, mock_embeddings, mock_chroma):
        """Test RAG system initialization."""
        from src.rag_system import TurkishRAGSystem
        
        rag = TurkishRAGSystem(
            data_folder=temp_data_folder,
            model_type="local"
        )
        
        assert rag.data_folder == temp_data_folder
        assert rag.model_type == "local"
        assert rag.vector_store is None  # Not initialized yet
    
    def test_initialization_with_api_key(self, temp_data_folder, mock_embeddings, mock_chroma):
        """Test initialization with API key."""
        from src.rag_system import TurkishRAGSystem
        
        rag = TurkishRAGSystem(
            data_folder=temp_data_folder,
            model_type="deepseek",
            api_key="test-api-key"
        )
        
        assert rag.model_type == "deepseek"
        assert rag.api_key == "test-api-key"
    
    def test_fallback_to_local_without_api_key(self, temp_data_folder, mock_embeddings, mock_chroma):
        """Test that system falls back to local when API key is missing."""
        from src.rag_system import TurkishRAGSystem
        
        rag = TurkishRAGSystem(
            data_folder=temp_data_folder,
            model_type="deepseek"
            # No API key provided
        )
        
        # Should use local provider as fallback
        assert rag.llm_provider is not None
    
    def test_calculate_relevance_score(self, temp_data_folder, mock_embeddings, mock_chroma):
        """Test relevance score calculation."""
        from src.rag_system import TurkishRAGSystem
        try:
            from langchain_core.documents import Document
        except ImportError:  # older langchain
            from langchain.schema import Document
        
        rag = TurkishRAGSystem(data_folder=temp_data_folder, model_type="local")
        
        question = "algoritma nedir"
        documents = [
            Document(page_content="Algoritma bir problem çözme yöntemidir.", metadata={}),
            Document(page_content="Veri yapıları hakkında bilgi.", metadata={})
        ]
        
        score = rag.calculate_relevance_score(question, documents)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_calculate_relevance_score_empty_docs(self, temp_data_folder, mock_embeddings, mock_chroma):
        """Test relevance score with empty documents."""
        from src.rag_system import TurkishRAGSystem
        
        rag = TurkishRAGSystem(data_folder=temp_data_folder, model_type="local")
        
        score = rag.calculate_relevance_score("test question", [])
        
        assert score == 0.0
    
    def test_get_stats(self, temp_data_folder, mock_embeddings, mock_chroma):
        """Test getting system statistics."""
        from src.rag_system import TurkishRAGSystem
        
        rag = TurkishRAGSystem(
            data_folder=temp_data_folder,
            model_type="local"
        )
        
        stats = rag.get_stats()
        
        assert "model_type" in stats
        assert "api_available" in stats
        assert "vector_store_ready" in stats
        assert stats["model_type"] == "local"
        assert stats["api_available"] is False
    
    def test_ask_without_initialization(self, temp_data_folder, mock_embeddings, mock_chroma):
        """Test asking question without initializing vector store."""
        from src.rag_system import TurkishRAGSystem
        
        rag = TurkishRAGSystem(data_folder=temp_data_folder, model_type="local")
        
        result = rag.ask("Test question")
        
        assert result["source"] == "error"
        assert "not initialized" in result["answer"].lower() or "vector store" in result["answer"].lower()


class TestRAGSystemPrompts:
    """Test cases for prompt templates."""
    
    def test_system_prompt_format(self):
        """Test that system prompt has correct placeholders."""
        from src.rag_system import TURKISH_SYSTEM_PROMPT
        
        assert "{context}" in TURKISH_SYSTEM_PROMPT
        assert "{question}" in TURKISH_SYSTEM_PROMPT
        assert "BAĞLAM" in TURKISH_SYSTEM_PROMPT
        assert "SORU" in TURKISH_SYSTEM_PROMPT
    
    def test_system_prompt_substitution(self):
        """Test prompt substitution works correctly."""
        from src.rag_system import TURKISH_SYSTEM_PROMPT
        
        formatted = TURKISH_SYSTEM_PROMPT.format(
            context="Test context",
            question="Test question"
        )
        
        assert "Test context" in formatted
        assert "Test question" in formatted


class TestIntegration:
    """Integration test scenarios."""
    
    @pytest.fixture
    def full_test_setup(self):
        """Create a complete test environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            test_data = [
                {
                    "topic": "Python",
                    "definition": "Python yüksek seviyeli bir programlama dilidir.",
                    "examples": ["Django", "Flask", "NumPy"]
                }
            ]
            
            data_folder = Path(tmpdir) / "data"
            data_folder.mkdir()
            
            with open(data_folder / "test.json", "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False)
            
            chroma_folder = Path(tmpdir) / "chroma"
            chroma_folder.mkdir()
            
            yield {
                "data_folder": str(data_folder),
                "chroma_folder": str(chroma_folder)
            }
    
    @pytest.mark.integration
    def test_document_loading_flow(self, full_test_setup):
        """Test the document loading flow."""
        from src.document_loader import JSONDocumentLoader
        
        loader = JSONDocumentLoader(full_test_setup["data_folder"])
        documents = loader.load_all()
        
        assert len(documents) > 0
        
        # Check content is extracted
        content = documents[0].page_content.lower()
        assert "python" in content or "programlama" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
