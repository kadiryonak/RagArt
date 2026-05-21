"""
Tests for the embeddings module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from langchain_core.documents import Document
except ImportError:  # older langchain
    from langchain.schema import Document


class TestEmbeddingManager:
    """Test cases for EmbeddingManager class."""
    
    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings for testing without loading real models."""
        with patch("src.embeddings.HuggingFaceEmbeddings") as mock_hf:
            mock_instance = Mock()
            mock_instance.embed_query.return_value = [0.1] * 384
            mock_instance.embed_documents.return_value = [[0.1] * 384, [0.2] * 384]
            mock_hf.return_value = mock_instance
            yield mock_hf
    
    def test_initialization(self, mock_embeddings):
        """Test embedding manager initialization."""
        from src.embeddings import EmbeddingManager
        
        manager = EmbeddingManager()
        
        assert manager.model_name is not None
        assert manager.device == "cpu"
        assert manager.chunk_size == 800
        assert manager.chunk_overlap == 150
    
    def test_custom_configuration(self, mock_embeddings):
        """Test initialization with custom configuration."""
        from src.embeddings import EmbeddingManager
        
        manager = EmbeddingManager(
            model_name="custom-model",
            device="cuda",
            chunk_size=500,
            chunk_overlap=100
        )
        
        assert manager.model_name == "custom-model"
        assert manager.device == "cuda"
        assert manager.chunk_size == 500
        assert manager.chunk_overlap == 100
    
    def test_text_splitter_creation(self, mock_embeddings):
        """Test text splitter is created correctly."""
        from src.embeddings import EmbeddingManager
        
        manager = EmbeddingManager(chunk_size=500, chunk_overlap=50)
        splitter = manager.text_splitter
        
        assert splitter is not None
        assert splitter._chunk_size == 500
        assert splitter._chunk_overlap == 50
    
    def test_split_documents(self, mock_embeddings):
        """Test document splitting."""
        from src.embeddings import EmbeddingManager
        
        manager = EmbeddingManager(chunk_size=100, chunk_overlap=20)
        
        # Create a document with content longer than chunk size
        long_content = "This is a test sentence. " * 20
        documents = [Document(page_content=long_content, metadata={"source": "test"})]
        
        split_docs = manager.split_documents(documents)
        
        assert len(split_docs) > 1  # Should be split into multiple chunks
        
        for doc in split_docs:
            assert len(doc.page_content) <= 100 + 20  # Within chunk size + some tolerance
    
    def test_embed_query(self, mock_embeddings):
        """Test query embedding."""
        from src.embeddings import EmbeddingManager
        
        manager = EmbeddingManager()
        
        # Access embeddings property to trigger initialization
        _ = manager.embeddings
        
        result = manager.embed_query("test query")
        
        assert isinstance(result, list)
        assert len(result) == 384  # Mock returns 384-dim vector
    
    def test_embed_documents(self, mock_embeddings):
        """Test document embedding."""
        from src.embeddings import EmbeddingManager
        
        manager = EmbeddingManager()
        
        # Access embeddings property to trigger initialization
        _ = manager.embeddings
        
        result = manager.embed_documents(["doc1", "doc2"])
        
        assert isinstance(result, list)
        assert len(result) == 2
    
    def test_get_embedding_dimension(self, mock_embeddings):
        """Test getting embedding dimension."""
        from src.embeddings import EmbeddingManager
        
        manager = EmbeddingManager()
        
        # Access embeddings property to trigger initialization
        _ = manager.embeddings
        
        dim = manager.get_embedding_dimension()
        
        assert dim == 384  # Mock returns 384-dim vector


class TestTextSplitter:
    """Test cases for text splitting functionality."""
    
    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings."""
        with patch("src.embeddings.HuggingFaceEmbeddings") as mock_hf:
            mock_instance = Mock()
            mock_hf.return_value = mock_instance
            yield mock_hf
    
    def test_turkish_separators(self, mock_embeddings):
        """Test that Turkish text separators work correctly."""
        from src.embeddings import EmbeddingManager
        
        manager = EmbeddingManager(chunk_size=50, chunk_overlap=10)
        splitter = manager.text_splitter
        
        # Check separators include Turkish-appropriate ones
        assert "." in splitter._separators
        assert "!" in splitter._separators
        assert "?" in splitter._separators
    
    def test_preserves_metadata(self, mock_embeddings):
        """Test that splitting preserves document metadata."""
        from src.embeddings import EmbeddingManager
        
        manager = EmbeddingManager(chunk_size=50, chunk_overlap=10)
        
        document = Document(
            page_content="A" * 200,  # Long content
            metadata={"source": "test.json", "custom_key": "custom_value"}
        )
        
        split_docs = manager.split_documents([document])
        
        for doc in split_docs:
            assert doc.metadata["source"] == "test.json"
            assert doc.metadata["custom_key"] == "custom_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
