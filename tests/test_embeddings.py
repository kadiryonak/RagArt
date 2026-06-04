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


class TestPerFormatChunking:
    """Format-aware splitting: each format gets a tuned recursive splitter."""

    @pytest.fixture
    def mock_embeddings(self):
        with patch("src.embeddings.HuggingFaceEmbeddings") as mock_hf:
            mock_hf.return_value = Mock()
            yield mock_hf

    def test_unknown_format_uses_default_splitter(self, mock_embeddings):
        from src.embeddings import EmbeddingManager
        m = EmbeddingManager(chunk_size=500)
        # No format / unknown format → the shared default splitter instance.
        assert m._splitter_for(None) is m.text_splitter
        assert m._splitter_for("") is m.text_splitter
        assert m._splitter_for("xlsx") is m.text_splitter

    def test_markdown_splitter_is_heading_aware(self, mock_embeddings):
        from src.embeddings import EmbeddingManager
        m = EmbeddingManager(chunk_size=400)
        md = m._splitter_for("md")
        assert md is not m.text_splitter
        assert "\n## " in md._separators          # splits on headings first
        assert md._chunk_size == 500              # 400 * 1.25 scale

    def test_json_splitter_uses_larger_chunks(self, mock_embeddings):
        from src.embeddings import EmbeddingManager
        m = EmbeddingManager(chunk_size=400)
        assert m._splitter_for("json")._chunk_size == 600  # 400 * 1.5

    def test_pdf_splitter_is_prose_and_cached(self, mock_embeddings):
        from src.embeddings import EmbeddingManager
        m = EmbeddingManager(chunk_size=400)
        pdf = m._splitter_for("pdf")
        assert pdf._chunk_size == 400             # scale 1.0
        assert ". " in pdf._separators            # sentence-aware prose
        assert m._splitter_for("pdf") is pdf      # cached, not rebuilt

    def test_split_documents_routes_by_format(self, mock_embeddings):
        from src.embeddings import EmbeddingManager
        m = EmbeddingManager(chunk_size=120, chunk_overlap=10)

        used = {}
        real = m._splitter_for

        def spy(fmt):
            s = real(fmt)
            used[fmt] = s
            return s

        m._splitter_for = spy

        docs = [
            Document(page_content="# Başlık\n\n" + "Madde. " * 40,
                     metadata={"source": "a.md", "format": "md"}),
            Document(page_content="Düz metin cümlesi. " * 40,
                     metadata={"source": "b.txt", "format": "txt"}),
        ]
        out = m.split_documents(docs)
        assert len(out) > 2                        # both were chunked
        assert set(used) == {"md", "txt"}          # each format routed
        # Metadata is preserved through splitting.
        assert {d.metadata["format"] for d in out} == {"md", "txt"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
