"""
Tests for the document loader module.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_loader import JSONDocumentLoader, create_sample_data


class TestJSONDocumentLoader:
    """Test cases for JSONDocumentLoader class."""
    
    @pytest.fixture
    def temp_data_folder(self):
        """Create a temporary data folder with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test JSON file
            test_data = [
                {
                    "id": 1,
                    "topic": "Test Topic",
                    "content": "This is test content for the RAG system."
                },
                {
                    "id": 2,
                    "topic": "Another Topic",
                    "content": "More test content here."
                }
            ]
            
            test_file = Path(tmpdir) / "test_data.json"
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False)
            
            yield tmpdir
    
    @pytest.fixture
    def empty_folder(self):
        """Create an empty temporary folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_initialization(self, temp_data_folder):
        """Test loader initialization."""
        loader = JSONDocumentLoader(temp_data_folder)
        
        assert loader.data_folder == Path(temp_data_folder)
        assert loader.document_count == 0
    
    def test_load_json_data(self, temp_data_folder):
        """Test loading JSON files."""
        loader = JSONDocumentLoader(temp_data_folder)
        documents = loader.load_all()
        
        assert len(documents) > 0
        assert loader.document_count > 0
    
    def test_document_content(self, temp_data_folder):
        """Test that loaded documents contain expected content."""
        loader = JSONDocumentLoader(temp_data_folder)
        documents = loader.load_all()
        
        # Check that content is extracted
        all_content = " ".join([doc.page_content for doc in documents])
        assert "Test Topic" in all_content or "test content" in all_content.lower()
    
    def test_document_metadata(self, temp_data_folder):
        """Test that documents have proper metadata."""
        loader = JSONDocumentLoader(temp_data_folder)
        documents = loader.load_all()
        
        for doc in documents:
            assert "source" in doc.metadata
            assert doc.metadata["source"].endswith(".json")
    
    def test_empty_folder_handling(self, empty_folder):
        """Test handling of empty data folder."""
        loader = JSONDocumentLoader(empty_folder)
        documents = loader.load_all()
        
        assert documents == []
        assert loader.document_count == 0
    
    def test_nonexistent_folder_handling(self):
        """Test handling of non-existent folder."""
        loader = JSONDocumentLoader("/nonexistent/path/to/data")
        documents = loader.load_all()
        
        assert documents == []
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid JSON file
            invalid_file = Path(tmpdir) / "invalid.json"
            with open(invalid_file, "w") as f:
                f.write("{ invalid json content")
            
            loader = JSONDocumentLoader(tmpdir)
            documents = loader.load_all()
            
            # Should not crash, just skip invalid files
            assert isinstance(documents, list)
    
    def test_get_file_info(self, temp_data_folder):
        """Test file information retrieval."""
        loader = JSONDocumentLoader(temp_data_folder)
        file_info = loader.get_file_info()
        
        assert len(file_info) > 0
        
        for info in file_info:
            assert "filename" in info
            if "error" not in info:
                assert "document_count" in info
                assert "size_kb" in info


class TestCreateSampleData:
    """Test cases for create_sample_data function."""
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_sample_data(tmpdir)
            
            sample_file = Path(tmpdir) / "sample_knowledge.json"
            assert sample_file.exists()
            
            with open(sample_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            assert isinstance(data, list)
            assert len(data) > 0
    
    def test_sample_data_content(self):
        """Test that sample data has expected structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            create_sample_data(tmpdir)
            
            sample_file = Path(tmpdir) / "sample_knowledge.json"
            with open(sample_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for item in data:
                assert "id" in item
                assert "topic" in item
                assert "definition" in item


class TestTextExtraction:
    """Test cases for text extraction from JSON objects."""
    
    @pytest.fixture
    def loader(self):
        """Create a loader instance for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield JSONDocumentLoader(tmpdir)
    
    def test_extract_from_dict(self, loader):
        """Test text extraction from dictionary."""
        json_obj = {
            "title": "Test Title",
            "content": "Test content here"
        }
        
        text = loader._extract_text(json_obj)
        
        assert "title: Test Title" in text
        assert "content: Test content here" in text
    
    def test_extract_from_nested_dict(self, loader):
        """Test text extraction from nested dictionary."""
        json_obj = {
            "main": {
                "nested_key": "Nested value"
            }
        }
        
        text = loader._extract_text(json_obj)
        
        assert "nested_key: Nested value" in text
    
    def test_extract_from_list(self, loader):
        """Test text extraction from list."""
        json_obj = {
            "items": ["Item 1", "Item 2", "Item 3"]
        }
        
        text = loader._extract_text(json_obj)
        
        assert "Item 1" in text or "Item 2" in text
    
    def test_empty_values_filtered(self, loader):
        """Test that empty values are filtered out."""
        json_obj = {
            "valid": "Valid content",
            "empty": "",
            "short": "ab"  # Too short
        }
        
        text = loader._extract_text(json_obj)
        
        assert "valid: Valid content" in text
        assert "empty" not in text or "empty:" not in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
