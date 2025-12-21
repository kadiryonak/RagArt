"""
Tests for the LLM providers module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_providers import (
    BaseLLMProvider,
    DeepSeekProvider,
    OpenAIProvider,
    OllamaProvider,
    LocalProvider,
    LLMProviderFactory
)


class TestDeepSeekProvider:
    """Test cases for DeepSeek provider."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = DeepSeekProvider(api_key="test-key")
        
        assert provider.api_key == "test-key"
        assert provider.model == "deepseek-chat"
        assert provider.temperature == 0.1
    
    @patch("requests.post")
    def test_generate_success(self, mock_post):
        """Test successful response generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        provider = DeepSeekProvider(api_key="test-key")
        result = provider.generate("Test prompt")
        
        assert result == "Test response"
        mock_post.assert_called_once()
    
    @patch("requests.post")
    def test_generate_api_error(self, mock_post):
        """Test API error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        provider = DeepSeekProvider(api_key="test-key")
        result = provider.generate("Test prompt")
        
        assert "error" in result.lower()
    
    @patch("requests.post")
    def test_generate_connection_error(self, mock_post):
        """Test connection error handling."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Network error")
        
        provider = DeepSeekProvider(api_key="test-key")
        result = provider.generate("Test prompt")
        
        assert "error" in result.lower()


class TestOpenAIProvider:
    """Test cases for OpenAI provider."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = OpenAIProvider(api_key="test-key")
        
        assert provider.api_key == "test-key"
        assert provider.model == "gpt-3.5-turbo"
    
    @patch("requests.post")
    def test_generate_success(self, mock_post):
        """Test successful response generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OpenAI response"}}]
        }
        mock_post.return_value = mock_response
        
        provider = OpenAIProvider(api_key="test-key")
        result = provider.generate("Test prompt")
        
        assert result == "OpenAI response"


class TestOllamaProvider:
    """Test cases for Ollama provider."""
    
    def test_initialization(self):
        """Test provider initialization."""
        provider = OllamaProvider()
        
        assert provider.model == "llama2:7b"
        assert provider.timeout == 60
    
    def test_custom_model(self):
        """Test initialization with custom model."""
        provider = OllamaProvider(model="mistral:7b")
        
        assert provider.model == "mistral:7b"
    
    @patch("requests.post")
    def test_generate_success(self, mock_post):
        """Test successful response generation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Ollama response"}
        mock_post.return_value = mock_response
        
        provider = OllamaProvider()
        result = provider.generate("Test prompt")
        
        assert result == "Ollama response"


class TestLocalProvider:
    """Test cases for local provider."""
    
    def test_generate_with_context(self):
        """Test response generation with context."""
        provider = LocalProvider()
        
        prompt = """BAĞLAM:
Algoritma: Belirli bir problemi çözmek için tasarlanmış adımlar.

SORU: Algoritma nedir?

YANITIN:"""
        
        result = provider.generate(prompt)
        
        # Should return something relevant
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_generate_without_proper_format(self):
        """Test response with invalid prompt format."""
        provider = LocalProvider()
        result = provider.generate("Invalid prompt format")
        
        assert "error" in result.lower() or "format" in result.lower()
    
    def test_generate_general(self):
        """Test general response generation."""
        provider = LocalProvider()
        result = provider.generate_general("Test question")
        
        assert isinstance(result, str)
        assert len(result) > 0


class TestLLMProviderFactory:
    """Test cases for LLM provider factory."""
    
    def test_create_deepseek_provider(self):
        """Test creating DeepSeek provider."""
        provider = LLMProviderFactory.create("deepseek", api_key="test-key")
        
        assert isinstance(provider, DeepSeekProvider)
        assert provider.api_key == "test-key"
    
    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        provider = LLMProviderFactory.create("openai", api_key="test-key")
        
        assert isinstance(provider, OpenAIProvider)
    
    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        provider = LLMProviderFactory.create("ollama")
        
        assert isinstance(provider, OllamaProvider)
    
    def test_create_local_provider(self):
        """Test creating local provider."""
        provider = LLMProviderFactory.create("local")
        
        assert isinstance(provider, LocalProvider)
    
    def test_case_insensitive_type(self):
        """Test that provider type is case insensitive."""
        provider1 = LLMProviderFactory.create("LOCAL")
        provider2 = LLMProviderFactory.create("Local")
        
        assert isinstance(provider1, LocalProvider)
        assert isinstance(provider2, LocalProvider)
    
    def test_unknown_provider_raises_error(self):
        """Test that unknown provider type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LLMProviderFactory.create("unknown")
        
        assert "Unknown provider type" in str(exc_info.value)
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LLMProviderFactory.create("deepseek")
        
        assert "requires an API key" in str(exc_info.value)
    
    def test_get_available_providers(self):
        """Test getting list of available providers."""
        providers = LLMProviderFactory.get_available_providers()
        
        assert "deepseek" in providers
        assert "openai" in providers
        assert "ollama" in providers
        assert "local" in providers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
