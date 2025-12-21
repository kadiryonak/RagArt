"""
LLM provider implementations for the RAG system.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

import requests

from src.utils import get_logger, StatusEmoji

logger = get_logger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def generate_general(self, question: str) -> str:
        """
        Generate a general response without RAG context.
        
        Args:
            question: The user's question
            
        Returns:
            Generated response text
        """
        pass


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API provider."""
    
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    def __init__(self, api_key: str):
        """
        Initialize the DeepSeek provider.
        
        Args:
            api_key: DeepSeek API key
        """
        self.api_key = api_key
        self.model = "deepseek-chat"
        self.temperature = 0.1
        self.max_tokens = 800
        self.timeout = 30
    
    def generate(self, prompt: str) -> str:
        """Generate a response using DeepSeek API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }
            
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"{StatusEmoji.ERROR} DeepSeek API error: {response.status_code}")
                return f"API error: {response.status_code}"
                
        except Exception as e:
            logger.error(f"{StatusEmoji.ERROR} DeepSeek connection error: {e}")
            return f"Connection error: {e}"
    
    def generate_general(self, question: str) -> str:
        """Generate a general response without RAG context."""
        general_prompt = f"""Sen Türkçe konuşan bir yapay zeka asistanısın. Kullanıcının sorusunu mümkün olduğunca detaylı ve doğru şekilde yanıtla.

Soru: {question}

Lütfen:
1. Türkçe yanıt ver
2. Bilgilendirici ve faydalı ol
3. Emin olmadığın konularda "kesin değilim" diyebilirsin
4. Kısa ve öz yanıt ver (maksimum 3-4 paragraf)

Yanıtın:"""
        
        return self.generate(general_prompt)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, api_key: str):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self.model = "gpt-3.5-turbo"
        self.temperature = 0.1
        self.max_tokens = 500
        self.timeout = 30
    
    def generate(self, prompt: str) -> str:
        """Generate a response using OpenAI API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"OpenAI API error: {response.status_code}"
                
        except Exception as e:
            return f"OpenAI connection error: {e}"
    
    def generate_general(self, question: str) -> str:
        """Generate a general response without RAG context."""
        return self.generate(f"Please answer this question in Turkish: {question}")


class OllamaProvider(BaseLLMProvider):
    """Ollama local API provider."""
    
    API_URL = "http://localhost:11434/api/generate"
    
    def __init__(self, model: str = "llama2:7b"):
        """
        Initialize the Ollama provider.
        
        Args:
            model: Ollama model name
        """
        self.model = model
        self.timeout = 60
    
    def generate(self, prompt: str) -> str:
        """Generate a response using Ollama API."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                self.API_URL,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["response"]
            else:
                return f"Ollama error: {response.status_code}"
                
        except Exception as e:
            return f"Ollama connection failed: {e}"
    
    def generate_general(self, question: str) -> str:
        """Generate a general response without RAG context."""
        return self.generate(f"Please answer this question in Turkish: {question}")


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Inference API provider using OpenAI-compatible endpoint."""
    
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the HuggingFace provider.
        
        Args:
            api_key: HuggingFace API token
            model: Model name on HuggingFace
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = 500
    
    def generate(self, prompt: str) -> str:
        """Generate a response using HuggingFace OpenAI-compatible API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": 0.1
            }
            
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"].strip()
                return "No response generated."
            elif response.status_code == 402:
                return "HuggingFace kredi limiti doldu. PRO hesaba geçin veya local mode kullanın."
            else:
                error_text = response.text[:200] if response.text else "Unknown error"
                logger.error(f"{StatusEmoji.ERROR} HuggingFace API error: {response.status_code}")
                return f"HuggingFace API error: {response.status_code} - {error_text}"
                
        except Exception as e:
            logger.error(f"{StatusEmoji.ERROR} HuggingFace error: {e}")
            return f"HuggingFace error: {e}"
    
    def generate_general(self, question: str) -> str:
        """Generate a general response without RAG context."""
        general_prompt = f"""Sen Türkçe konuşan bir yapay zeka asistanısın. 
Kullanıcının sorusunu Türkçe olarak yanıtla.

Soru: {question}

Yanıt:"""
        return self.generate(general_prompt)


class LocalProvider(BaseLLMProvider):
    """Local context-aware response generator (no API required)."""
    
    def generate(self, prompt: str) -> str:
        """Generate a context-aware response locally."""
        # Parse context and question from the prompt
        if "BAĞLAM:" in prompt and "SORU:" in prompt:
            context_start = prompt.find("BAĞLAM:") + 8
            context_end = prompt.find("SORU:")
            context = prompt[context_start:context_end].strip()
            
            question_start = prompt.find("SORU:") + 5
            question_end = prompt.find("YANITIN:")
            question = prompt[question_start:question_end].strip()
            
            return self._generate_contextual_answer(context, question)
        
        return "Prompt format error."
    
    def generate_general(self, question: str) -> str:
        """Generate a general response (limited without API)."""
        return "Bu konuda daha detaylı yanıt için bir API anahtarı gereklidir."
    
    def _generate_contextual_answer(self, context: str, question: str) -> str:
        """
        Generate an answer based on the provided context.
        
        Args:
            context: The context text
            question: The user's question
            
        Returns:
            Generated answer
        """
        if not context or len(context.strip()) < 10:
            return "Bu konuda verilen bilgilerde yeterli detay bulunmuyor."
        
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Algorithm-related questions
        if "algoritma" in question_lower:
            if "algoritma" in context_lower:
                lines = context.split("\n")
                relevant_lines = [
                    line for line in lines 
                    if "algoritma" in line.lower() or "algorithm" in line.lower()
                ]
                if relevant_lines:
                    return f"Algorithm information found: {' '.join(relevant_lines[:3])}"
            return "No detailed algorithm information found in the provided data."
        
        # Definition questions
        if any(word in question_lower for word in ["nedir", "ne"]):
            lines = context.split("\n")
            definition_lines = [line for line in lines if ":" in line and len(line) < 200]
            
            if definition_lines:
                return f"Definition: {definition_lines[0]}. {definition_lines[1] if len(definition_lines) > 1 else ''}"
            
            sample = context[:200] + "..." if len(context) > 200 else context
            return f"Available information: {sample}"
        
        # List questions
        if any(word in question_lower for word in ["hangi", "ne gibi", "nasıl"]):
            lines = context.split("\n")
            items = [
                line.strip() for line in lines
                if line.strip() and (
                    line.startswith("-") or line.startswith("•") or 
                    line.startswith("*") or ":" in line
                )
            ]
            
            if items:
                return f"Found items: {', '.join(items[:5])}"
            return f"Available information: {context[:250]}..."
        
        # Default: return context sample
        lines = context.split("\n")
        relevant = [line for line in lines if line.strip() and len(line) > 20][:3]
        
        if relevant:
            return f"Available information: {' '.join(relevant)}"
        
        sample = context[:300] + "..." if len(context) > 300 else context
        return f"Context information: {sample}"


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""
    
    PROVIDERS = {
        "deepseek": DeepSeekProvider,
        "openai": OpenAIProvider,
        "huggingface": HuggingFaceProvider,
        "ollama": OllamaProvider,
        "local": LocalProvider
    }
    
    @classmethod
    def create(
        cls,
        provider_type: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            provider_type: Type of provider ('deepseek', 'openai', 'huggingface', 'ollama', 'local')
            api_key: API key for the provider (if required)
            **kwargs: Additional arguments for the provider
            
        Returns:
            LLM provider instance
            
        Raises:
            ValueError: If provider type is unknown
        """
        provider_type = provider_type.lower()
        
        if provider_type not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider type: {provider_type}. Available: {list(cls.PROVIDERS.keys())}")
        
        provider_class = cls.PROVIDERS[provider_type]
        
        if provider_type in ("deepseek", "openai", "huggingface"):
            if not api_key:
                raise ValueError(f"{provider_type} requires an API key")
            if provider_type == "huggingface":
                model = kwargs.get("model", "meta-llama/Llama-3.1-8B-Instruct")
                return provider_class(api_key, model=model)
            return provider_class(api_key)
        elif provider_type == "ollama":
            return provider_class(**kwargs)
        else:
            return provider_class()
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available provider types."""
        return list(cls.PROVIDERS.keys())

