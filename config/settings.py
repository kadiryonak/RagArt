"""
Configuration settings for the RAG system.

This module loads configuration from environment variables and .env files.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent.parent / ".env"
loaded = load_dotenv(env_path, override=True)
print(f"Loaded .env from {env_path}: {loaded}")


class Settings:
    """Application settings loaded from environment variables."""
    
    @property
    def DEEPSEEK_API_KEY(self) -> Optional[str]:
        return os.getenv("DEEPSEEK_API_KEY")
    
    @property
    def OPENAI_API_KEY(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")
    
    @property
    def MODEL_TYPE(self) -> str:
        return os.getenv("MODEL_TYPE", "local")
    
    @property
    def EMBEDDING_MODEL(self) -> str:
        return os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    
    # Paths
    DATA_FOLDER: str = os.getenv("DATA_FOLDER", "./data")
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "5000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.1"))
    TOP_K_DOCUMENTS: int = int(os.getenv("TOP_K_DOCUMENTS", "5"))
    
    def get_api_key(self) -> Optional[str]:
        """
        Get the appropriate API key based on model type.
        
        Returns:
            API key or None if not available
        """
        if self.MODEL_TYPE == "deepseek":
            return self.DEEPSEEK_API_KEY
        elif self.MODEL_TYPE == "openai":
            return self.OPENAI_API_KEY
        return None
    
    @classmethod
    def validate_api_key(cls, api_key: Optional[str], service: str = "deepseek") -> bool:
        """
        Validate an API key format.
        
        Args:
            api_key: API key to validate
            service: Service type ('deepseek' or 'openai')
            
        Returns:
            True if the API key format is valid
        """
        if not api_key or not isinstance(api_key, str):
            return False
        
        api_key = api_key.strip()
        
        if service.lower() == "deepseek":
            return api_key.startswith("sk-") and len(api_key) >= 32
        elif service.lower() == "openai":
            return api_key.startswith("sk-") and len(api_key) >= 40
        
        return False
    
    @classmethod
    def print_status(cls) -> None:
        """Print current configuration status."""
        print("=" * 50)
        print("RAG System Configuration")
        print("=" * 50)
        
        print(f"Model Type: {cls.MODEL_TYPE}")
        print(f"Data Folder: {cls.DATA_FOLDER}")
        print(f"ChromaDB Path: {cls.CHROMA_DB_PATH}")
        print(f"Server: {cls.HOST}:{cls.PORT}")
        
        # API key status
        if cls.DEEPSEEK_API_KEY:
            valid = cls.validate_api_key(cls.DEEPSEEK_API_KEY, "deepseek")
            status = "✅ Valid" if valid else "❌ Invalid"
            print(f"DeepSeek API Key: {status}")
        else:
            print("DeepSeek API Key: ⚠️ Not set")
        
        if cls.OPENAI_API_KEY:
            valid = cls.validate_api_key(cls.OPENAI_API_KEY, "openai")
            status = "✅ Valid" if valid else "❌ Invalid"
            print(f"OpenAI API Key: {status}")
        else:
            print("OpenAI API Key: ⚠️ Not set")
        
        print("=" * 50)


# Create a singleton settings instance
settings = Settings()
