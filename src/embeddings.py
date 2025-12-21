"""
Embedding management for the RAG system.
"""

from typing import List, Optional

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.utils import get_logger, StatusEmoji

logger = get_logger(__name__)


# Default model optimized for multilingual (including Turkish) embeddings
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingManager:
    """
    Manages text embeddings and text splitting for the RAG system.
    
    This class handles:
    - Loading and managing embedding models
    - Splitting documents into chunks
    - Generating embeddings for text
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        chunk_size: int = 800,
        chunk_overlap: int = 150
    ):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to run the model on ('cpu' or 'cuda')
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.model_name = model_name
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get the embeddings model, initializing if necessary.
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        if self._embeddings is None:
            logger.info(f"{StatusEmoji.LOADING} Loading embedding model: {self.model_name}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device}
            )
            logger.info(f"{StatusEmoji.SUCCESS} Embedding model loaded")
        return self._embeddings
    
    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        Get the text splitter, initializing if necessary.
        
        Returns:
            RecursiveCharacterTextSplitter instance
        """
        if self._text_splitter is None:
            # Separators optimized for Turkish text
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "]
            )
        return self._text_splitter
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of split Document objects
        """
        logger.info(f"{StatusEmoji.LOADING} Splitting {len(documents)} documents...")
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"{StatusEmoji.SUCCESS} Created {len(split_docs)} text chunks")
        return split_docs
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a query text.
        
        Args:
            text: Query text
            
        Returns:
            List of embedding values
        """
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        sample_embedding = self.embed_query("test")
        return len(sample_embedding)
