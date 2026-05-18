"""
Main RAG (Retrieval-Augmented Generation) system implementation.
"""

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain.vectorstores import Chroma
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.document_loader import JSONDocumentLoader, create_sample_data
from src.embeddings import EmbeddingManager
from src.llm_providers import BaseLLMProvider, LLMProviderFactory
from src.utils import get_logger, StatusEmoji, calculate_word_overlap

logger = get_logger(__name__)


# Turkish system prompt template for RAG
TURKISH_SYSTEM_PROMPT = """Sen Türkçe konuşan bir yapay zeka asistanısın. Görevin verilen BAĞLAM bilgilerini kullanarak kullanıcının sorusunu yanıtla.

KURALLAR:
1. Sadece verilen BAĞLAM bilgilerini kullan
2. Bağlamda bilgi yoksa "Bu konuda verilen bilgilerde yeterli detay bulunmuyor" de
3. Kısa ve öz yanıt ver
4. Bağlamdan doğrudan alıntı yapabilirsin
5. Türkçe yanıt ver

BAĞLAM:
{context}

SORU: {question}

YANITIN:"""


class TurkishRAGSystem:
    """
    A Retrieval-Augmented Generation system optimized for Turkish language.
    
    This system provides:
    - Document loading from JSON files
    - Vector-based semantic search
    - LLM-powered question answering
    - Fallback to general knowledge when context is insufficient
    
    Attributes:
        data_folder: Path to JSON data files
        model_type: LLM provider type
        vector_store: ChromaDB vector store instance
    """
    
    # Relevance threshold for determining if RAG context is sufficient
    RELEVANCE_THRESHOLD = 0.1
    
    def __init__(
        self,
        data_folder: str = "./data",
        model_type: str = "local",
        api_key: Optional[str] = None,
        chroma_db_path: str = "./chroma_db",
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the RAG system.
        
        Args:
            data_folder: Path to the folder containing JSON data files
            model_type: LLM provider type ('deepseek', 'openai', 'ollama', 'local')
            api_key: API key for the LLM provider (if required)
            chroma_db_path: Path for ChromaDB persistence
            embedding_model: Custom embedding model name (optional)
        """
        self.data_folder = data_folder
        self.model_type = model_type
        self.api_key = api_key
        self.chroma_db_path = chroma_db_path
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Initialize embedding manager
        embed_kwargs = {}
        if embedding_model:
            embed_kwargs["model_name"] = embedding_model
        self.embedding_manager = EmbeddingManager(**embed_kwargs)
        
        # Initialize LLM provider
        self.llm_provider: BaseLLMProvider = self._create_llm_provider()
        
        # Vector store (initialized later)
        self.vector_store: Optional[Chroma] = None
        
        # Document loader
        self.document_loader = JSONDocumentLoader(data_folder)
        
        # System prompt template
        self.system_prompt = TURKISH_SYSTEM_PROMPT
    
    def _create_llm_provider(self) -> BaseLLMProvider:
        """Create the LLM provider based on configuration."""
        try:
            return LLMProviderFactory.create(
                provider_type=self.model_type,
                api_key=self.api_key
            )
        except ValueError as e:
            logger.warning(f"{StatusEmoji.WARNING} {e}. Falling back to local provider.")
            return LLMProviderFactory.create("local")
    
    def initialize(self) -> None:
        """
        Initialize the RAG system by creating the vector store.
        
        This method:
        1. Tries to load existing collection if available
        2. If not available, loads documents and creates new collection
        """
        logger.info(f"{StatusEmoji.ROCKET} Initializing RAG system...")
        
        # Try to load existing collection first
        if self.load_existing_vector_store():
            logger.info(f"{StatusEmoji.SUCCESS} RAG system ready (loaded existing collection)!")
            return
        
        # Create new collection from documents
        self.create_vector_store()
        logger.info(f"{StatusEmoji.SUCCESS} RAG system ready!")
    
    def create_vector_store(self) -> None:
        """
        Create the vector store from loaded documents.
        
        Raises:
            ValueError: If no documents are loaded
        """
        logger.info(f"{StatusEmoji.DOCUMENT} Loading JSON data...")
        documents = self.document_loader.load_all()
        
        if not documents:
            raise ValueError(
                f"{StatusEmoji.ERROR} No documents loaded! Check your data folder: {self.data_folder}"
            )
        
        # Split documents into chunks
        split_docs = self.embedding_manager.split_documents(documents)
        
        logger.info(f"{StatusEmoji.DOCUMENT} Created {len(split_docs)} text chunks")
        
        # Clear existing collection
        try:
            self.chroma_client.delete_collection("turkish_rag_collection")
            logger.info(f"{StatusEmoji.INFO} Deleted old collection")
        except Exception:
            pass
        
        logger.info(f"{StatusEmoji.DATABASE} Creating vector store...")
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embedding_manager.embeddings,
            client=self.chroma_client,
            collection_name="turkish_rag_collection"
        )
        
        logger.info(f"{StatusEmoji.SUCCESS} Vector store created successfully!")
    
    def load_existing_vector_store(self) -> bool:
        """
        Try to load an existing vector store if available.
        
        Returns:
            True if existing collection was loaded, False otherwise
        """
        try:
            # Check if collection exists
            collections = self.chroma_client.list_collections()
            collection_names = [c.name for c in collections]
            
            if "turkish_rag_collection" in collection_names:
                # Load existing collection
                self.vector_store = Chroma(
                    client=self.chroma_client,
                    collection_name="turkish_rag_collection",
                    embedding_function=self.embedding_manager.embeddings
                )
                
                # Verify it has documents
                collection = self.chroma_client.get_collection("turkish_rag_collection")
                doc_count = collection.count()
                
                if doc_count > 0:
                    logger.info(f"{StatusEmoji.SUCCESS} Loaded existing collection with {doc_count} documents")
                    return True
                else:
                    logger.info(f"{StatusEmoji.WARNING} Collection exists but is empty")
                    return False
            
            return False
        except Exception as e:
            logger.warning(f"{StatusEmoji.WARNING} Could not load existing collection: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of relevant Document objects
        """
        if not self.vector_store:
            return []
        
        return self.vector_store.similarity_search(query, k=k)
    
    def calculate_relevance_score(self, question: str, documents: List[Document]) -> float:
        """
        Calculate the relevance score between a question and retrieved documents.
        
        Args:
            question: The user's question
            documents: Retrieved documents
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not documents:
            return 0.0
        
        total_score = 0.0
        for doc in documents:
            score = calculate_word_overlap(question, doc.page_content)
            total_score += score
        
        return total_score / len(documents)
    
    def ask(self, question: str, k: int = 5) -> Dict[str, Any]:
        """
        Ask a question and get an answer using the RAG system.
        
        Args:
            question: The question in Turkish
            k: Number of context documents to retrieve
            
        Returns:
            Dictionary containing:
            - question: Original question
            - answer: Generated answer
            - source_documents: List of source documents used
            - context_used: Context text used
            - source: Answer source ('rag_system', 'deepseek_fallback', etc.)
            - relevance_score: Relevance score of retrieved documents
        """
        if not self.vector_store:
            return {
                "question": question,
                "answer": f"{StatusEmoji.WARNING} Vector store not initialized!",
                "source_documents": [],
                "context_used": "",
                "source": "error"
            }
        
        try:
            # Search for relevant documents
            logger.info(f"{StatusEmoji.SEARCH} Searching for: '{question}'...")
            relevant_docs = self.search(question, k=k)
            
            # Calculate relevance score
            relevance_score = self.calculate_relevance_score(question, relevant_docs)
            logger.info(f"{StatusEmoji.INFO} Relevance score: {relevance_score:.3f}")
            
            # Check if we have sufficient context
            if not relevant_docs or relevance_score < self.RELEVANCE_THRESHOLD:
                logger.info(f"{StatusEmoji.INFO} Insufficient context, using fallback...")
                return self._fallback_response(question, relevance_score)
            
            # Build context from documents
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                source = doc.metadata.get("source", "Unknown")
                context_parts.append(f"[Source {i} - {source}]\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            full_prompt = self.system_prompt.format(
                context=context,
                question=question
            )
            
            logger.info(f"{StatusEmoji.ROBOT} Generating response ({self.model_type})...")
            
            # Get LLM response
            answer = self.llm_provider.generate(full_prompt)
            
            # Return result
            return {
                "question": question,
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "metadata": doc.metadata
                    }
                    for doc in relevant_docs
                ],
                "context_used": context[:500] + "..." if len(context) > 500 else context,
                "source": "rag_system",
                "relevance_score": relevance_score
            }
            
        except Exception as e:
            logger.error(f"{StatusEmoji.ERROR} Error: {e}")
            return {
                "question": question,
                "answer": f"{StatusEmoji.ERROR} Error: {str(e)}",
                "source_documents": [],
                "context_used": "",
                "source": "error"
            }
    
    def _fallback_response(self, question: str, relevance_score: float) -> Dict[str, Any]:
        """
        Generate a fallback response when RAG context is insufficient.
        
        Args:
            question: The user's question
            relevance_score: The relevance score that triggered fallback
            
        Returns:
            Response dictionary
        """
        if self.model_type in ("deepseek", "openai") and self.api_key:
            # Use LLM for general knowledge
            general_answer = self.llm_provider.generate_general(question)
            
            # Get data summary
            data_summary = self._get_data_summary()
            
            final_answer = f"""Bu konuda mevcut verilerimde yeterli detay bulunamadı, ancak genel bilgilerim şunlar:

{general_answer}

---
📚 **Available Data Topics:**
{data_summary}

💡 **Note:** This response is from general knowledge ({self.model_type}). You can expand your knowledge base for more specific information."""
            
            return {
                "question": question,
                "answer": final_answer,
                "source_documents": [],
                "context_used": "",
                "source": "deepseek_fallback",
                "relevance_score": relevance_score
            }
        else:
            return {
                "question": question,
                "answer": f"Insufficient information in the knowledge base. Relevance score: {relevance_score:.3f}. Consider expanding your data.",
                "source_documents": [],
                "context_used": "",
                "source": "insufficient_data",
                "relevance_score": relevance_score
            }
    
    def _get_data_summary(self) -> str:
        """
        Get a summary of available data topics.
        
        Returns:
            Summary string
        """
        try:
            sample_docs = self.vector_store.similarity_search("", k=5)
            
            topics = set()
            for doc in sample_docs:
                source = doc.metadata.get("source", "")
                if source:
                    topic = source.replace(".json", "").replace("_", " ").title()
                    topics.add(topic)
            
            if topics:
                return f"Topics: {', '.join(list(topics)[:8])}"
            return "Various technical topics available."
            
        except Exception:
            return "Technical topics available."
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system stats
        """
        return {
            "model_type": self.model_type,
            "api_available": bool(self.api_key),
            "document_count": self.document_loader.document_count,
            "vector_store_ready": self.vector_store is not None,
            "data_folder": self.data_folder
        }
    
    def run_tests(self) -> List[Dict[str, Any]]:
        """
        Run test questions to verify system functionality.
        
        Returns:
            List of test results
        """
        test_questions = [
            "Bu veriler hakkında ne biliyorsun?",
            "Algoritma nedir?",
            "Hangi konular ele alınıyor?",
            "Yapay zeka nedir?",
            "Python hakkında bilgi ver"
        ]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{StatusEmoji.CHECK} RUNNING SYSTEM TESTS")
        logger.info(f"{'='*60}")
        
        results = []
        for question in test_questions:
            logger.info(f"\n{StatusEmoji.QUESTION} Question: {question}")
            result = self.ask(question)
            
            results.append({
                "question": question,
                "answer_preview": result["answer"][:200] + "...",
                "source": result.get("source", "unknown"),
                "relevance_score": result.get("relevance_score", 0.0),
                "sources_count": len(result["source_documents"])
            })
            
            logger.info(f"{StatusEmoji.ANSWER} Answer: {result['answer'][:100]}...")
            logger.info(f"{StatusEmoji.SOURCE} Source: {result.get('source', 'unknown')}")
        
        return results
