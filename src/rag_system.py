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
from src.llm_providers import BaseLLMProvider, LLMProviderFactory, LocalProvider
from src.retrievers import (
    BaseRetriever,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankedRetriever,
    RetrievedDoc,
)
from src.memory import (
    BaseMemory,
    ConversationTurn,
    NoMemory,
    SlidingWindowMemory,
    SummaryBufferMemory,
    VectorRetrievalMemory,
)
from src.context import (
    LostInTheMiddleReorderer,
    ProcessorChain,
    RedundancyFilter,
    TokenBudgetTrimmer,
)
from src.prompt_strategies import (
    BasePromptStrategy,
    CustomStrategy,
    DirectStrategy,
    PromptStrategyFactory,
    RoleBasedStrategy,
    StrategyContext,
)
from src.prompt_strategies.multi_query import MultiQueryStrategy
from src.utils import get_logger, StatusEmoji, calculate_word_overlap

logger = get_logger(__name__)


# Turkish system prompt template for RAG
TURKISH_SYSTEM_PROMPT = """Sen Türkçe konuşan bir yapay zeka asistanısın. Görevin verilen BAĞLAM bilgilerini kullanarak kullanıcının sorusunu yanıtla.

KURALLAR (sıkı sıkıya uyulacak):
1. Sadece verilen BAĞLAM bilgilerini kullan; dış bilgi kullanma.
2. Karar mantığı:
   - BAĞLAM soruyu cevaplamak için yeterliyse → doğrudan, akıcı bir cevap yaz.
   - BAĞLAM yetersizse → SADECE şu cümleyi yaz, başka HİÇBİR ŞEY ekleme:
     "Bu konuda verilen bilgilerde yeterli detay bulunmuyor."
   - "Yetersiz" deyip ardından alıntı yapmak/parça parça bilgi vermek YASAK.
3. Cevap içinde "[Source 1 - ...]", "BAĞLAM" gibi etiketleri ASLA gösterme.
4. Cevap doğal, akıcı Türkçe olsun; meta yorum yapma ("bağlamda şu var" gibi).
5. Mümkün olduğunca kısa ve net yaz (1-3 paragraf yeterli).

BAĞLAM:
{context}

SORU: {question}

YANIT (sadece son cevap metni; etiket/alıntı meta yok):"""


TURKISH_SYSTEM_PROMPT_WITH_MEMORY = """Sen Türkçe konuşan bir yapay zeka asistanısın. Görevin verilen BAĞLAM bilgilerini ve önceki KONUŞMA geçmişini kullanarak kullanıcının sorusunu yanıtla.

KURALLAR (sıkı sıkıya uyulacak):
1. Sadece verilen BAĞLAM bilgilerini kullan; dış bilgi kullanma.
2. KONUŞMA geçmişini sadece referansları çözmek için kullan ("o", "bu konu" gibi).
3. Karar mantığı:
   - BAĞLAM soruyu cevaplamak için yeterliyse → doğrudan, akıcı bir cevap yaz.
   - BAĞLAM yetersizse → SADECE şu cümleyi yaz, başka HİÇBİR ŞEY ekleme:
     "Bu konuda verilen bilgilerde yeterli detay bulunmuyor."
   - "Yetersiz" deyip ardından alıntı/parça bilgi vermek YASAK.
4. Cevap içinde "[Source 1 - ...]", "BAĞLAM" gibi etiketleri ASLA gösterme.
5. Doğal, akıcı Türkçe; meta yorum yok.

ÖNCEKİ KONUŞMA:
{memory_context}

BAĞLAM:
{context}

SORU: {question}

YANIT (sadece son cevap metni; etiket/alıntı meta yok):"""


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

        # Retrievers — built when vector store is ready (dense always,
        # sparse from the same docs, hybrid as RRF of the two).
        # Reranker is lazy: only constructed when first requested (model
        # download ~400MB on first use).
        self._dense_retriever: Optional[BaseRetriever] = None
        self._sparse_retriever: Optional[BaseRetriever] = None
        self._hybrid_retriever: Optional[BaseRetriever] = None
        self._reranker_cache: Dict[str, RerankedRetriever] = {}
        
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

        # Sparse + hybrid retrievers built over the same split chunks
        self._build_retrievers(split_docs)

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
                    # BM25 is in-memory only — rebuild from the source JSON files
                    # so dense/sparse/hybrid all stay coherent.
                    self._build_retrievers(self._reload_split_chunks())
                    logger.info(f"{StatusEmoji.SUCCESS} Loaded existing collection with {doc_count} documents")
                    return True
                else:
                    logger.info(f"{StatusEmoji.WARNING} Collection exists but is empty")
                    return False
            
            return False
        except Exception as e:
            logger.warning(f"{StatusEmoji.WARNING} Could not load existing collection: {e}")
            return False
    
    # ----- Retriever lifecycle -----

    def _reload_split_chunks(self) -> List[Document]:
        """Disk'teki JSON'ları yükle ve chunk'la — BM25 in-memory için."""
        documents = self.document_loader.load_all()
        if not documents:
            return []
        return self.embedding_manager.split_documents(documents)

    def _build_retrievers(self, split_docs: List[Document]) -> None:
        """Vector store hazırken dense+sparse+hybrid retriever'ları kur.

        ÖNEMLİ: Reindex bu metodu yeniden çağırır. Eski cache'lenmiş
        RerankedRetriever'lar ESKİ vector_store'a sarılı kalır ve sonraki
        sorgularda silinmiş ChromaDB koleksiyonuna istek atar
        ("Collection [UUID] does not exist"). Bu yüzden her rebuild'de
        reranker cache'ini SIFIRLAMAK ŞART.
        """
        if self.vector_store is not None:
            self._dense_retriever = DenseRetriever(self.vector_store)
        if split_docs:
            self._sparse_retriever = BM25Retriever(split_docs)
        if self._dense_retriever and self._sparse_retriever:
            self._hybrid_retriever = HybridRetriever(
                dense=self._dense_retriever,
                sparse=self._sparse_retriever,
            )
        # Invalidate reranker cache — see docstring above.
        self._reranker_cache.clear()
        logger.info(
            f"{StatusEmoji.INFO} Retrievers ready: "
            f"dense={self._dense_retriever is not None}, "
            f"sparse={self._sparse_retriever is not None}, "
            f"hybrid={self._hybrid_retriever is not None} "
            f"(reranker cache cleared)"
        )

    def _select_retriever(
        self,
        strategy: Optional[str],
        *,
        rerank: bool = False,
        rerank_fetch_k: int = 20,
    ) -> Optional[BaseRetriever]:
        """Strateji adından retriever'ı çöz; rerank=True ise üstüne reranker sar.

        Default strategy: hybrid (varsa), yoksa dense.
        Reranker isteğe bağlı — lazy load.
        """
        if strategy == "dense":
            base = self._dense_retriever
        elif strategy == "sparse":
            base = self._sparse_retriever
        elif strategy == "hybrid":
            base = self._hybrid_retriever
        else:
            # auto / None — production default
            base = self._hybrid_retriever or self._dense_retriever

        if not rerank or base is None:
            return base

        # Cache by base retriever name so we don't reload the cross-encoder
        cache_key = base.name
        if cache_key not in self._reranker_cache:
            self._reranker_cache[cache_key] = RerankedRetriever(
                base, fetch_k=rerank_fetch_k
            )
        return self._reranker_cache[cache_key]

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        strategy: Optional[str] = None,
        rerank: bool = False,
        rerank_fetch_k: int = 20,
        context_chain: Optional[ProcessorChain] = None,
    ) -> List[Document]:
        """
        Search for relevant documents using the chosen retrieval strategy.

        Args:
            query: Search query
            k: Number of documents to return
            strategy: 'dense' | 'sparse' | 'hybrid' | None (auto)
            rerank: Apply cross-encoder rerank on top of the strategy
            rerank_fetch_k: How many candidates to feed the reranker
            context_chain: Optional context-engineering processors applied
                           AFTER retrieval (dedup / token budget / reorder)

        Returns:
            List of relevant Document objects
        """
        retriever = self._select_retriever(
            strategy, rerank=rerank, rerank_fetch_k=rerank_fetch_k,
        )
        if retriever is None:
            # Legacy fallback path — no context processors applied
            if not self.vector_store:
                return []
            try:
                return self.vector_store.similarity_search(query, k=k)
            except Exception as e:
                logger.error(
                    f"{StatusEmoji.ERROR} Stale vector_store ({type(e).__name__}: {e}). "
                    "Reindex may be required."
                )
                # Tell ask() the index is gone by raising a recognisable error
                raise RuntimeError("STALE_INDEX") from e

        try:
            retrieved = retriever.retrieve(query, k=k)
        except Exception as e:
            err = str(e)
            # ChromaDB throws "Collection [UUID] does not exist" when the
            # vector_store reference points at a deleted collection. Most
            # commonly happens if reindex ran in a way that left a stale
            # cached retriever. After the fix in _build_retrievers this
            # shouldn't happen, but be defensive.
            if "does not exist" in err or "Collection" in err:
                logger.error(
                    f"{StatusEmoji.ERROR} Stale ChromaDB collection — "
                    "drop reranker cache and surface a clean error."
                )
                self._reranker_cache.clear()
                raise RuntimeError("STALE_INDEX") from e
            raise

        # Apply context engineering processors (if configured) on the
        # RetrievedDoc layer so processors can see retrieval scores.
        if context_chain is not None and retrieved:
            retrieved = context_chain.process(query, retrieved)

        # Adapt RetrievedDoc → langchain Document for the rest of the pipeline
        return [
            Document(page_content=r.page_content, metadata=dict(r.metadata))
            for r in retrieved
        ]
    
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
    
    def _build_memory(
        self,
        strategy: Optional[str],
        llm_for_summary: Optional[BaseLLMProvider] = None,
    ) -> BaseMemory:
        """Strateji adından memory instance üret."""
        if strategy in (None, "", "none"):
            return NoMemory()
        if strategy == "sliding_window":
            return SlidingWindowMemory(window_size=5)
        if strategy == "summary_buffer":
            return SummaryBufferMemory(llm=llm_for_summary or self.llm_provider)
        if strategy == "vector":
            return VectorRetrievalMemory(
                embed_fn=self.embedding_manager.embed_query, top_k=3
            )
        return NoMemory()

    def _fuse_retrievals(self, queries, k, retrieve_fn):
        """Multi-query RRF: her query'den retrieve et, rank-based birleştir.

        K_RRF=60 (Microsoft paper'daki ampirik default). Sonuçların kimliği
        (source + page_content hash) üzerinden dedup yapılır.
        """
        K_RRF = 60
        scores: dict = {}
        docs_by_id: dict = {}
        for q in queries:
            ds = retrieve_fn(q, k * 2)  # over-sample
            for rank, doc in enumerate(ds):
                doc_id = (
                    doc.metadata.get("source", "?"),
                    doc.metadata.get("item_index", -1),
                    hash(doc.page_content[:200]) & 0xFFFFFFFF,
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (K_RRF + rank + 1)
                docs_by_id.setdefault(doc_id, doc)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return [docs_by_id[did] for did, _ in ranked[:k]]

    def _resolve_prompt_strategy(
        self,
        name: Optional[str],
        *,
        custom_role: Optional[str] = None,
        custom_prompt_template: Optional[str] = None,
    ) -> BasePromptStrategy:
        """Strategy adından instance üret.

        Special cases:
            - 'role_based' → user-provided role is injected
            - 'custom'     → user-provided template is injected
            - None / unknown → DirectStrategy (safe default)
        """
        key = (name or "direct").strip().lower()
        if key == "role_based":
            return RoleBasedStrategy(role=custom_role)
        if key == "custom":
            return CustomStrategy(template=custom_prompt_template)
        if PromptStrategyFactory.is_available(key):
            return PromptStrategyFactory.create(key)
        # Unknown / mistyped strategy → fall back rather than 500
        return DirectStrategy()

    def _build_context_chain(
        self,
        *,
        deduplicate: bool = False,
        reorder: bool = False,
        max_context_tokens: Optional[int] = None,
        dedup_threshold: float = 0.92,
    ) -> Optional[ProcessorChain]:
        """Hangi context processor'ların aktif olduğuna göre zincir kur.

        Sıra önemli — docstring src.context.base.ProcessorChain'de:
            redundancy → token budget → reorderer
        """
        processors: list = []
        if deduplicate:
            processors.append(RedundancyFilter(
                embed_fn=self.embedding_manager.embed_query,
                similarity_threshold=dedup_threshold,
            ))
        if max_context_tokens is not None:
            processors.append(TokenBudgetTrimmer(max_tokens=max_context_tokens))
        if reorder:
            processors.append(LostInTheMiddleReorderer())
        if not processors:
            return None
        return ProcessorChain(processors)

    def ask(
        self,
        question: str,
        k: int = 5,
        *,
        llm_provider: Optional[BaseLLMProvider] = None,
        llm_params: Optional[Dict[str, Any]] = None,
        retrieval_strategy: Optional[str] = None,
        rerank: bool = False,
        rerank_fetch_k: int = 20,
        history: Optional[List[ConversationTurn]] = None,
        memory_strategy: Optional[str] = None,
        deduplicate_context: bool = False,
        reorder_context: bool = False,
        max_context_tokens: Optional[int] = None,
        allow_general_knowledge_fallback: bool = False,
        prompt_strategy: Optional[str] = None,
        custom_role: Optional[str] = None,
        custom_prompt_template: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            # Build context-engineering chain (None if no flags set)
            context_chain = self._build_context_chain(
                deduplicate=deduplicate_context,
                reorder=reorder_context,
                max_context_tokens=max_context_tokens,
            )
            context_label_parts = []
            if deduplicate_context:
                context_label_parts.append("dedup")
            if max_context_tokens is not None:
                context_label_parts.append(f"budget={max_context_tokens}")
            if reorder_context:
                context_label_parts.append("reorder")
            context_label = "+ctx[" + ",".join(context_label_parts) + "]" if context_label_parts else ""

            # Resolve prompt strategy + build the strategy execution
            # context that multi-step strategies will need.
            strategy = self._resolve_prompt_strategy(
                prompt_strategy,
                custom_role=custom_role,
                custom_prompt_template=custom_prompt_template,
            )
            provider = llm_provider or self.llm_provider

            def _retrieve(q: str, kk: int):
                return self.search(
                    q, k=kk, strategy=retrieval_strategy,
                    rerank=rerank, rerank_fetch_k=rerank_fetch_k,
                    context_chain=context_chain,
                )

            strategy_ctx = StrategyContext(
                llm=provider,
                retrieve_fn=_retrieve,
                embed_fn=self.embedding_manager.embed_query,
                llm_params=dict(llm_params or {}),
            )

            # Search for relevant documents — multi-query strategies expand
            # the original question into N variants, retrieve per variant
            # and fuse with RRF; single-query strategies just retrieve once.
            strategy_label = (
                (retrieval_strategy or "auto")
                + ("+rerank" if rerank else "")
                + context_label
                + f"+prompt[{strategy.name}]"
            )
            logger.info(
                f"{StatusEmoji.SEARCH} Searching ({strategy_label}) for: '{question}'..."
            )

            if strategy.is_multi_query:
                variants = strategy.generate_query_variations(question, strategy_ctx)
                logger.info(
                    f"{StatusEmoji.INFO} Multi-query expanded to {len(variants)} variants"
                )
                relevant_docs = self._fuse_retrievals(variants, k, _retrieve)
            else:
                relevant_docs = _retrieve(question, k)
            
            # Calculate relevance score
            relevance_score = self.calculate_relevance_score(question, relevant_docs)
            logger.info(f"{StatusEmoji.INFO} Relevance score: {relevance_score:.3f}")
            
            # Check if we have sufficient context
            if not relevant_docs or relevance_score < self.RELEVANCE_THRESHOLD:
                logger.info(f"{StatusEmoji.INFO} Insufficient context, using fallback...")
                return self._fallback_response(
                    question, relevance_score,
                    llm_provider=llm_provider, llm_params=llm_params,
                    allow_general_knowledge=allow_general_knowledge_fallback,
                )
            
            # Build context from documents
            context_parts = []
            for i, doc in enumerate(relevant_docs, 1):
                source = doc.metadata.get("source", "Unknown")
                context_parts.append(f"[Source {i} - {source}]\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)

            # Apply memory strategy (returns "" if NoMemory or empty history)
            memory = self._build_memory(memory_strategy, llm_for_summary=llm_provider)
            memory_context = memory.apply(history or [], question).strip()

            provider_label = getattr(provider, "model", self.model_type)
            logger.info(
                f"{StatusEmoji.ROBOT} Generating ({provider_label}) "
                f"via strategy={strategy.name}..."
            )

            # Strategy owns the prompt construction AND any extra LLM calls
            # (CoT extracts the YANIT block; multi-step ones can override).
            answer = strategy.execute(
                strategy_ctx,
                question=question,
                context=context,
                memory_context=memory_context,
            )
            
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
                "relevance_score": relevance_score,
                "retrieval_strategy": strategy_label,
                "memory_strategy": memory_strategy or "none",
                "memory_used": bool(memory_context),
                "prompt_strategy": strategy.name,
            }
            
        except Exception as e:
            if isinstance(e, RuntimeError) and str(e) == "STALE_INDEX":
                # Friendly Turkish message — UI surfaces this directly
                logger.error(f"{StatusEmoji.ERROR} Stale index detected")
                return {
                    "question": question,
                    "answer": (
                        "Vektör tabanı eski bir koleksiyona referans veriyor "
                        "(muhtemelen yeniden indeksleme sırasında oluştu). "
                        "Lütfen 'Dosyaları Yönet' sekmesinden "
                        "**Bilgi Tabanını Yeniden İndeksle** butonuna basın "
                        "ve aramayı tekrarlayın."
                    ),
                    "source_documents": [],
                    "context_used": "",
                    "source": "stale_index",
                }
            logger.error(f"{StatusEmoji.ERROR} Error: {e}")
            return {
                "question": question,
                "answer": f"{StatusEmoji.ERROR} Error: {str(e)}",
                "source_documents": [],
                "context_used": "",
                "source": "error"
            }
    
    def _fallback_response(
        self,
        question: str,
        relevance_score: float,
        *,
        llm_provider: Optional[BaseLLMProvider] = None,
        llm_params: Optional[Dict[str, Any]] = None,
        allow_general_knowledge: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a fallback response when RAG context is insufficient.

        SAFETY DEFAULT: ``allow_general_knowledge=False``.

        When the retrieval finds nothing relevant, the previous behaviour
        was to let the cloud LLM produce a "general knowledge" answer.
        That answer is unconstrained and the LLM happily HALLUCINATES
        proper-noun content — a query about a private person whose CV was
        just uploaded but indexed badly would come back as a fabricated
        biography of someone else entirely. For personal / proprietary
        data this is unacceptable.

        With the default (False) we now return a clean Turkish "no info"
        response and tell the user how to fix it (upload + reindex,
        enable reranker, etc.). Power users can opt in via the header
        X-Allow-General-Knowledge: true when they trust the LLM's
        general knowledge for the topic of their knowledge base.

        Args:
            question: The user's question
            relevance_score: The relevance score that triggered fallback
            llm_provider: Optional per-request provider override
            llm_params: Optional per-request LLM param overrides
            allow_general_knowledge: Opt-in to let the LLM answer from
                its own training data (HALLUCINATION RISK for
                personal/proprietary content).

        Returns:
            Response dictionary
        """
        provider = llm_provider or self.llm_provider
        is_cloud = not isinstance(provider, LocalProvider)

        # Safe default: no LLM call, no hallucination
        if not allow_general_knowledge:
            data_summary = self._get_data_summary()
            return {
                "question": question,
                "answer": (
                    "Bu konuda bilgi tabanında yeterli detay bulunamadı.\n\n"
                    "Olası nedenler:\n"
                    "• İlgili dosya henüz yüklenmedi veya yüklendiyse "
                    "'Bilgi Tabanını Yeniden İndeksle' butonuna basılmadı.\n"
                    "• PDF dosyası taranmış (görsel) olabilir — pypdf "
                    "metin katmanı olmayan PDF'lerden çıkarım yapamaz.\n"
                    "• Soru, mevcut belgelerden ayrı bir konuda olabilir.\n\n"
                    f"Mevcut konular: {data_summary}\n\n"
                    "İpucu: Ayarlardan **Cross-encoder reranker**'ı açmak "
                    "alakalı belgeleri yukarı çıkarabilir; isterseniz "
                    "**Ayarlar → 'Genel bilgi fallback'i'** ile LLM'in "
                    "kendi bilgisinden cevap üretmesine izin verebilirsiniz "
                    "(halüsinasyon riski içerir)."
                ),
                "source_documents": [],
                "context_used": "",
                "source": "insufficient_data",
                "relevance_score": relevance_score,
            }

        # Opt-in branch: user explicitly accepted the hallucination risk
        if is_cloud:
            general_answer = provider.generate_general(question, **(llm_params or {}))
            data_summary = self._get_data_summary()
            final_answer = (
                "⚠️ Bilgi tabanında yeterli detay bulunamadı; aşağıdaki cevap "
                f"{self.model_type} modelinin GENEL bilgisinden üretildi "
                "(halüsinasyon olasılığı yüksek — özellikle özel isim, "
                "kuruluş veya proprietary bilgi içeren sorularda):\n\n"
                f"{general_answer}\n\n"
                f"---\n📚 Mevcut konular: {data_summary}"
            )
            return {
                "question": question,
                "answer": final_answer,
                "source_documents": [],
                "context_used": "",
                "source": "general_knowledge_fallback",
                "relevance_score": relevance_score,
            }
        return {
            "question": question,
            "answer": (
                f"Bilgi tabanında yeterli detay bulunamadı (relevance score: "
                f"{relevance_score:.3f}). Lütfen ilgili belgeleri yükleyin "
                "veya soru ifadenizi değiştirin."
            ),
            "source_documents": [],
            "context_used": "",
            "source": "insufficient_data",
            "relevance_score": relevance_score,
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
