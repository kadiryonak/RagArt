"""
Embedding management for the RAG system.
"""

from typing import List, Optional

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain_core.schema import Document

from src.utils import get_logger, StatusEmoji

logger = get_logger(__name__)


# Default model optimized for multilingual (including Turkish) embeddings
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ── Per-format chunking ────────────────────────────────────────────────
# Different document formats have different natural structure, so one-size
# chunking is suboptimal. Each entry tunes the recursive splitter for that
# format: `scale` multiplies the manager's base chunk_size, and `separators`
# are tried in order (earlier = preferred split point). Formats not listed
# (or documents without a `format` tag) fall back to the manager default,
# so behaviour is unchanged for anything we don't explicitly optimize.
#
#   prose      → sentence-aware ("\n\n" > "\n" > ". " > ...) for pdf/docx/txt
#   markdown   → split on headings first so a section stays whole
#   json       → larger chunks so a single record/entry isn't fragmented
_PROSE_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
_MARKDOWN_SEPARATORS = [
    "\n## ", "\n### ", "\n#### ", "\n# ", "\n\n", "\n", ". ", " ", "",
]
_JSON_SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]

_FORMAT_SPLIT = {
    "pdf":      {"scale": 1.0, "separators": _PROSE_SEPARATORS},
    "docx":     {"scale": 1.0, "separators": _PROSE_SEPARATORS},
    "txt":      {"scale": 1.0, "separators": _PROSE_SEPARATORS},
    "md":       {"scale": 1.25, "separators": _MARKDOWN_SEPARATORS},
    "markdown": {"scale": 1.25, "separators": _MARKDOWN_SEPARATORS},
    "json":     {"scale": 1.5, "separators": _JSON_SEPARATORS},
}


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
        chunk_overlap: int = 150,
        split_strategy: str = "recursive",
    ):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to run the model on ('cpu' or 'cuda')
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            split_strategy: 'recursive' (default) or 'semantic'
        """
        self.model_name = model_name
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_strategy = split_strategy
        
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        # Cache of per-format splitters, lazily built on first use.
        self._format_splitters: dict = {}
    
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
    
    def _splitter_for(self, fmt: Optional[str]) -> RecursiveCharacterTextSplitter:
        """Return a recursive splitter tuned for the given document format.

        Falls back to the manager's default splitter for unknown/missing
        formats, so documents without a `format` tag behave exactly as before.
        """
        key = (fmt or "").lower()
        cfg = _FORMAT_SPLIT.get(key)
        if cfg is None:
            return self.text_splitter
        if key not in self._format_splitters:
            size = max(1, int(self.chunk_size * cfg.get("scale", 1.0)))
            self._format_splitters[key] = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=cfg["separators"],
            )
        return self._format_splitters[key]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Uses semantic splitting when split_strategy='semantic'. Otherwise uses
        recursive character splitting with FORMAT-AWARE settings: each document
        is chunked by a splitter tuned for its `metadata["format"]` (prose for
        pdf/docx/txt, heading-aware for markdown, larger chunks for json).
        Documents are processed in order so the chunk sequence is stable.

        Args:
            documents: List of Document objects

        Returns:
            List of split Document objects
        """
        logger.info(
            f"{StatusEmoji.LOADING} Splitting {len(documents)} documents "
            f"(strategy={self.split_strategy})..."
        )

        if self.split_strategy == "semantic":
            splitter = SemanticTextSplitter(
                embed_fn=self.embed_query,
                max_chunk_size=self.chunk_size,
            )
            split_docs = splitter.split_documents(documents)
        else:
            split_docs = []
            for doc in documents:
                splitter = self._splitter_for(doc.metadata.get("format"))
                split_docs.extend(splitter.split_documents([doc]))

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


# ── Semantic Text Splitter ────────────────────────────────────────────

import re as _re


def _split_sentences(text: str) -> List[str]:
    """Türkçe-uyumlu cümle bölücü."""
    # Split on sentence-ending punctuation followed by space/newline
    parts = _re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if s.strip()]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticTextSplitter:
    """Embedding similarity tabanlı metin bölücü.

    NASIL ÇALIŞIR:
        1. Metin → cümlelere böl
        2. Ardışık cümlelerin embedding similarity'sini hesapla
        3. Similarity threshold altına düştüğünde yeni chunk başlat
        4. max_chunk_size aşılırsa da yeni chunk başlat

    AVANTAJ:
        - Anlamsal bütünlüğü korur (ilgili cümleler aynı chunk'ta kalır)
        - RecursiveCharacterTextSplitter'ın karakter-tabanlı bölmesinden
          daha akıllı

    MALİYET:
        - Her cümle embed edilir → ilk indeksleme yavaşlar
        - Cache varsa sonraki çağrılar hızlı
    """

    DEFAULT_THRESHOLD = 0.5

    def __init__(
        self,
        embed_fn,
        max_chunk_size: int = 800,
        similarity_threshold: float = DEFAULT_THRESHOLD,
    ):
        self._embed = embed_fn
        self.max_chunk_size = max_chunk_size
        self.threshold = similarity_threshold

    def split_text(self, text: str) -> List[str]:
        """Metni semantic chunk'lara böl."""
        sentences = _split_sentences(text)
        if not sentences:
            return [text] if text.strip() else []

        if len(sentences) <= 1:
            return sentences

        # Embed all sentences
        try:
            embeddings = [self._embed(s) for s in sentences]
        except Exception:
            # Embedding failed → fallback to single chunk
            return [text]

        chunks: List[str] = []
        current_chunk: List[str] = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            sim = _cosine_sim(embeddings[i - 1], embeddings[i])
            sent_len = len(sentences[i])

            # Break conditions: low similarity OR chunk too large
            if sim < self.threshold or (current_len + sent_len) > self.max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
                current_len = sent_len
            else:
                current_chunk.append(sentences[i])
                current_len += sent_len

        # Last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Document listesini semantic chunk'lara böl."""
        result: List[Document] = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                meta = dict(doc.metadata)
                meta["chunk_index"] = i
                meta["chunk_count"] = len(chunks)
                meta["split_strategy"] = "semantic"
                result.append(Document(page_content=chunk, metadata=meta))
        return result

