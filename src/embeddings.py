"""
Embedding management for the RAG system.
"""

import math
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


# ── Token budget ───────────────────────────────────────────────────────
# Every embedding model has a hard input cap (its `max_seq_length`). For our
# default paraphrase-multilingual-MiniLM-L12-v2 it is 128 wordpiece tokens —
# anything past that is SILENTLY TRUNCATED at embed time, so the tail of an
# oversized chunk never gets embedded and is invisible to retrieval. That is
# why chunk length is measured in TOKENS (via the model's own tokenizer),
# not characters: a char-sized chunk of dense Turkish easily blows past 128
# tokens. See `EmbeddingManager.token_len`.
DEFAULT_MAX_TOKENS = 128
_RESERVED_TOKENS = 2            # [CLS] + [SEP] the model adds around the text
# When the real tokenizer can't be loaded (offline CI, custom model) we
# estimate tokens from characters. Turkish wordpieces are short, so this is
# deliberately conservative (fewer chars/token → over-counts → smaller, safe
# chunks rather than oversized ones).
_FALLBACK_CHARS_PER_TOKEN = 3.0

# Tokenizers are cached per model name so repeated managers/workspaces don't
# reload them. `local_files_only=True` keeps it hermetic — never hits the
# network; if the model isn't already in the HF cache we fall back cleanly.
_TOKENIZER_CACHE: dict = {}


def _load_tokenizer(model_name: str):
    """Best-effort load of `model_name`'s tokenizer from the local HF cache.

    Returns the tokenizer, or None if transformers isn't installed / the
    model isn't cached. Never raises and never touches the network.
    """
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tok = None
    _TOKENIZER_CACHE[model_name] = tok
    return tok


# ── Per-format chunking ────────────────────────────────────────────────
# Different document formats have different natural structure, so one-size
# chunking is suboptimal. Each entry tunes the recursive splitter for that
# format: `scale` multiplies the manager's base chunk_size (in TOKENS), and
# `separators` are tried in order (earlier = preferred split point). The
# scaled size is always capped at the model's token limit, so a "larger"
# format can never produce a chunk that would be truncated at embed time.
# Formats not listed (or documents without a `format` tag) fall back to the
# manager default.
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
        chunk_size: int = 110,
        chunk_overlap: int = 20,
        split_strategy: str = "recursive",
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize the embedding manager.

        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to run the model on ('cpu' or 'cuda')
            chunk_size: Target chunk length in TOKENS (not characters). Capped
                at the model's token limit so no chunk is truncated at embed
                time. Default 110 leaves headroom under the 128-token cap.
            chunk_overlap: Overlap between chunks, in tokens.
            split_strategy: 'recursive' (default) or 'semantic'
            max_tokens: Override the embedding model's token limit. When None,
                uses DEFAULT_MAX_TOKENS (correct for the default MiniLM model);
                set this if you swap in a model with a different max_seq_length.
        """
        self.model_name = model_name
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_strategy = split_strategy
        self.max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS

        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        # Cache of per-format splitters, lazily built on first use.
        self._format_splitters: dict = {}

    # ── Token measurement ──────────────────────────────────────────────
    def token_len(self, text: str) -> int:
        """Length of `text` in the embedding model's tokens.

        Uses the real tokenizer when it's available in the local cache;
        otherwise falls back to a conservative character-based estimate so
        the splitter still keeps chunks safely under the token limit.
        """
        if not text:
            return 0
        tok = _load_tokenizer(self.model_name)
        if tok is not None:
            try:
                ids = tok.encode(text, add_special_tokens=False)
                if isinstance(ids, (list, tuple)):
                    return len(ids)
            except Exception:
                pass
        return max(1, math.ceil(len(text) / _FALLBACK_CHARS_PER_TOKEN))

    def _size_cap(self) -> int:
        """Max chunk size (tokens) that still fits the model after specials."""
        return max(8, self.max_tokens - _RESERVED_TOKENS)
    
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
            # Separators optimized for Turkish text. Length is measured in
            # tokens (token_len) and capped at the model limit so chunks are
            # never silently truncated when embedded.
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=min(self.chunk_size, self._size_cap()),
                chunk_overlap=self.chunk_overlap,
                length_function=self.token_len,
                separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]
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
            # Scale the base token budget for this format, but never exceed the
            # model's token cap — a "larger" json chunk that got truncated at
            # embed time would be worse than a slightly smaller intact one.
            size = max(1, int(self.chunk_size * cfg.get("scale", 1.0)))
            size = min(size, self._size_cap())
            self._format_splitters[key] = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=self.chunk_overlap,
                length_function=self.token_len,
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

