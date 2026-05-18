"""Document loader plugins.

Bir bilgi tabanı dosyası → LangChain Document'lerine dönüştürme.
Tüm loader'lar BaseLoader sözleşmesini uygular:

    JSONLoader      .json    yapılandırılmış kayıtlar (Wikipedia-tarzı)
    PDFLoader       .pdf     pypdf ile sayfa-bazlı extract
    DOCXLoader      .docx    python-docx ile paragraph-bazlı
    MarkdownLoader  .md      düz metin + heading-aware splitting
    TextLoader      .txt     UTF-8 düz metin

LoaderRegistry, uzantıya göre doğru loader'ı seçer ve düzgün
metadata ('source', 'format', 'item_index') üretir.
"""

from src.loaders.base import BaseLoader, LoaderRegistry, get_default_registry
from src.loaders.json_loader import JSONLoader
from src.loaders.pdf_loader import PDFLoader
from src.loaders.docx_loader import DOCXLoader
from src.loaders.markdown_loader import MarkdownLoader
from src.loaders.text_loader import TextLoader

__all__ = [
    "BaseLoader",
    "LoaderRegistry",
    "get_default_registry",
    "JSONLoader",
    "PDFLoader",
    "DOCXLoader",
    "MarkdownLoader",
    "TextLoader",
]
