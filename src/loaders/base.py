"""BaseLoader sözleşmesi + LoaderRegistry (uzantı → loader dispatcher).

Her loader bir dosya yolu alır, LangChain Document'leri listesi döndürür.
Metadata standart alanları:
    source       : dosya adı (chunk'ların kaynağını izlemek için)
    format       : 'json' | 'pdf' | 'docx' | 'md' | 'txt'
    file_path    : tam yol (debug için)
    item_index   : multi-item dosyalarda (JSON list, PDF sayfaları)
    item_count   : opsiyonel — toplam item sayısı
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Set

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


class BaseLoader(ABC):
    """Tüm loader'ların türetildiği temel sınıf."""

    name: str = "base"
    extensions: Set[str] = set()  # alt sınıf doldurur

    @abstractmethod
    def load(self, file_path: Path) -> List[Document]:
        """Dosyayı oku, Document listesi döndür.

        Hata yönetimi: dosya okunamazsa veya boşsa boş liste döndürülmeli;
        bu metot exception fırlatabilir ama caller (registry) yakalar.
        """

    def supports(self, file_path: Path) -> bool:
        """Bu loader bu dosyayı işleyebilir mi? Default: uzantı eşleşmesi."""
        return file_path.suffix.lower() in self.extensions


class LoaderRegistry:
    """Uzantıya göre loader seçen dispatcher."""

    def __init__(self) -> None:
        self._loaders: List[BaseLoader] = []

    def register(self, loader: BaseLoader) -> None:
        self._loaders.append(loader)

    def get_loader(self, file_path: Path) -> BaseLoader | None:
        """Bu dosyayı işleyebilen ilk loader'ı bul."""
        for loader in self._loaders:
            if loader.supports(file_path):
                return loader
        return None

    @property
    def supported_extensions(self) -> Set[str]:
        out: Set[str] = set()
        for loader in self._loaders:
            out |= loader.extensions
        return out

    def load(self, file_path: Path) -> List[Document]:
        """Doğru loader'ı seç + çağır. Loader yoksa boş döner."""
        loader = self.get_loader(file_path)
        if loader is None:
            return []
        return loader.load(file_path)


def get_default_registry() -> LoaderRegistry:
    """Tüm bilinen loader'larla dolu LoaderRegistry."""
    # Lazy import: dairesel referans önlemek için
    from src.loaders.json_loader import JSONLoader
    from src.loaders.pdf_loader import PDFLoader
    from src.loaders.docx_loader import DOCXLoader
    from src.loaders.markdown_loader import MarkdownLoader
    from src.loaders.text_loader import TextLoader

    r = LoaderRegistry()
    # Kayıt sırası önemli değil çünkü uzantılar disjunkt
    r.register(JSONLoader())
    r.register(PDFLoader())
    r.register(DOCXLoader())
    r.register(MarkdownLoader())
    r.register(TextLoader())
    return r
