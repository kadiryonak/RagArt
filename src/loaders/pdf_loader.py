"""PDFLoader — pypdf ile sayfa-bazlı extract.

Her sayfa ayrı bir Document olur (item_index = sayfa numarası).
Bu sayede retrieval doğru sayfayı geri çekebilir; çok büyük PDF'lerde
chunking sayfa içinde olur, sayfalar arası bağlam karışmaz.

Sınırlar
    - Taranmış PDF (görüntü-bazlı) için OCR gerekir; pypdf sadece
      text-layer'ı okur. Boş sayfa → atlanır.
    - Tablo, çoklu sütun gibi karmaşık layout'larda sıralama bozulabilir
      (pypdf'in bilinen sınırı). v1 için bu yeterli; v2'de pdfplumber.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.loaders.base import BaseLoader


class PDFLoader(BaseLoader):
    name = "pdf"
    extensions = {".pdf"}

    MIN_PAGE_TEXT = 20  # bu kadar karakterden az → boş sayfa say

    def load(self, file_path: Path) -> List[Document]:
        # Lazy import: pypdf kurulu değilse, sadece bu loader fail eder
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError(
                "pypdf is required for PDF loading. "
                "Install with: pip install pypdf"
            ) from e

        reader = PdfReader(str(file_path))
        documents: List[Document] = []
        total = len(reader.pages)

        for idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                # Bozuk sayfa — atla, gerisi devam
                continue

            text = text.strip()
            if len(text) < self.MIN_PAGE_TEXT:
                continue

            documents.append(Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "format": "pdf",
                    "file_path": str(file_path),
                    "item_index": idx,
                    "item_count": total,
                    "page": idx + 1,  # 1-tabanlı insan-okur
                },
            ))
        return documents
