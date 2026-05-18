"""TextLoader — düz UTF-8 metin.

En basit loader: dosyayı oku, tek Document yap. Encoding sorununda
fallback'le latin-1 dener (Türkçe dahil çoğu eski sistemde çalışır).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.loaders.base import BaseLoader


class TextLoader(BaseLoader):
    name = "text"
    extensions = {".txt"}

    MIN_CONTENT = 10

    def load(self, file_path: Path) -> List[Document]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback: latin-1 hiçbir zaman fail etmez
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()

        content = content.strip()
        if len(content) < self.MIN_CONTENT:
            return []

        return [Document(
            page_content=content,
            metadata={
                "source": file_path.name,
                "format": "txt",
                "file_path": str(file_path),
                "item_index": 0,
            },
        )]
