"""MarkdownLoader — UTF-8 metin, hafif markdown sanitization.

Markdown syntax karakterleri (# * _ ` []()) embedding kalitesini düşürebilir
ama içerik için kritik. Biz sadece kod block sınırlayıcılarını kaldırıyoruz;
gerisini olduğu gibi bırakıyoruz çünkü Türkçe içerik beklenmiyor.

Heading-bazlı bölme YOK (v1): text splitter zaten chunk'lar. Heading-aware
splitter ileride parent-child retrieval için gerekecek.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.loaders.base import BaseLoader


class MarkdownLoader(BaseLoader):
    name = "markdown"
    extensions = {".md", ".markdown"}

    MIN_CONTENT = 20

    def _sanitize(self, text: str) -> str:
        # Code fences (``` ... ```) — block içeriği koru, fence'leri at
        text = re.sub(r"^```[a-zA-Z]*\n", "", text, flags=re.MULTILINE)
        text = re.sub(r"^```\s*$", "", text, flags=re.MULTILINE)
        # Frontmatter (--- ... ---) — atla
        text = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL)
        return text.strip()

    def load(self, file_path: Path) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
        content = self._sanitize(raw)
        if len(content) < self.MIN_CONTENT:
            return []
        return [Document(
            page_content=content,
            metadata={
                "source": file_path.name,
                "format": "md",
                "file_path": str(file_path),
                "item_index": 0,
            },
        )]
