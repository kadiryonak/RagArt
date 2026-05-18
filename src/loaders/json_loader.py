"""JSONLoader — mevcut JSONDocumentLoader davranışını wrapping eder.

Wikipedia-tarzı `{"title": "...", "content": "..."}` veya liste formatı
beklenir. İç içe yapılarda recursive text extraction yapar.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.loaders.base import BaseLoader


def _clean_text(text: str) -> str:
    text = re.sub(r"[=]{2,}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_text(obj: Any) -> str:
    """İç içe JSON yapısından metin akışı çıkarır."""
    parts: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and len(value.strip()) > 2:
                    cleaned = _clean_text(value)
                    if cleaned:
                        parts.append(f"{key}: {cleaned}")
                elif isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, str) and len(item.strip()) > 2:
                    cleaned = _clean_text(item)
                    if cleaned:
                        parts.append(cleaned)
                elif isinstance(item, (dict, list)):
                    walk(item)

    walk(obj)

    # Tekrarları sırasını koruyarak ele
    seen, out = set(), []
    for p in parts:
        if p not in seen and len(p.strip()) > 5:
            seen.add(p)
            out.append(p)
    return "\n".join(out)


class JSONLoader(BaseLoader):
    name = "json"
    extensions = {".json"}

    MIN_CONTENT_LEN = 10

    def load(self, file_path: Path) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents: List[Document] = []
        if isinstance(data, list):
            for idx, item in enumerate(data):
                content = _extract_text(item)
                if content and len(content.strip()) > self.MIN_CONTENT_LEN:
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": file_path.name,
                            "format": "json",
                            "file_path": str(file_path),
                            "item_index": idx,
                            "item_count": len(data),
                        },
                    ))
        else:
            content = _extract_text(data)
            if content and len(content.strip()) > self.MIN_CONTENT_LEN:
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": file_path.name,
                        "format": "json",
                        "file_path": str(file_path),
                        "item_index": 0,
                    },
                ))
        return documents
