"""
Document loading and text extraction utilities.

Multi-format support via src.loaders LoaderRegistry. The class is still
named JSONDocumentLoader for backward compatibility with rag_system.py,
but it loads ANY supported format (json/pdf/docx/md/txt).
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from src.loaders import get_default_registry
from src.utils import get_logger, StatusEmoji

logger = get_logger(__name__)


class JSONDocumentLoader:
    """Loads documents from a folder (multi-format via LoaderRegistry).

    Despite the legacy name, this loader handles json/pdf/docx/md/txt
    based on the registered loaders. The folder is scanned once per
    load_all() call; each file is dispatched to the matching loader.
    """

    def __init__(self, data_folder: str, registry=None):
        """
        Args:
            data_folder: Path to the folder containing source files
            registry: LoaderRegistry instance (test injection); default
                      is the standard registry with all built-in loaders.
        """
        self.data_folder = Path(data_folder)
        self._documents: List[Document] = []
        self._registry = registry or get_default_registry()

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def supported_extensions(self) -> set[str]:
        return self._registry.supported_extensions

    def load_all(self) -> List[Document]:
        """Load every supported file in the data folder."""
        self._documents = []

        if not self.data_folder.exists():
            logger.warning(f"{StatusEmoji.WARNING} Data folder not found: {self.data_folder}")
            return self._documents

        files: List[Path] = []
        for ext in self.supported_extensions:
            files.extend(self.data_folder.glob(f"*{ext}"))
        # Deterministic ordering — useful for reproducible eval baselines
        files.sort(key=lambda p: p.name)

        logger.info(
            f"{StatusEmoji.FOLDER} Found {len(files)} files "
            f"({', '.join(sorted(self.supported_extensions))})"
        )

        for file_path in files:
            try:
                documents = self._registry.load(file_path)
                self._documents.extend(documents)
            except Exception as e:
                logger.error(f"{StatusEmoji.ERROR} Error loading {file_path.name}: {e}")

        logger.info(f"{StatusEmoji.SUCCESS} Loaded {len(self._documents)} documents")
        return self._documents
    
    def get_file_info(self) -> List[Dict[str, Any]]:
        """List all supported files in the data folder with quick metadata.

        Does NOT load the documents — used for UI listing where we only
        need filename + size + per-file document estimate. PDF page count
        is exact (cheap to read); other formats fall back to 1.
        """
        file_info = []

        if not self.data_folder.exists():
            return file_info

        for ext in sorted(self.supported_extensions):
            for file_path in sorted(self.data_folder.glob(f"*{ext}")):
                try:
                    size_kb = round(file_path.stat().st_size / 1024, 2)
                    file_info.append({
                        "filename": file_path.name,
                        "format": ext.lstrip("."),
                        "document_count": _estimate_doc_count(file_path),
                        "size_kb": size_kb,
                    })
                except Exception as e:
                    file_info.append({
                        "filename": file_path.name,
                        "format": ext.lstrip("."),
                        "error": str(e),
                    })

        return file_info


def _estimate_doc_count(file_path: Path) -> int:
    """Quick file-type-specific doc count (no full load)."""
    ext = file_path.suffix.lower()
    if ext == ".json":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return len(data) if isinstance(data, list) else 1
        except Exception:
            return 0
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            return len(PdfReader(str(file_path)).pages)
        except Exception:
            return 1
    # docx / md / txt → 1 Document per file
    return 1


def create_sample_data(data_folder: str) -> None:
    """
    Create sample knowledge base data for testing.
    
    Args:
        data_folder: Path to the data folder
    """
    sample_data = [
        {
            "id": 1,
            "topic": "Algorithm",
            "definition": "Algoritma, belirli bir problemi çözmek için tasarlanmış adımların sıralı bir listesidir.",
            "details": "Algoritma, bilgisayar biliminin temel taşlarından biridir. Bir problemi çözmek için izlenmesi gereken adımları tanımlar.",
            "examples": ["Sorting algorithms", "Search algorithms", "Graph algorithms"],
            "category": "Computer Science"
        },
        {
            "id": 2,
            "topic": "Data Structures",
            "definition": "Veri yapıları, bilgisayar belleğinde verileri organize etme ve depolama yöntemleridir.",
            "details": "Veri yapıları, verilerin etkili bir şekilde depolanması ve erişilmesi için kullanılan düzenlemelerdir.",
            "examples": ["Array", "Linked List", "Stack", "Queue", "Tree", "Graph"],
            "category": "Computer Science"
        },
        {
            "id": 3,
            "topic": "Artificial Intelligence",
            "definition": "Yapay zeka, makinelerin insan benzeri zeka gerektiren görevleri yerine getirebilme yeteneğidir.",
            "details": "Yapay zeka, machine learning, deep learning, natural language processing ve computer vision gibi alt dalları içerir.",
            "examples": ["Machine Learning", "Deep Learning", "NLP", "Computer Vision"],
            "category": "Technology"
        },
        {
            "id": 4,
            "topic": "Python Programming",
            "definition": "Python, kolay öğrenilen ve güçlü bir yüksek seviye programlama dilidir.",
            "details": "Python, temiz sözdizimi ve geniş kütüphane desteği ile veri bilimi, web geliştirme ve yapay zeka projelerinde yaygın olarak kullanılır.",
            "examples": ["Django", "Flask", "Pandas", "NumPy", "Scikit-learn"],
            "category": "Programming"
        },
        {
            "id": 5,
            "topic": "Web Development",
            "definition": "Web geliştirme, internet siteleri ve web uygulamaları oluşturma sürecidir.",
            "details": "Web geliştirme frontend ve backend olmak üzere iki ana bölümden oluşur.",
            "examples": ["React", "Vue.js", "Node.js", "Python Flask", "MySQL"],
            "category": "Technology"
        }
    ]
    
    folder = Path(data_folder)
    folder.mkdir(parents=True, exist_ok=True)
    
    output_file = folder / "sample_knowledge.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"{StatusEmoji.SUCCESS} Created sample data file: {output_file}")
