"""
Document loading and text extraction utilities.
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

from src.utils import get_logger, StatusEmoji

logger = get_logger(__name__)


class JSONDocumentLoader:
    """
    Loads and processes JSON documents for the RAG system.
    
    This class handles:
    - Loading JSON files from a directory
    - Extracting text content from nested JSON structures
    - Creating LangChain Document objects with metadata
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the document loader.
        
        Args:
            data_folder: Path to the folder containing JSON files
        """
        self.data_folder = Path(data_folder)
        self._documents: List[Document] = []
    
    @property
    def document_count(self) -> int:
        """Return the number of loaded documents."""
        return len(self._documents)
    
    def load_all(self) -> List[Document]:
        """
        Load all JSON files from the data folder.
        
        Returns:
            List of Document objects
        """
        self._documents = []
        
        if not self.data_folder.exists():
            logger.warning(f"{StatusEmoji.WARNING} Data folder not found: {self.data_folder}")
            return self._documents
        
        json_files = list(self.data_folder.glob("*.json"))
        logger.info(f"{StatusEmoji.FOLDER} Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                documents = self._load_file(json_file)
                self._documents.extend(documents)
            except Exception as e:
                logger.error(f"{StatusEmoji.ERROR} Error loading {json_file.name}: {e}")
        
        logger.info(f"{StatusEmoji.SUCCESS} Loaded {len(self._documents)} documents")
        return self._documents
    
    def _load_file(self, file_path: Path) -> List[Document]:
        """
        Load a single JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of Document objects from this file
        """
        documents = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for idx, item in enumerate(data):
                content = self._extract_text(item)
                if content and len(content.strip()) > 10:
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file_path.name,
                            "item_index": idx,
                            "file_path": str(file_path),
                            "item_count": len(data)
                        }
                    )
                    documents.append(doc)
        else:
            content = self._extract_text(data)
            if content and len(content.strip()) > 10:
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path.name,
                        "file_path": str(file_path)
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _extract_text(self, json_obj: Any, parent_key: str = "") -> str:
        """
        Extract text content from a JSON object recursively.
        
        Args:
            json_obj: JSON object (dict, list, or primitive)
            parent_key: Parent key for context
            
        Returns:
            Extracted text content
        """
        text_parts = []
        
        def clean_text(text: str) -> str:
            """Clean extracted text."""
            text = re.sub(r"[=]{2,}", "", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            return text.strip()
        
        def extract_recursive(obj: Any, current_key: str = "") -> None:
            """Recursively extract text from nested structures."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and len(value.strip()) > 2:
                        clean_value = clean_text(value)
                        if clean_value:
                            text_parts.append(f"{key}: {clean_value}")
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value, key)
                        
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str) and len(item.strip()) > 2:
                        clean_item = clean_text(item)
                        if clean_item:
                            text_parts.append(clean_item)
                    elif isinstance(item, (dict, list)):
                        extract_recursive(item, current_key)
        
        extract_recursive(json_obj, parent_key)
        
        # Remove duplicates while preserving order
        unique_parts = []
        for part in text_parts:
            if part not in unique_parts and len(part.strip()) > 5:
                unique_parts.append(part)
        
        return "\n".join(unique_parts)
    
    def get_file_info(self) -> List[Dict[str, Any]]:
        """
        Get information about loaded files.
        
        Returns:
            List of file information dictionaries
        """
        file_info = []
        
        if not self.data_folder.exists():
            return file_info
        
        for json_file in self.data_folder.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                doc_count = len(data) if isinstance(data, list) else 1
                
                file_info.append({
                    "filename": json_file.name,
                    "document_count": doc_count,
                    "size_kb": round(json_file.stat().st_size / 1024, 2)
                })
            except Exception as e:
                file_info.append({
                    "filename": json_file.name,
                    "error": str(e)
                })
        
        return file_info


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
