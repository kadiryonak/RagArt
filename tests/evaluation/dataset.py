"""Golden test dataset loader.

Bir GoldenItem şunları içerir:
    - question:         sorulacak soru
    - reference_answer: ideal cevap (insan tarafından yazılmış)
    - keywords:         cevapta bulunması beklenen anahtar terimler (L1 için)
    - expected_sources: bu sorunun cevabını içermesi gereken JSON dosya isimleri
    - difficulty:       easy | medium | hard
    - category:         factual | analytical | synthesis | edge_case
    - critical:         L4 (LLM judge) bu örnekte çalışsın mı (True → kritik)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

Difficulty = Literal["easy", "medium", "hard"]
Category = Literal["factual", "analytical", "synthesis", "edge_case"]


@dataclass
class GoldenItem:
    """Bir test örneği — soru + beklenen cevap + meta veri."""

    id: str
    question: str
    reference_answer: str
    keywords: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    difficulty: Difficulty = "medium"
    category: Category = "factual"
    critical: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenItem":
        return cls(
            id=d["id"],
            question=d["question"],
            reference_answer=d["reference_answer"],
            keywords=d.get("keywords", []),
            expected_sources=d.get("expected_sources", []),
            difficulty=d.get("difficulty", "medium"),
            category=d.get("category", "factual"),
            critical=d.get("critical", False),
        )


def load_golden_dataset(path: Optional[str] = None) -> List[GoldenItem]:
    """Load golden dataset JSON. Default: evaluation/datasets/golden_qa.json."""
    if path is None:
        path = Path(__file__).parent / "datasets" / "golden_qa.json"
    else:
        path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [GoldenItem.from_dict(item) for item in raw]
