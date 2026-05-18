"""Base contracts for evaluation layers.

Tasarım: Her katman bir BaseEvaluator implementasyonudur. Aynı (GoldenItem, RAGOutput)
çiftini girdi olarak alır, bir LayerResult döndürür. Bu sayede katmanları sırayla
veya seçici şekilde çalıştırabiliriz.

Skor sözleşmesi:
    - score ∈ [0.0, 1.0] (her zaman normalize)
    - passed: skor o katmanın eşiğini geçti mi (boolean)
    - details: katmana özel ekstra bilgi (ROUGE-1, BLEU-2, regex matches vb.)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tests.evaluation.dataset import GoldenItem


@dataclass
class RAGOutput:
    """RAG sisteminin bir soruya verdiği cevabın standart temsili.

    Eval harness, gerçek RAG sisteminden veya mock'lardan beslenebilsin
    diye sistemden bağımsız bir yapı kullanıyoruz.
    """

    answer: str
    retrieved_sources: List[str] = field(default_factory=list)  # source filenames
    retrieved_context: str = ""  # birleştirilmiş retrieved metin
    latency_ms: float = 0.0
    model: str = "unknown"


@dataclass
class LayerResult:
    """Bir katmanın bir örnek üzerindeki sonucu."""

    layer: str
    score: float  # [0.0, 1.0]
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "score": round(self.score, 4),
            "passed": self.passed,
            "details": self.details,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


class BaseEvaluator(ABC):
    """Tüm katmanların türetildiği temel sınıf."""

    name: str = "base"
    threshold: float = 0.5  # pass/fail eşiği

    def evaluate(self, item: GoldenItem, output: RAGOutput) -> LayerResult:
        """Timing + error handling sarmalayıcı; alt sınıf _evaluate yazsın."""
        start = time.perf_counter()
        try:
            score, details = self._evaluate(item, output)
            score = max(0.0, min(1.0, float(score)))
            return LayerResult(
                layer=self.name,
                score=score,
                passed=score >= self.threshold,
                details=details,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            return LayerResult(
                layer=self.name,
                score=0.0,
                passed=False,
                details={},
                latency_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(e).__name__}: {e}",
            )

    @abstractmethod
    def _evaluate(self, item: GoldenItem, output: RAGOutput) -> tuple[float, Dict[str, Any]]:
        """Returns (score in [0,1], details dict)."""
        ...
