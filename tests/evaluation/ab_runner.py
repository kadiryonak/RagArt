"""A/B Testing Framework — iki stratejiyi aynı soru setinde karşılaştır.

KULLANIM:
    from tests.evaluation.ab_runner import ABTest

    test = ABTest(
        questions=["Algoritma nedir?", "Python hakkında bilgi ver"],
        strategy_a="direct",
        strategy_b="hyde",
    )
    report = test.run(rag_system)
    test.save(report, "eval-history/ab_direct_vs_hyde.json")

ÇIKTI:
    Her soru için:
        - answer_a, answer_b
        - latency_a, latency_b
        - relevance_a, relevance_b
        - groundedness_a, groundedness_b
        - cache_hit_a, cache_hit_b
    Toplam:
        - ortalama latency, relevance, groundedness
        - kazanan strateji (hangi metrikte daha iyi)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QuestionResult:
    """Tek bir soru için A/B sonuçları."""
    question: str
    answer_a: str = ""
    answer_b: str = ""
    latency_a: float = 0.0
    latency_b: float = 0.0
    relevance_a: float = 0.0
    relevance_b: float = 0.0
    groundedness_a: float = 0.0
    groundedness_b: float = 0.0
    cache_hit_a: Any = False
    cache_hit_b: Any = False
    error_a: Optional[str] = None
    error_b: Optional[str] = None


@dataclass
class ABReport:
    """Tüm A/B test sonuçları."""
    strategy_a: str
    strategy_b: str
    timestamp: str = ""
    results: List[QuestionResult] = field(default_factory=list)

    @property
    def avg_latency_a(self) -> float:
        valid = [r.latency_a for r in self.results if r.error_a is None]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def avg_latency_b(self) -> float:
        valid = [r.latency_b for r in self.results if r.error_b is None]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def avg_relevance_a(self) -> float:
        valid = [r.relevance_a for r in self.results if r.error_a is None]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def avg_relevance_b(self) -> float:
        valid = [r.relevance_b for r in self.results if r.error_b is None]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def avg_groundedness_a(self) -> float:
        valid = [r.groundedness_a for r in self.results if r.error_a is None]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def avg_groundedness_b(self) -> float:
        valid = [r.groundedness_b for r in self.results if r.error_b is None]
        return sum(valid) / len(valid) if valid else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "strategy_a": self.strategy_a,
            "strategy_b": self.strategy_b,
            "num_questions": len(self.results),
            "avg_latency": {
                self.strategy_a: round(self.avg_latency_a, 3),
                self.strategy_b: round(self.avg_latency_b, 3),
            },
            "avg_relevance": {
                self.strategy_a: round(self.avg_relevance_a, 3),
                self.strategy_b: round(self.avg_relevance_b, 3),
            },
            "avg_groundedness": {
                self.strategy_a: round(self.avg_groundedness_a, 3),
                self.strategy_b: round(self.avg_groundedness_b, 3),
            },
            "winner_latency": self.strategy_a if self.avg_latency_a < self.avg_latency_b else self.strategy_b,
            "winner_relevance": self.strategy_a if self.avg_relevance_a > self.avg_relevance_b else self.strategy_b,
            "winner_groundedness": self.strategy_a if self.avg_groundedness_a > self.avg_groundedness_b else self.strategy_b,
        }


class ABTest:
    """İki prompt stratejisini aynı soru seti üzerinde karşılaştırır."""

    def __init__(
        self,
        questions: List[str],
        strategy_a: str = "direct",
        strategy_b: str = "hyde",
    ):
        self.questions = questions
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b

    def run(self, rag_system: Any, **ask_kwargs) -> ABReport:
        """Her soruyu iki stratejiyle çalıştır, sonuçları topla."""
        report = ABReport(
            strategy_a=self.strategy_a,
            strategy_b=self.strategy_b,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        for q in self.questions:
            qr = QuestionResult(question=q)

            # Strategy A
            try:
                t0 = time.perf_counter()
                result_a = rag_system.ask(
                    q,
                    prompt_strategy=self.strategy_a,
                    use_response_cache=False,  # fair comparison
                    **ask_kwargs,
                )
                qr.latency_a = time.perf_counter() - t0
                qr.answer_a = result_a.get("answer", "")
                qr.relevance_a = result_a.get("relevance_score", 0.0)
                qr.groundedness_a = result_a.get("groundedness_score", 0.0)
                qr.cache_hit_a = result_a.get("cache_hit", False)
            except Exception as e:
                qr.error_a = str(e)

            # Strategy B
            try:
                t0 = time.perf_counter()
                result_b = rag_system.ask(
                    q,
                    prompt_strategy=self.strategy_b,
                    use_response_cache=False,
                    **ask_kwargs,
                )
                qr.latency_b = time.perf_counter() - t0
                qr.answer_b = result_b.get("answer", "")
                qr.relevance_b = result_b.get("relevance_score", 0.0)
                qr.groundedness_b = result_b.get("groundedness_score", 0.0)
                qr.cache_hit_b = result_b.get("cache_hit", False)
            except Exception as e:
                qr.error_b = str(e)

            report.results.append(qr)

        return report

    @staticmethod
    def save(report: ABReport, path: str) -> None:
        """Raporu JSON olarak kaydet."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": report.summary(),
            "results": [asdict(r) for r in report.results],
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """Kaydedilmiş raporu oku."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
