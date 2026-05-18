"""Eval orchestration: katmanları sırayla çalıştır, sonuçları topla, rapor üret.

Akış:
    1. Golden dataset yükle
    2. Her item için:
        a. RAG sisteminden (veya verilen runner callable'dan) cevap al
        b. Sırayla L1 → L2 → L3 → (varsa) L4 katmanlarını uygula
        c. Sonuçları ItemReport'a topla
    3. Tüm sonuçları RunReport'a aggregate et
    4. Markdown + JSON çıktı

Tasarım kararı: 'rag_callable' bir Protocol; gerçek RAG sistemi veya mock
geçilebilir. Bu sayede eval harness'ı RAG'dan bağımsız test edilebilir.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from tests.evaluation.dataset import GoldenItem, load_golden_dataset
from tests.evaluation.layers.base import BaseEvaluator, LayerResult, RAGOutput

RAGCallable = Callable[[str], RAGOutput]


@dataclass
class ItemReport:
    """Bir golden item için tüm katmanların sonuçları."""

    item: GoldenItem
    output: RAGOutput
    layers: List[LayerResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Tüm katmanların basit ortalaması (skip'ler hariç)."""
        valid = [
            l for l in self.layers
            if l.error is None and not l.details.get("skipped")
        ]
        if not valid:
            return 0.0
        return sum(l.score for l in valid) / len(valid)

    @property
    def all_passed(self) -> bool:
        valid = [
            l for l in self.layers
            if l.error is None and not l.details.get("skipped")
        ]
        return bool(valid) and all(l.passed for l in valid)

    def to_dict(self) -> dict:
        return {
            "id": self.item.id,
            "question": self.item.question,
            "difficulty": self.item.difficulty,
            "category": self.item.category,
            "critical": self.item.critical,
            "answer": self.output.answer,
            "retrieved_sources": self.output.retrieved_sources,
            "rag_latency_ms": round(self.output.latency_ms, 2),
            "overall_score": round(self.overall_score, 4),
            "all_passed": self.all_passed,
            "layers": [l.to_dict() for l in self.layers],
        }


@dataclass
class RunReport:
    """Tüm koşumun aggregate raporu."""

    items: List[ItemReport] = field(default_factory=list)
    total_duration_s: float = 0.0
    layers_used: List[str] = field(default_factory=list)

    def aggregate_by_layer(self) -> Dict[str, Dict[str, float]]:
        """Her katman için ortalama skor, pass oranı, ortalama latency."""
        out: Dict[str, Dict[str, float]] = {}
        for layer_name in self.layers_used:
            scores, passes, latencies = [], [], []
            for item in self.items:
                for layer in item.layers:
                    if layer.layer != layer_name:
                        continue
                    if layer.error or layer.details.get("skipped"):
                        continue
                    scores.append(layer.score)
                    passes.append(1.0 if layer.passed else 0.0)
                    latencies.append(layer.latency_ms)
            if not scores:
                out[layer_name] = {
                    "n": 0, "avg_score": 0.0, "pass_rate": 0.0, "avg_latency_ms": 0.0
                }
                continue
            out[layer_name] = {
                "n": len(scores),
                "avg_score": round(sum(scores) / len(scores), 4),
                "pass_rate": round(sum(passes) / len(passes), 4),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            }
        return out

    def aggregate_by_difficulty(self) -> Dict[str, Dict[str, float]]:
        groups: Dict[str, List[float]] = {}
        for item in self.items:
            groups.setdefault(item.item.difficulty, []).append(item.overall_score)
        return {
            k: {"n": len(v), "avg_score": round(sum(v) / len(v), 4)}
            for k, v in groups.items()
        }

    def to_dict(self) -> dict:
        return {
            "total_items": len(self.items),
            "total_duration_s": round(self.total_duration_s, 2),
            "layers_used": self.layers_used,
            "by_layer": self.aggregate_by_layer(),
            "by_difficulty": self.aggregate_by_difficulty(),
            "overall_avg_score": round(
                sum(i.overall_score for i in self.items) / len(self.items), 4
            ) if self.items else 0.0,
            "items": [i.to_dict() for i in self.items],
        }


class EvalRunner:
    """Layered evaluation orchestrator."""

    def __init__(
        self,
        rag_callable: RAGCallable,
        evaluators: Sequence[BaseEvaluator],
    ):
        self.rag_callable = rag_callable
        self.evaluators = list(evaluators)

    def run(
        self,
        items: Sequence[GoldenItem],
        *,
        on_item: Optional[Callable[[ItemReport], None]] = None,
    ) -> RunReport:
        start = time.perf_counter()
        report = RunReport(layers_used=[e.name for e in self.evaluators])

        for item in items:
            t0 = time.perf_counter()
            output = self.rag_callable(item.question)
            output.latency_ms = (time.perf_counter() - t0) * 1000

            item_report = ItemReport(item=item, output=output)
            for evaluator in self.evaluators:
                result = evaluator.evaluate(item, output)
                item_report.layers.append(result)

            report.items.append(item_report)
            if on_item:
                on_item(item_report)

        report.total_duration_s = time.perf_counter() - start
        return report


def run_evaluation(
    rag_callable: RAGCallable,
    evaluators: Sequence[BaseEvaluator],
    dataset_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> RunReport:
    """Convenience wrapper: load dataset + run."""
    items = load_golden_dataset(dataset_path)
    if limit:
        items = items[:limit]
    runner = EvalRunner(rag_callable, evaluators)
    return runner.run(items)
