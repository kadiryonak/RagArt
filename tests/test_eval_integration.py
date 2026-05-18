"""Integration test: harness uçtan uca çalışıyor mu?

Bu test L4'ü mock'lar (Groq API gerektirmesin), L2 için fake embedder kullanır.
Gerçek RAG sistemi yerine mock RAG callable kullanılır — eval altyapısının
RAG'dan bağımsız çalışabildiğini doğrular.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from tests.evaluation.dataset import load_golden_dataset
from tests.evaluation.layers import (
    L1RulesEvaluator,
    L2VectorEvaluator,
    L3LexicalEvaluator,
    L4JudgeEvaluator,
)
from tests.evaluation.layers.base import RAGOutput
from tests.evaluation.report import render_markdown, write_reports
from tests.evaluation.runner import EvalRunner


def _fake_embed(text: str):
    """Deterministik fake embedder: kelime sayısı + char sum."""
    return [len(text.split()), sum(ord(c) for c in text[:50])]


def _fake_groq_client():
    mock = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps({
            "faithfulness": 4, "relevance": 4, "completeness": 3,
            "reasoning": "mock",
        })}}]
    }
    mock.post.return_value = response
    return mock


def _good_mock_rag(question: str) -> RAGOutput:
    """Algoritma sorularına anlamlı cevap veren mock RAG."""
    answers = {
        "Algoritma nedir?": (
            "Algoritma, bir problemi çözmek için tasarlanan, "
            "adım adım uygulanan sonlu işlemler kümesidir."
        ),
        "Yapay zeka nedir?": (
            "Yapay zeka, makinelerin insan benzeri zeka gerektiren görevleri "
            "yerine getirme yeteneğidir. Makine öğrenmesi ve derin öğrenme alt dallarıdır."
        ),
    }
    answer = answers.get(question, "Bu konuda bilgi bulamadım.")
    return RAGOutput(
        answer=answer,
        retrieved_sources=["Algoritma.json"],
        retrieved_context="Algoritma adım adım çalışan bir yöntemdir.",
        model="mock",
    )


class TestEndToEnd:
    def test_three_layers_run(self):
        """L1+L2+L3 üzerinden 3 örnek çalıştır, hepsi rapor üretsin."""
        evaluators = [
            L1RulesEvaluator(),
            L2VectorEvaluator(embed_fn=_fake_embed),
            L3LexicalEvaluator(),
        ]
        items = load_golden_dataset()[:3]
        runner = EvalRunner(_good_mock_rag, evaluators)
        report = runner.run(items)

        assert len(report.items) == 3
        assert report.layers_used == ["L1_rules", "L2_vector", "L3_lexical"]
        for item_report in report.items:
            assert len(item_report.layers) == 3
            # Her katman bir skor/error üretmiş olmalı
            for layer in item_report.layers:
                assert layer.error is None, f"{layer.layer}: {layer.error}"

    def test_l4_runs_only_on_critical(self):
        """L4 sadece critical=True item'larda çalışmalı, diğerleri skip."""
        client = _fake_groq_client()
        evaluators = [
            L1RulesEvaluator(),
            L4JudgeEvaluator(api_key="fake", client=client, only_critical=True),
        ]
        items = load_golden_dataset()
        runner = EvalRunner(_good_mock_rag, evaluators)
        report = runner.run(items)

        critical_count = sum(1 for i in items if i.critical)
        non_critical_count = sum(1 for i in items if not i.critical)

        l4_runs = sum(
            1 for item in report.items
            for layer in item.layers
            if layer.layer == "L4_judge" and not layer.details.get("skipped")
        )
        l4_skips = sum(
            1 for item in report.items
            for layer in item.layers
            if layer.layer == "L4_judge" and layer.details.get("skipped")
        )

        assert l4_runs == critical_count
        assert l4_skips == non_critical_count
        # Groq mock yalnızca critical'lar için çağrıldı
        assert client.post.call_count == critical_count

    def test_report_aggregations(self):
        evaluators = [L1RulesEvaluator(), L3LexicalEvaluator()]
        items = load_golden_dataset()[:5]
        runner = EvalRunner(_good_mock_rag, evaluators)
        report = runner.run(items)

        by_layer = report.aggregate_by_layer()
        assert "L1_rules" in by_layer
        assert "L3_lexical" in by_layer
        assert 0.0 <= by_layer["L1_rules"]["avg_score"] <= 1.0

        by_diff = report.aggregate_by_difficulty()
        # En az 'easy' kategori dolu olmalı
        assert sum(stats["n"] for stats in by_diff.values()) == 5

    def test_markdown_report_renders(self):
        evaluators = [L1RulesEvaluator()]
        items = load_golden_dataset()[:2]
        runner = EvalRunner(_good_mock_rag, evaluators)
        report = runner.run(items)

        md = render_markdown(report, title="Test")
        assert "# Test" in md
        assert "L1_rules" in md
        assert items[0].id in md

    def test_write_reports_creates_files(self, tmp_path):
        evaluators = [L1RulesEvaluator()]
        items = load_golden_dataset()[:1]
        runner = EvalRunner(_good_mock_rag, evaluators)
        report = runner.run(items)

        paths = write_reports(report, out_dir=str(tmp_path), name_prefix="ut")
        assert Path(paths["md"]).exists()
        assert Path(paths["json"]).exists()
        # JSON parse edilebilmeli
        data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        assert data["total_items"] == 1
