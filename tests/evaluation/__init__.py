"""Layered evaluation harness for the Turkish RAG system.

Katmanlar:
    L1 → Kural tabanlı (regex, format) ......... ms, ücretsiz
    L2 → Vektör benzerliği ..................... sn, minimal maliyet
    L3 → BLEU / ROUGE .......................... sn, ücretsiz
    L4 → LLM-as-a-Judge (Groq free tier) ....... dk, ücretsiz/düşük

L4 sadece kritik checkpoint'lerde, RAGAS sadece major release'lerde çalışır.
"""

from tests.evaluation.dataset import GoldenItem, load_golden_dataset
from tests.evaluation.runner import EvalRunner, run_evaluation

__all__ = ["GoldenItem", "load_golden_dataset", "EvalRunner", "run_evaluation"]
