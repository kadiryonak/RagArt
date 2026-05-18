"""CLI entry point: layered RAG evaluation.

Usage:
    python scripts/run_eval.py                          # tüm katmanlar, L4 hariç
    python scripts/run_eval.py --with-judge             # L4 dahil
    python scripts/run_eval.py --layers L1,L3           # sadece L1+L3
    python scripts/run_eval.py --limit 3                # ilk 3 örnek
    python scripts/run_eval.py --dataset path/to.json   # alternatif dataset
    python scripts/run_eval.py --name baseline          # rapor ad öneki
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Repo kök dizinini PYTHONPATH'a ekle
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.evaluation.dataset import load_golden_dataset  # noqa: E402
from tests.evaluation.layers import (  # noqa: E402
    L1RulesEvaluator,
    L2VectorEvaluator,
    L3LexicalEvaluator,
    L4JudgeEvaluator,
)
from tests.evaluation.layers.base import RAGOutput  # noqa: E402
from tests.evaluation.report import write_reports  # noqa: E402
from tests.evaluation.runner import EvalRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Layered RAG evaluation runner")
    p.add_argument("--dataset", default=None, help="Golden dataset JSON path")
    p.add_argument(
        "--layers",
        default="L1,L2,L3",
        help="Comma-separated layer names (L1,L2,L3,L4)",
    )
    p.add_argument(
        "--with-judge",
        action="store_true",
        help="Include L4 (Groq LLM judge). Requires GROQ_API_KEY.",
    )
    p.add_argument("--limit", type=int, default=None, help="Max items to evaluate")
    p.add_argument("--name", default="baseline", help="Report filename prefix")
    p.add_argument(
        "--mock-rag",
        action="store_true",
        help="Skip real RAG, use a trivial mock (sanity check the harness).",
    )
    p.add_argument(
        "--provider",
        default=None,
        help="LLM provider override (deepseek/openai/groq/ollama/huggingface/local). "
             "Uses server default when omitted.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="API key for the chosen provider. Falls back to <PROVIDER>_API_KEY env.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model name override (e.g. llama-3.3-70b-versatile for groq).",
    )
    p.add_argument(
        "--retrieval",
        default=None,
        choices=["dense", "sparse", "hybrid"],
        help="Retrieval strategy override (default: auto = hybrid if available).",
    )
    p.add_argument(
        "--rerank",
        action="store_true",
        help="Apply cross-encoder reranker on top of retrieval. "
             "First call downloads ~400MB (bge-reranker-v2-m3).",
    )
    p.add_argument(
        "--rerank-fetch-k",
        type=int, default=20,
        help="Candidates fed to reranker (default 20). "
             "Higher = better recall, slower (~50ms per candidate).",
    )
    return p.parse_args()


def build_rag_callable(
    use_mock: bool,
    *,
    provider_override: str | None = None,
    api_key_override: str | None = None,
    model_override: str | None = None,
    retrieval_strategy: str | None = None,
    rerank: bool = False,
    rerank_fetch_k: int = 20,
):
    """Gerçek RAG sistemini veya mock'u döndür.

    provider_override verildiğinde, server-side default RAG hâlâ embedder+
    retrieval için kullanılır, ama her ask() çağrısında LLM bu override ile
    swap edilir. Bu, UI'daki BYOK akışının CLI eşdeğeridir.
    """
    if use_mock:
        def mock(question: str) -> RAGOutput:
            return RAGOutput(
                answer=f"[MOCK] Bu bir mock cevaptır. Soru: {question}",
                retrieved_sources=["mock.json"],
                retrieved_context="Mock bağlam metni.",
                model="mock",
            )
        return mock, None

    from src.rag_system import TurkishRAGSystem
    from src.llm_providers import LLMProviderFactory
    from config.settings import settings

    rag = TurkishRAGSystem(
        data_folder=settings.DATA_FOLDER,
        model_type=settings.MODEL_TYPE,
        api_key=settings.get_api_key(),
        chroma_db_path=settings.CHROMA_DB_PATH,
    )
    rag.initialize()

    llm_override = None
    if provider_override:
        # Env fallback: --api-key None → look up <PROVIDER>_API_KEY
        api_key = api_key_override or os.getenv(f"{provider_override.upper()}_API_KEY")
        llm_override = LLMProviderFactory.create(
            provider_override,
            api_key=api_key,
            model=model_override,
        )
        print(f"[info] LLM override: {provider_override} "
              f"(model={getattr(llm_override, 'model', '?')})")

    if retrieval_strategy:
        print(f"[info] Retrieval strategy: {retrieval_strategy}")
    if rerank:
        print(f"[info] Reranker enabled (fetch_k={rerank_fetch_k})")

    def real(question: str) -> RAGOutput:
        result = rag.ask(
            question,
            llm_provider=llm_override,
            retrieval_strategy=retrieval_strategy,
            rerank=rerank,
            rerank_fetch_k=rerank_fetch_k,
        )
        return RAGOutput(
            answer=result.get("answer", ""),
            retrieved_sources=[
                d.get("source", "") for d in result.get("source_documents", [])
            ],
            retrieved_context=result.get("context_used", ""),
            model=provider_override or rag.model_type,
        )

    return real, rag


def build_evaluators(layer_names: set[str], rag_obj=None, with_judge: bool = False):
    """Seçilen katmanlara karşılık gelen evaluator listesi."""
    evaluators = []
    if "L1" in layer_names:
        evaluators.append(L1RulesEvaluator())
    if "L2" in layer_names:
        if rag_obj is None:
            print("[uyarı] L2 için gerçek RAG embedder gerekli; --mock-rag ile atlandı.")
        else:
            evaluators.append(L2VectorEvaluator(embed_fn=rag_obj.embedding_manager.embed_query))
    if "L3" in layer_names:
        evaluators.append(L3LexicalEvaluator())
    if "L4" in layer_names or with_judge:
        judge = L4JudgeEvaluator(only_critical=True)
        if judge.is_available():
            evaluators.append(judge)
        else:
            print("[uyarı] GROQ_API_KEY tanımlı değil — L4 katmanı atlandı.")
    return evaluators


def main() -> int:
    args = parse_args()
    layer_set = {x.strip().upper() for x in args.layers.split(",") if x.strip()}

    print(f"== Layered RAG Evaluation ==")
    print(f"Dataset:  {args.dataset or '(default)'}")
    print(f"Layers:   {sorted(layer_set)}{' + L4' if args.with_judge and 'L4' not in layer_set else ''}")
    print(f"Mock RAG: {args.mock_rag}")

    items = load_golden_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]
    print(f"Items:    {len(items)}")

    if args.mock_rag:
        rag_callable, rag_obj = build_rag_callable(use_mock=True)
    else:
        rag_callable, rag_obj = build_rag_callable(
            use_mock=False,
            provider_override=args.provider,
            api_key_override=args.api_key,
            model_override=args.model,
            retrieval_strategy=args.retrieval,
            rerank=args.rerank,
            rerank_fetch_k=args.rerank_fetch_k,
        )

    evaluators = build_evaluators(layer_set, rag_obj=rag_obj, with_judge=args.with_judge)
    if not evaluators:
        print("[hata] Hiç katman seçilmedi.")
        return 2

    print(f"Evaluators: {[e.name for e in evaluators]}")
    print("")

    runner = EvalRunner(rag_callable, evaluators)
    t0 = time.perf_counter()

    def progress(item_report):
        flag = "PASS" if item_report.all_passed else "FAIL"
        print(
            f"  [{flag}] {item_report.item.id:<32} "
            f"score={item_report.overall_score:.3f} "
            f"({item_report.output.latency_ms:.0f}ms)"
        )

    report = runner.run(items, on_item=progress)
    elapsed = time.perf_counter() - t0

    paths = write_reports(report, name_prefix=args.name)
    print("")
    print(f"Tamamlandı in {elapsed:.2f}s")
    print(f"  Markdown: {paths['md']}")
    print(f"  JSON:     {paths['json']}")
    print(f"  Ortalama skor: {report.to_dict()['overall_avg_score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
