"""Unit tests for TurkishRAGSystem helper methods.

These cover the pure-logic methods of the orchestrator that don't need a
live ChromaDB / embedding model: provider selection, retriever selection,
memory/strategy/context-chain builders, RRF fusion, the streaming payload
helpers and the fallback response. Heavy externals are stubbed with the
same pattern used in test_reindex_cache.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("langchain_chroma")

try:
    from langchain_core.documents import Document
except ImportError:  # older langchain
    from langchain.schema import Document


class _FakeEmbeddings:
    def embed_query(self, text):
        return [0.0] * 4

    def embed_documents(self, texts):
        return [[0.0] * 4 for _ in texts]


@pytest.fixture
def rag(monkeypatch):
    from src.rag_system import TurkishRAGSystem
    from src.embeddings import EmbeddingManager
    import chromadb

    monkeypatch.setattr(
        EmbeddingManager, "embeddings",
        property(lambda self: _FakeEmbeddings()),
    )
    monkeypatch.setattr(chromadb, "PersistentClient", lambda path: MagicMock())

    return TurkishRAGSystem(
        data_folder=".",
        model_type="local",
        api_key=None,
        chroma_db_path="./does-not-exist",
    )


def _doc(source, content="content", item_index=0):
    return Document(page_content=content, metadata={"source": source, "item_index": item_index})


# ── _create_llm_provider ───────────────────────────────────────────────


class TestCreateLLMProvider:
    def test_local_provider_created(self, rag):
        from src.llm_providers import LocalProvider
        assert isinstance(rag._create_llm_provider(), LocalProvider)

    def test_invalid_config_falls_back_to_local(self, rag):
        from src.llm_providers import LocalProvider
        rag.model_type = "groq"
        rag.api_key = None  # groq requires a key → ValueError → local
        assert isinstance(rag._create_llm_provider(), LocalProvider)


# ── _select_retriever ──────────────────────────────────────────────────


class TestSelectRetriever:
    def _wire(self, rag):
        rag._dense_retriever = MagicMock(name="dense")
        rag._dense_retriever.name = "dense"
        rag._sparse_retriever = MagicMock(name="sparse")
        rag._sparse_retriever.name = "sparse"
        rag._hybrid_retriever = MagicMock(name="hybrid")
        rag._hybrid_retriever.name = "hybrid"
        rag._reranker_cache = {}

    def test_dense(self, rag):
        self._wire(rag)
        assert rag._select_retriever("dense") is rag._dense_retriever

    def test_sparse(self, rag):
        self._wire(rag)
        assert rag._select_retriever("sparse") is rag._sparse_retriever

    def test_hybrid(self, rag):
        self._wire(rag)
        assert rag._select_retriever("hybrid") is rag._hybrid_retriever

    def test_auto_prefers_hybrid(self, rag):
        self._wire(rag)
        assert rag._select_retriever(None) is rag._hybrid_retriever

    def test_auto_falls_back_to_dense_when_no_hybrid(self, rag):
        self._wire(rag)
        rag._hybrid_retriever = None
        assert rag._select_retriever(None) is rag._dense_retriever

    def test_none_base_returns_none(self, rag):
        rag._dense_retriever = rag._sparse_retriever = rag._hybrid_retriever = None
        rag._reranker_cache = {}
        assert rag._select_retriever("dense") is None

    def test_rerank_wraps_and_caches(self, rag, monkeypatch):
        import src.rag_system as mod
        self._wire(rag)
        made = []

        def fake_reranked(base, fetch_k):
            made.append(base)
            return MagicMock(name="reranked")
        monkeypatch.setattr(mod, "RerankedRetriever", fake_reranked)

        first = rag._select_retriever("dense", rerank=True)
        second = rag._select_retriever("dense", rerank=True)
        assert first is second           # cached, not rebuilt
        assert len(made) == 1            # RerankedRetriever built once


# ── deferred sparse/BM25 build (no-503 startup) ────────────────────────


class TestSparseBuild:
    def test_sync_build_runs_inline(self, rag, monkeypatch):
        calls = []
        monkeypatch.setattr(rag, "_reload_split_chunks", lambda: ["chunk"])
        monkeypatch.setattr(
            rag, "_build_retrievers", lambda docs: calls.append(docs)
        )
        rag._build_sparse_retrievers(defer=False)
        # Synchronous: retrievers built before the call returns.
        assert calls == [["chunk"]]

    def test_deferred_build_runs_in_background(self, rag, monkeypatch):
        import threading

        started = threading.Event()
        release = threading.Event()
        built = []

        def slow_reload():
            started.set()
            release.wait(2)
            return ["chunk"]

        monkeypatch.setattr(rag, "_reload_split_chunks", slow_reload)
        monkeypatch.setattr(rag, "_build_retrievers", lambda docs: built.append(docs))

        rag._build_sparse_retrievers(defer=True)
        # Returned immediately while the build is still blocked.
        assert started.wait(2)
        assert built == []
        # Let it finish and confirm ensure_sparse_ready joins it.
        release.set()
        rag.ensure_sparse_ready(timeout=2)
        assert built == [["chunk"]]

    def test_deferred_build_swallows_errors(self, rag, monkeypatch):
        def boom():
            raise RuntimeError("disk gone")

        monkeypatch.setattr(rag, "_reload_split_chunks", boom)
        # Must not raise — a failed background build just stays dense-only.
        rag._build_sparse_retrievers(defer=False)
        assert rag._sparse_retriever is None


# ── _build_memory ──────────────────────────────────────────────────────


class TestBuildMemory:
    @pytest.mark.parametrize("strategy", [None, "", "none", "unknown"])
    def test_no_memory_variants(self, rag, strategy):
        from src.memory import NoMemory
        assert isinstance(rag._build_memory(strategy), NoMemory)

    def test_sliding_window(self, rag):
        from src.memory import SlidingWindowMemory
        assert isinstance(rag._build_memory("sliding_window"), SlidingWindowMemory)

    def test_summary_buffer(self, rag):
        from src.memory import SummaryBufferMemory
        assert isinstance(rag._build_memory("summary_buffer"), SummaryBufferMemory)

    def test_vector(self, rag):
        from src.memory import VectorRetrievalMemory
        assert isinstance(rag._build_memory("vector"), VectorRetrievalMemory)


# ── _fuse_retrievals (RRF) ─────────────────────────────────────────────


class TestFuseRetrievals:
    def test_dedup_across_queries(self, rag):
        shared = _doc("a.json", "shared content", 0)
        only_b = _doc("b.json", "other content", 1)

        def retrieve_fn(q, k):
            if q == "q1":
                return [shared, only_b]
            return [shared]  # q2 returns the shared doc again

        out = rag._fuse_retrievals(["q1", "q2"], k=5, retrieve_fn=retrieve_fn)
        # Shared doc deduped → 2 unique docs total
        sources = sorted(d.metadata["source"] for d in out)
        assert sources == ["a.json", "b.json"]

    def test_respects_k_limit(self, rag):
        docs = [_doc(f"f{i}.json", f"c{i}", i) for i in range(10)]
        out = rag._fuse_retrievals(["q"], k=3, retrieve_fn=lambda q, k: docs)
        assert len(out) == 3

    def test_higher_rank_scores_higher(self, rag):
        top = _doc("top.json", "top", 0)
        bottom = _doc("bottom.json", "bottom", 1)
        out = rag._fuse_retrievals(["q"], k=2, retrieve_fn=lambda q, k: [top, bottom])
        # rank 0 (top) should fuse to a higher RRF score → appear first
        assert out[0].metadata["source"] == "top.json"


# ── _resolve_prompt_strategy ───────────────────────────────────────────


class TestResolvePromptStrategy:
    def test_default_is_direct(self, rag):
        from src.prompt_strategies import DirectStrategy
        assert isinstance(rag._resolve_prompt_strategy(None), DirectStrategy)

    def test_unknown_falls_back_to_direct(self, rag):
        from src.prompt_strategies import DirectStrategy
        assert isinstance(rag._resolve_prompt_strategy("nonsense"), DirectStrategy)

    def test_role_based_injects_role(self, rag):
        from src.prompt_strategies import RoleBasedStrategy
        s = rag._resolve_prompt_strategy("role_based", custom_role="hukuk danışmanı")
        assert isinstance(s, RoleBasedStrategy)
        assert "hukuk danışmanı" in s.role

    def test_custom_injects_template(self, rag):
        from src.prompt_strategies import CustomStrategy
        s = rag._resolve_prompt_strategy("custom", custom_prompt_template="T={question}")
        assert isinstance(s, CustomStrategy)

    def test_known_strategy_created(self, rag):
        from src.prompt_strategies import ChainOfThoughtStrategy
        assert isinstance(
            rag._resolve_prompt_strategy("chain_of_thought"), ChainOfThoughtStrategy
        )


# ── _build_context_chain ───────────────────────────────────────────────


class TestBuildContextChain:
    def test_no_processors_returns_none(self, rag):
        assert rag._build_context_chain() is None

    def test_dedup_builds_chain(self, rag):
        chain = rag._build_context_chain(deduplicate=True)
        assert chain is not None

    def test_all_processors(self, rag):
        chain = rag._build_context_chain(
            deduplicate=True, reorder=True, max_context_tokens=100,
        )
        assert chain is not None


# ── _build_query_request ───────────────────────────────────────────────


class TestBuildQueryRequest:
    def test_defaults(self, rag):
        req = rag._build_query_request("soru")
        assert req.question == "soru"
        assert req.k == 5
        assert req.rerank is False
        assert req.selected_files == ()
        assert req.use_response_cache is True

    def test_overrides_applied(self, rag):
        req = rag._build_query_request(
            "soru", k=10, rerank=True, selected_files=["a.json"],
            prompt_strategy="hyde", use_semantic_cache=True,
        )
        assert req.k == 10
        assert req.rerank is True
        assert req.selected_files == ("a.json",)
        assert req.prompt_strategy == "hyde"
        assert req.use_semantic_cache is True


# ── _stream_sources / _stream_done_payload ─────────────────────────────


class TestStreamHelpers:
    def test_stream_sources_empty(self, rag):
        assert rag._stream_sources([]) == []

    def test_stream_sources_truncates_long_content(self, rag):
        d = _doc("x.json", "y" * 500)
        out = rag._stream_sources([d])
        assert out[0]["title"] == "x.json"
        assert out[0]["content"].endswith("...")
        assert len(out[0]["content"]) == 303  # 300 + "..."

    def test_stream_sources_short_content_untouched(self, rag):
        out = rag._stream_sources([_doc("x.json", "kısa")])
        assert out[0]["content"] == "kısa"

    def test_stream_done_payload_reads_response(self, rag):
        state = MagicMock()
        state.answer = "fallback"
        state.timings = {"retrieval": 0.1}
        resp = {
            "answer": "asıl cevap", "source": "rag_system",
            "relevance_score": 0.7, "groundedness_score": 0.5,
            "cache_hit": True,
        }
        out = rag._stream_done_payload(resp, state)
        assert out["answer"] == "asıl cevap"
        assert out["relevance_score"] == 0.7
        assert out["groundedness"] == 0.5
        assert out["cache_hit"] is True
        assert out["timings"] == {"retrieval": 0.1}

    def test_stream_done_payload_uses_state_answer_fallback(self, rag):
        state = MagicMock()
        state.answer = "state answer"
        state.timings = {}
        out = rag._stream_done_payload({}, state)
        assert out["answer"] == "state answer"
        assert out["source_type"] == "rag_system"


# ── _fallback_response ─────────────────────────────────────────────────


class TestFallbackResponse:
    def test_safe_default_no_llm_call(self, rag):
        provider = MagicMock()
        out = rag._fallback_response("soru", 0.05, llm_provider=provider)
        assert out["source"] == "insufficient_data"
        provider.generate_general.assert_not_called()

    def test_opt_in_cloud_calls_llm(self, rag):
        provider = MagicMock()
        provider.generate_general.return_value = "genel cevap"
        rag.vector_store = MagicMock()
        rag.vector_store.similarity_search.return_value = []
        out = rag._fallback_response(
            "soru", 0.05, llm_provider=provider, allow_general_knowledge=True,
        )
        assert out["source"] == "general_knowledge_fallback"
        assert "genel cevap" in out["answer"]
        provider.generate_general.assert_called_once()

    def test_opt_in_local_provider_no_hallucination(self, rag):
        from src.llm_providers import LocalProvider
        out = rag._fallback_response(
            "soru", 0.05, llm_provider=LocalProvider(),
            allow_general_knowledge=True,
        )
        assert out["source"] == "insufficient_data"
        assert "0.050" in out["answer"]


# ── _get_data_summary ──────────────────────────────────────────────────


class TestGetDataSummary:
    def test_topics_from_sources(self, rag):
        rag.vector_store = MagicMock()
        rag.vector_store.similarity_search.return_value = [
            _doc("machine_learning.json"), _doc("data_structures.json"),
        ]
        out = rag._get_data_summary()
        assert "Topics:" in out
        assert "Machine Learning" in out

    def test_empty_returns_generic(self, rag):
        rag.vector_store = MagicMock()
        rag.vector_store.similarity_search.return_value = []
        assert "Various technical" in rag._get_data_summary()

    def test_exception_returns_generic(self, rag):
        rag.vector_store = MagicMock()
        rag.vector_store.similarity_search.side_effect = RuntimeError("down")
        assert "Technical topics" in rag._get_data_summary()


# ── calculate_relevance_score (best-doc, not mean) ─────────────────────


class TestRelevanceScore:
    def test_empty_docs_is_zero(self, rag):
        assert rag.calculate_relevance_score("kadir kimdir", []) == 0.0

    def test_uses_best_doc_not_mean(self, rag):
        # One strongly-matching doc + distractors. Mean would dilute to ~0.12
        # and fail the 0.1 gate; max keeps the real hit at 0.5.
        docs = [
            _doc("cv.txt", "Kadir Yönak bir yapay zeka mühendisidir."),
            _doc("a.json", "Tamamen alakasız bir metin burada."),
            _doc("b.json", "Başka alakasız içerik."),
            _doc("c.json", "Yine farklı bir konu."),
        ]
        score = rag.calculate_relevance_score("kadir kimdir", docs)
        # "kadir" present in best doc → 1 of 2 question words → 0.5
        assert score == pytest.approx(0.5)

    def test_out_of_domain_stays_zero(self, rag):
        docs = [_doc("a.json", "Python programlama dili"), _doc("b.json", "Algoritma")]
        # Neither doc shares a word with the question → still rejected.
        assert rag.calculate_relevance_score("deprem nedir", docs) == 0.0


# ── search with per-file selection ─────────────────────────────────────


class TestSearchAllowedSources:
    def test_no_vector_store_returns_empty(self, rag):
        rag.vector_store = None
        assert rag.search("q", allowed_sources={"a.json"}) == []

    def test_uses_chroma_where_filter(self, rag):
        rag.vector_store = MagicMock()
        rag.vector_store.similarity_search.return_value = [_doc("a.json")]
        out = rag.search("q", k=3, allowed_sources={"a.json"})
        assert len(out) == 1
        _args, kwargs = rag.vector_store.similarity_search.call_args
        assert kwargs["filter"] == {"source": {"$in": ["a.json"]}}

    def test_filter_failure_raises_stale_index(self, rag):
        rag.vector_store = MagicMock()
        rag.vector_store.similarity_search.side_effect = RuntimeError("collection gone")
        with pytest.raises(RuntimeError, match="STALE_INDEX"):
            rag.search("q", allowed_sources={"a.json"})
