"""Integration-test harness: a LIVE RagArt app over a temp workspace.

Unlike the unit tests (which stub `runtime.get_rag_for` with a MagicMock),
these fixtures stand up the *real* stack end-to-end:

    Flask app  →  blueprints  →  RagRegistry  →  TurkishRAGSystem
               →  real pipeline  →  real ChromaDB  →  LocalProvider

The ONLY thing faked is the embedding model — loading the real
sentence-transformers model would pull ~470 MB and is far too heavy for
CI. We swap in a deterministic bag-of-words embedder so retrieval is real
(ChromaDB nearest-neighbour over genuine vectors) but instant and stable.
Relevance scoring is word-overlap based (see calculate_relevance_score),
so lexical overlap between question and document drives the result — which
is exactly what the fake embedder preserves.
"""

from __future__ import annotations

import json

import pytest

# These heavy deps are required for a real run; skip the whole module if a
# minimal environment lacks them rather than erroring at collection.
pytest.importorskip("langchain_chroma")
pytest.importorskip("chromadb")
pytest.importorskip("rank_bm25")


# ── Deterministic bag-of-words embedder ────────────────────────────────
# A vector dimension per vocabulary word + a bias dim so empty strings
# (e.g. the "" probe in _get_data_summary) never produce a zero vector.

_VOCAB = [
    "algoritma", "problem", "çözme", "adım", "sıralı", "işlem", "liste",
    "python", "programlama", "dil", "yüksek", "seviye", "veri", "yapısı",
    "dizi", "fonksiyon", "değişken", "döngü", "koşul", "nesne", "sınıf",
    "yapay", "zeka", "makine", "öğrenme", "model", "sinir", "ağ", "derin",
    "nedir", "nasıl", "tanım", "kavram", "bilgisayar", "yazılım", "bilim",
]


def _vec(text):
    t = (text or "").lower()
    v = [float(t.count(w)) for w in _VOCAB]
    v.append(0.5)  # bias → no zero vectors
    return v


class _FakeEmbeddings:
    """Mimics the langchain Embeddings interface used by Chroma + retrievers."""

    def embed_query(self, text):
        return _vec(text)

    def embed_documents(self, texts):
        return [_vec(t) for t in texts]


def _doc_json(topic, definition, details):
    return json.dumps(
        [{"topic": topic, "definition": definition, "details": details}],
        ensure_ascii=False,
    )


# Seeded knowledge base for the DEFAULT workspace.
_SEED = {
    "algoritma.json": _doc_json(
        "Algoritma",
        "Algoritma, bir problemi çözmek için tasarlanan sıralı adımların listesidir.",
        "Algoritma bilgisayar biliminin temel kavramıdır; her adım bir işlem yapar.",
    ),
    "python.json": _doc_json(
        "Python",
        "Python, yüksek seviyeli bir programlama dilidir.",
        "Python veri yapısı, liste ve dizi gibi kavramları destekler.",
    ),
}


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    """A live test client backed by a temp workspace + fake embedder.

    Module-scoped: the heavy-ish build (real ChromaDB index) happens once
    and is shared across the end-to-end flow tests.
    """
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    data_root = tmp_path_factory.mktemp("ragart_live")

    # 1. Local provider, no network, no API key.
    #    Import settings FIRST so its module-level load_dotenv(override=True)
    #    runs before our overrides — otherwise, when this module is run in
    #    isolation, the .env's MODEL_TYPE would clobber ours.
    import config.settings  # noqa: F401  (side effect: loads .env once)
    mp.setenv("MODEL_TYPE", "local")
    mp.delenv("GROQ_API_KEY", raising=False)

    # 2. Fake embedder — no 470 MB model download.
    from src.embeddings import EmbeddingManager
    mp.setattr(
        EmbeddingManager, "embeddings",
        property(lambda self: _FakeEmbeddings()),
    )

    # 3. Real WorkspaceManager + RagRegistry pointed at the temp dir.
    from src.services import RagRegistry
    from src.workspaces import DEFAULT_WORKSPACE_ID, WorkspaceManager

    wm = WorkspaceManager(str(data_root))
    files = wm.files_dir(DEFAULT_WORKSPACE_ID)
    for name, content in _SEED.items():
        (files / name).write_text(content, encoding="utf-8")
    reg = RagRegistry(wm)

    # 4. Repoint the runtime singletons (functions resolve these globals at
    #    call time, so the blueprints transparently use our temp stack).
    from src.api import runtime
    mp.setattr(runtime, "workspace_manager", wm)
    mp.setattr(runtime, "rag_registry", reg)

    # 5. Warm-start the default workspace (real index build over the seeds).
    #    Production defers the BM25 build to a background thread; block on it
    #    here so hybrid retrieval is deterministic for the assertions below.
    runtime.initialize_default_workspace()
    from src.workspaces import DEFAULT_WORKSPACE_ID as _DEF
    runtime.get_rag_for(_DEF).ensure_sparse_ready(timeout=30)
    runtime.system.ready = True
    runtime.system.status = "Ready"

    # 6. Hand back a test client over the real, fully-wired app.
    import app as app_module
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    yield client, wm, runtime

    mp.undo()


def parse_sse(raw_text):
    """Parse an SSE response body into a list of decoded event dicts."""
    events = []
    for chunk in raw_text.split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("data: "):
            continue
        payload = chunk[len("data: "):]
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events
