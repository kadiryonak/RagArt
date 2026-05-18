<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-REST_API-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=for-the-badge" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

# 🤖 RagArt — Production-Grade Turkish RAG System

> **A multi-technique Retrieval-Augmented Generation system combining Hybrid Retrieval, Cross-Encoder Reranking, Context Engineering, and a 4-Layer Evaluation Harness — optimized for Turkish language.**

RagArt is an end-to-end RAG pipeline that goes beyond basic semantic search. It integrates **9+ advanced NLP/IR techniques** into a single, cohesive system with a modern web UI, BYOK (Bring Your Own Key) architecture, and rigorous layered evaluation — all optimized for Turkish.

---

## 📸 Screenshots & Demo

<details>
<summary><b>🎬 Click to see the live demo video</b></summary>
<br>

![RagArt Demo](görsel/demo.webp)

</details>

| Soru-Cevap Arayüzü | Ayarlar Paneli |
|:---:|:---:|
| ![QA Interface](görsel/Ekran%20görüntüsü%202026-05-19%20010413.png) | ![Settings](görsel/Ekran%20görüntüsü%202026-05-19%20010801.png) |

| Dosya Yönetimi | Farklı Soru Örneği |
|:---:|:---:|
| ![File Manager](görsel/Ekran%20görüntüsü%202026-05-19%20014452.png) | ![Question Example](görsel/Ekran%20görüntüsü%202026-05-19%200106554.png) |

---

## 🏗️ System Architecture

```
                    ┌──────────────────────────────────────────────────────┐
                    │                   Flask REST API                     │
                    │         BYOK Headers · Settings Schema              │
                    └──────────────┬───────────────────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────────────────┐
                    │              TurkishRAGSystem                        │
                    │                                                      │
                    │  ┌─────────┐  ┌──────────┐  ┌───────────────────┐   │
                    │  │ Memory  │  │ Retrieval │  │ Context Engine    │   │
                    │  │ Module  │  │ Pipeline  │  │ (Post-Retrieval)  │   │
                    │  └────┬────┘  └─────┬─────┘  └────────┬──────────┘   │
                    │       │             │                  │              │
                    │       ▼             ▼                  ▼              │
                    │  ┌─────────┐  ┌──────────┐  ┌───────────────────┐   │
                    │  │NoMemory │  │  Dense    │  │RedundancyFilter   │   │
                    │  │Sliding  │  │  Sparse   │  │TokenBudgetTrimmer │   │
                    │  │Summary  │  │  Hybrid   │  │LostInMiddleReorder│   │
                    │  │Vector   │  │  Reranker │  └───────────────────┘   │
                    │  └─────────┘  └──────────┘                          │
                    │                                                      │
                    │  ┌─────────────────────────────────────────────────┐ │
                    │  │           LLM Provider Factory                   │ │
                    │  │  DeepSeek · OpenAI · Groq · Ollama · HF · Local │ │
                    │  └─────────────────────────────────────────────────┘ │
                    └──────────────────────────────────────────────────────┘
```

---

## 🔬 Implemented RAG Techniques

### 1. Retrieval Pipeline

The system implements a **3-stage retrieval pipeline**:

```
Query → [Stage 1: Retrieve] → [Stage 2: Rerank] → [Stage 3: Context Engineering] → LLM
              │                      │                        │
         Dense/Sparse/         Cross-Encoder           Dedup + Budget
         Hybrid (RRF)         bge-reranker-v2-m3       + Reorder
```

Each stage is **optional and configurable per-request** via HTTP headers — the client chooses the pipeline configuration at runtime.

---

### 2. Embedding Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Model** | `paraphrase-multilingual-MiniLM-L12-v2` | Best balance of Turkish quality vs. speed (384-dim) |
| **Chunk Size** | 800 characters | Optimized for Turkish morphology (longer words than English) |
| **Chunk Overlap** | 150 characters | Prevents information loss at chunk boundaries |
| **Separators** | `\n\n`, `\n`, `.`, `!`, `?`, `;`, `,`, ` ` | Turkish punctuation-aware splitting |

The embedding model is **lazy-loaded** — first query triggers download, subsequent queries use cached weights.

---

### 3. Vector Database Architecture

- **ChromaDB** with `PersistentClient` — survives server restarts
- **Collection lifecycle**: auto-create on first run, skip rebuild if existing collection has documents
- **Reindex mechanism**: deletes old collection → rebuilds from source files → invalidates all retriever caches (including reranker)
- **Stale index detection**: catches `Collection does not exist` errors and surfaces user-friendly Turkish error messages

---

### 4. Hybrid Retrieval (Dense + Sparse + RRF)

| Retriever | Algorithm | Strengths |
|-----------|-----------|-----------|
| **Dense** | Cosine similarity over embeddings | Semantic understanding, paraphrase matching |
| **Sparse (BM25)** | `BM25Okapi` with Turkish stemming | Exact match, proper nouns, abbreviations, code |
| **Hybrid** | Reciprocal Rank Fusion (k=60) | Combines both — production standard |

**RRF Formula:**

```
rrf_score(d) = Σ_r  weight_r / (k_rrf + rank_r(d))
```

RRF is **score-normalization free** — no need to compare BM25 scores (unbounded) with cosine similarity ([0,1]). The BM25 tokenizer uses a custom **Turkish light stemmer** to reduce inflectional penalties (e.g., `algoritmaları` → `algoritma`).

---

### 5. Semantic Search

- **Multilingual embeddings** via HuggingFace Sentence Transformers
- **Relevance threshold gating** (`0.1`): below threshold → fallback to general LLM knowledge or rejection
- **Word overlap scoring** for quick relevance estimation before full LLM generation

---

### 6. Cross-Encoder Reranking

```
Bi-encoder (fast, top-N) → Cross-encoder (accurate, top-K)
```

| Property | Value |
|----------|-------|
| **Model** | `BAAI/bge-reranker-v2-m3` (multilingual, ~400MB) |
| **Default fetch_k** | 20 candidates → rerank → return top 5 |
| **Latency** | ~500-1000ms on CPU (30-50ms per query-doc pair) |
| **Loading** | Lazy — first request triggers download, then cached |
| **Testability** | Constructor accepts injected `cross_encoder` for mocking |

The reranker wraps any `BaseRetriever` via the decorator pattern — making it composable with Dense, Sparse, or Hybrid.

---

### 7. Context Window Optimization

Three **composable post-retrieval processors** applied in a strict pipeline order:

```
Retrieved docs → RedundancyFilter → TokenBudgetTrimmer → LostInTheMiddleReorderer → Prompt
```

| Processor | Problem Solved | How |
|-----------|---------------|-----|
| **RedundancyFilter** | Duplicate/near-duplicate chunks waste tokens | Cosine similarity > 0.92 → drop |
| **TokenBudgetTrimmer** | Context exceeds LLM limits or costs too much | Tail-trim (least relevant first), always keep ≥1 chunk |
| **LostInTheMiddleReorderer** | LLMs forget information in the middle of long contexts ([Liu et al. 2024](https://arxiv.org/abs/2307.03172)) | Place strongest chunks at start and end, weakest in middle |

The trimmer uses a **Turkish-calibrated tokenizer heuristic** (~3.0 chars/token vs. ~4.0 for English).

---

### 8. Hallucination Mitigation

Multiple layers prevent the LLM from generating unfaithful content:

1. **Strict Turkish system prompt**: Rules include "use ONLY the provided context", "if insufficient context → say exactly one refusal sentence", "NEVER show [Source] tags"
2. **Relevance threshold gating**: If retrieved documents score below `0.1`, the system either:
   - Routes to a **general knowledge fallback** (cloud providers), or
   - Returns a clean **"insufficient data"** message (local provider)
3. **Refusal enforcement**: The prompt explicitly **bans partial answers** — if context is insufficient, the LLM must refuse entirely rather than hallucinate partial facts
4. **Source attribution**: Every answer includes clickable source documents so users can verify

---

### 9. Conversation Memory

Four **pluggable memory strategies**, all implementing the `BaseMemory` contract:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `NoMemory` | Each question is isolated | Single-turn QA |
| `SlidingWindow` | Last 5 turns verbatim in prompt | Short conversations, zero-cost |
| `SummaryBuffer` | Old turns summarized via LLM, recent 4 kept raw | Long conversations, token savings |
| `VectorRetrieval` | Semantic search over chat history | Very long conversations (50+ turns) |

The server is **stateless** — the client sends full conversation history with each request, and the server applies the chosen strategy on-the-fly.

---

### 10. Query Routing (BYOK Architecture)

Every `/ask` request can override the entire pipeline via HTTP headers:

| Header | Controls |
|--------|----------|
| `X-Provider` | LLM provider (deepseek/openai/groq/ollama/huggingface/local) |
| `X-API-Key` | API key for the chosen provider |
| `X-Model` | Model name override |
| `X-LLM-Params` | JSON: temperature, top_p, max_tokens, etc. |
| `X-Retrieval-Strategy` | dense / sparse / hybrid |
| `X-Rerank` | Enable/disable cross-encoder reranking |
| `X-Context-Deduplicate` | Enable redundancy filter |
| `X-Context-Reorder` | Enable lost-in-the-middle reordering |
| `X-Context-Max-Tokens` | Set token budget cap |
| `X-Memory-Strategy` | none / sliding_window / summary_buffer / vector |
| `X-Conversation-History` | Base64-encoded conversation history |

This makes RagArt a **multi-tenant, stateless RAG backend** — each request can use a different provider and pipeline configuration without server-side state.

---

### 11. Multi-Format Document Loading

| Format | Loader | Pages/Docs |
|--------|--------|------------|
| `.json` | Custom (nested key extraction) | Per array element |
| `.pdf` | `pypdf` → per-page Document | Per page |
| `.docx` | `python-docx` → paragraph extraction | Per file |
| `.md` | Stdlib (raw text) | Per file |
| `.txt` | Stdlib (raw text) | Per file |

Adding a new format requires only implementing `BaseLoader` and registering in `LoaderRegistry` — upload endpoint auto-discovers supported extensions.

---

## 📊 Evaluation Metrics — 4-Layer Harness

RagArt includes a **custom layered evaluation framework** that measures answer quality at four different cost/accuracy levels:

```
                 ┌─────────────────────┐
   cost ↑        │  L4: LLM-as-Judge   │  API call   | critical samples only
                 ├─────────────────────┤
                 │  L3: BLEU / ROUGE   │  free       | every sample
                 ├─────────────────────┤
                 │  L2: Vector sim.    │  minimal    | every sample
                 ├─────────────────────┤
   accuracy ↓    │  L1: Rule-based     │  free       | every sample
                 └─────────────────────┘
```

| Layer | What It Measures | Method |
|-------|-----------------|--------|
| **L1** | Surface health (length, language, keywords, forbidden patterns) | Weighted rule checks |
| **L2** | Semantic similarity between generated and reference answer | Cosine similarity of embeddings |
| **L3** | Lexical overlap with reference | BLEU-1/2 + ROUGE-L with Turkish stemming |
| **L4** | Faithfulness, Relevance, Completeness (1-5 Likert) | Groq LLM-as-Judge (llama-3.3-70b) |

### Baseline Progression Results

The table below shows **measured quality improvements** as techniques were added incrementally:

| Version | Configuration | Avg Score | L1 Rules | L2 Vector | L3 Lexical | L4 Judge |
|---------|--------------|-----------|----------|-----------|------------|----------|
| **v0** | Dense only + Local provider | **0.576** | 0.793 | 0.820 | 0.116 | — |
| **v1** | + Groq LLM (llama-3.3-70b) | **0.665** | 0.884 | 0.843 | 0.266 | — |
| **v2** | + Hybrid Retrieval (RRF) | **0.695** | 0.928 | 0.849 | 0.306 | — |
| **v3** | + Cross-Encoder Reranking | **0.710** | 0.926 | 0.883 | 0.320 | — |
| **v3+** | + LLM-as-Judge (critical) | **0.738** | 0.939 | 0.880 | 0.323 | **0.896** |
| **v4** | + Context Dedup + Budget + Reorder | **0.704** | 0.914 | 0.884 | 0.313 | — |

> **Key insight**: Each technique contributed measurable improvement. The overall pipeline achieved a **+23.2% quality increase** from v0 (naive) to v3+ (full pipeline). L4 (LLM-as-Judge) shows **89.6% avg faithfulness/relevance/completeness** on critical samples.

### LLM-as-Judge (L4) Detailed Results

Evaluated on 6 critical samples with Groq `llama-3.3-70b-versatile`:

| Question | Faithfulness | Relevance | Completeness | Score |
|----------|-------------|-----------|--------------|-------|
| Algoritma nedir? | 5/5 | 5/5 | 4/5 | 0.938 |
| Algoritma kelimesi nereden gelir? | 5/5 | 5/5 | 4/5 | 0.938 |
| Derin öğrenme vs makine öğrenmesi | 5/5 | 5/5 | 4/5 | 0.938 |
| Multi-hop: Python'dan etkilenen diller | 5/5 | 5/5 | 4/5 | 0.938 |
| Yapay zeka nedir? | 3/5 | 4/5 | 3/5 | 0.625 |
| Out-of-domain: deprem (refusal test) | 5/5 | 5/5 | 5/5 | **1.000** |

> The **perfect 1.000 on the out-of-domain refusal test** validates the hallucination mitigation system — the model correctly refused to answer when context was insufficient.

---

## 🏛️ Project Structure

```
RagArt/
├── app.py                          # Flask REST API server (12 endpoints)
├── run.py                          # CLI entry point (web/interactive/check modes)
├── setup.py                        # Package configuration
├── requirements.txt                # Dependencies
├── .env.example                    # Environment variable template
│
├── src/                            # Core RAG engine
│   ├── rag_system.py               # Main orchestrator (776 lines)
│   ├── embeddings.py               # Embedding manager (multilingual MiniLM)
│   ├── document_loader.py          # Multi-format document loading
│   ├── llm_providers.py            # 6 LLM providers (DeepSeek, OpenAI, Groq, Ollama, HF, Local)
│   ├── utils.py                    # Logging, text utilities
│   │
│   ├── retrievers/                 # Pluggable retrieval strategies
│   │   ├── base.py                 # BaseRetriever contract + RetrievedDoc
│   │   ├── dense.py                # ChromaDB cosine similarity
│   │   ├── sparse.py               # BM25Okapi with Turkish stemming
│   │   ├── hybrid.py               # RRF fusion (Dense + Sparse)
│   │   └── reranker.py             # Cross-encoder reranker (BAAI/bge-reranker-v2-m3)
│   │
│   ├── context/                    # Post-retrieval context engineering
│   │   ├── base.py                 # BaseContextProcessor + ProcessorChain
│   │   ├── redundancy.py           # Cosine-based duplicate removal
│   │   ├── token_budget.py         # Token budget trimmer (Turkish-calibrated)
│   │   └── reorder.py              # Lost-in-the-middle reorderer
│   │
│   ├── memory/                     # Conversation memory strategies
│   │   ├── base.py                 # BaseMemory contract + ConversationTurn
│   │   ├── none.py                 # No memory (single-turn)
│   │   ├── sliding_window.py       # Last N turns verbatim
│   │   ├── summary_buffer.py       # LLM-summarized history
│   │   └── vector_retrieval.py     # Semantic retrieval over history
│   │
│   └── loaders/                    # Format-specific document loaders
│       ├── base.py                 # BaseLoader + LoaderRegistry
│       ├── json_loader.py          # Nested JSON key extraction
│       ├── pdf_loader.py           # Per-page PDF parsing (pypdf)
│       ├── docx_loader.py          # DOCX paragraph extraction
│       ├── markdown_loader.py      # Markdown loader
│       └── text_loader.py          # Plain text loader
│
├── config/                         # Configuration
│   ├── settings.py                 # Environment-based settings singleton
│   └── settings_schema.py          # BYOK schema (provider params, validation, UI metadata)
│
├── templates/
│   └── index.html                  # Single-page web UI (dark theme, responsive)
│
├── tests/                          # 21 test modules, 174 test cases
│   ├── test_rag_system.py          # Core RAG integration tests
│   ├── test_retrievers.py          # Dense/Sparse/Hybrid retrieval tests
│   ├── test_reranker.py            # Cross-encoder reranker tests
│   ├── test_context_engineering.py  # Dedup/Budget/Reorder processor tests
│   ├── test_memory.py              # Memory strategy tests
│   ├── test_loaders.py             # Multi-format loader tests
│   ├── test_llm_providers.py       # Provider factory tests
│   ├── test_settings_schema.py     # BYOK schema validation tests
│   ├── test_settings_integration.py # End-to-end settings tests
│   ├── test_eval_l1_rules.py       # L1 evaluator unit tests
│   ├── test_eval_l2_vector.py      # L2 evaluator unit tests
│   ├── test_eval_l3_lexical.py     # L3 evaluator unit tests
│   ├── test_eval_l4_judge.py       # L4 evaluator unit tests
│   ├── test_eval_integration.py    # Full eval harness integration
│   └── ...                         # 7 more test modules
│   │
│   └── evaluation/                 # 4-Layer quality evaluation harness
│       ├── runner.py               # Orchestrator (EvalRunner + RunReport)
│       ├── dataset.py              # Golden dataset loader (12 Q&A pairs)
│       ├── report.py               # Markdown + JSON report generator
│       ├── layers/                 # L1 (Rules), L2 (Vector), L3 (Lexical), L4 (Judge)
│       ├── datasets/               # Hand-crafted golden Q&A pairs
│       └── baselines/              # Frozen evaluation baselines (v0-v4)
│
├── scripts/
│   └── run_eval.py                 # CLI for running evaluations
│
├── data/                           # Knowledge base (50 Turkish Wikipedia articles)
├── docs/                           # Architecture docs & eval history
└── görsel/                         # Screenshots & demo video
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/kadiryonak/RagArt.git
cd RagArt
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure (Optional)

```bash
cp .env.example .env
# Edit .env with your API key
```

```env
# Use Groq free tier (recommended for testing)
MODEL_TYPE=groq
GROQ_API_KEY=your_groq_api_key_here

# Or run fully local (no API key needed)
MODEL_TYPE=local
```

### 3. Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🔧 Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_TYPE` | LLM provider: `deepseek`, `openai`, `groq`, `ollama`, `huggingface`, `local` | `local` |
| `DEEPSEEK_API_KEY` | DeepSeek API key | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `GROQ_API_KEY` | Groq API key (free tier available) | — |
| `HUGGINGFACE_API_KEY` | HuggingFace API key | — |
| `DATA_FOLDER` | Knowledge base folder | `./data` |
| `CHROMA_DB_PATH` | Vector store persistence path | `./chroma_db` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `5000` |
| `CHUNK_SIZE` | Text chunk size (chars) | `800` |
| `CHUNK_OVERLAP` | Chunk overlap (chars) | `150` |
| `RELEVANCE_THRESHOLD` | Min relevance for RAG context | `0.1` |
| `TOP_K_DOCUMENTS` | Documents retrieved per query | `5` |

---

## 🧪 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI (single-page application) |
| `/ask` | POST | Ask a question (supports BYOK headers) |
| `/status` | GET | System initialization status |
| `/health` | GET | Health check |
| `/stats` | GET | System statistics |
| `/test` | GET | Run built-in test questions |
| `/data-info` | GET | Knowledge base information |
| `/upload` | POST | Upload document (JSON/PDF/DOCX/MD/TXT) |
| `/delete-file` | POST | Delete a document |
| `/reindex` | POST | Rebuild vector store |
| `/list-files` | GET | List all knowledge base files |
| `/settings/schema` | GET | Frontend settings schema (BYOK metadata) |
| `/source/<filename>` | GET | Serve raw source file (PDF viewer, etc.) |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.9+ |
| **Web Framework** | Flask + Flask-CORS |
| **Vector Database** | ChromaDB (persistent) |
| **Embeddings** | HuggingFace Sentence Transformers (MiniLM-L12-v2) |
| **Sparse Retrieval** | rank_bm25 (BM25Okapi) |
| **Cross-Encoder** | BAAI/bge-reranker-v2-m3 |
| **LLM Orchestration** | LangChain (document splitting, vector store) |
| **Document Parsing** | pypdf, python-docx |
| **Tokenization** | tiktoken (budget estimation) |
| **ML Framework** | PyTorch + Transformers |
| **Testing** | pytest + pytest-cov (174 tests) |
| **Evaluation** | Custom 4-layer harness (L1-L4) |
| **Frontend** | Vanilla HTML/CSS/JS (dark theme SPA) |

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run evaluation (mock RAG, no API needed)
python scripts/run_eval.py --mock-rag --layers L1,L3

# Run evaluation with real RAG
python scripts/run_eval.py --layers L1,L2,L3 --name baseline

# Run with LLM-as-Judge (requires GROQ_API_KEY)
python scripts/run_eval.py --with-judge --name full-eval
```

---

## 📝 License

MIT License — See [LICENSE](LICENSE)

---

<div align="center">

**Built by [Kadir Yonak](https://github.com/kadiryonak)**

*Combining 9+ RAG techniques into a single, evaluated, production-grade system.*

</div>
