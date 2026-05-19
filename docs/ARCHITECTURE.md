# RagArt — Mimari ve Stratejik Yol Haritası

> **Vizyon:** Açık kaynak, modüler, provider-agnostic, observable bir RAG/agent framework'ü.
> Hedef kullanıcı: kendi API key'iyle ve kendi tercihiyle (cloud / Ollama / local) çalıştırabilen,
> ileri RAG tekniklerini config'le açıp kapatabilen geliştiriciler.

Bu doküman, "daha iyi model kullan"ın ötesinde — production'da gerçekten fark yaratan
**7 alandaki stratejiyi** ve framework için **mimari prensipleri** tanımlar.

---

## 0. Mimari prensipler

Her yeni özellik bu prensiplere göre değerlendirilir:

| Prensip | Anlamı | Nasıl uygularız? |
|---|---|---|
| **Modüler** | Her teknik ayrı bir plugin | `BaseRetriever`, `BaseEmbedder`, `BaseReranker`, `BaseMemory`, `BaseAgent` |
| **Provider-agnostic** | LLM/VectorDB/Embedder swap edilebilir | Fabrika pattern + config-driven init |
| **Config-driven** | Kod değiştirmeden davranış değişimi | YAML/JSON pipeline tanımı + env vars |
| **Observable** | Her adımı görebilmek | Structured logging, trace per request, debug UI |
| **Local-first** | İnternetsiz çalışsın | Ollama, sentence-transformers, ChromaDB embed-only mode |
| **Reproducible** | Aynı config → aynı sonuç | Seed control, deterministic chunking, versioned datasets |
| **BYOK** | Kullanıcı kendi key'iyle gelir | Frontend-only key storage, server stateless |

---

## 1. Orchestration

**Durum:** Şu an tek-shot pipeline (retrieve → prompt → LLM). Branching, conditional logic, multi-hop yok.

**Strateji — kademeli:**

| Aşama | Yaklaşım | Ne zaman |
|---|---|---|
| **v1: Static pipeline** | YAML'da tanımlı sıralı adımlar (retrieve → rerank → prompt → llm) | Mevcut + küçük genişletme |
| **v2: Branching pipeline** | Soru türüne göre route (factual vs. analytical → farklı pipeline) | Settings UI bittikten sonra |
| **v3: Agent loop (ReAct)** | LLM araç seçer, tekrar ararsa multi-hop | Reranker + hybrid tamamlandıktan sonra |
| **v4: Graph orchestration** | LangGraph tarzı node graph + state | İhtiyaç oluşursa |

**Pipeline tanımı (hedef format):**

```yaml
# pipelines/default.yaml
nodes:
  - id: history_rewrite
    type: rewriter
    when: "context.history|length > 0"
  - id: hybrid_retrieve
    type: hybrid_retriever
    params: { dense_weight: 0.6, sparse_weight: 0.4, top_k: 50 }
  - id: rerank
    type: cross_encoder
    params: { model: "bge-reranker-v2-m3", top_k: 5 }
  - id: compress
    type: extractive_compress
    params: { max_tokens: 2000 }
  - id: generate
    type: llm
    params: { stream: true }
```

**Tradeoff:** Esneklik ↔ debug zorluğu. v3'e kadar static pipeline tutmak öğrenme/debug için daha iyi.

---

## 2. Retrieval engineering

**Durum:** Dense-only (Chroma + multilingual MiniLM), top-k=5, fixed chunk size.

**Strateji — öncelik sırası:**

### A. Hybrid retrieval (öncelik: yüksek)
- Dense (embedding) + Sparse (BM25) + RRF (Reciprocal Rank Fusion)
- **Niye?** Embedding semantik iyi ama exact match (isim, kod, kısaltma) kötü. BM25 tam tersi.
- **Kod boyutu:** ~80 satır. `rank_bm25` (50KB dep).

### B. Multi-stage retrieval (öncelik: yüksek)
```
[Query] → [Fast ANN (top-50)] → [Metadata filter] → [Rerank → top-5] → [Compress] → [Generate]
```
- **Reranker:** `bge-reranker-v2-m3` (multilingual, 400MB, CPU OK).
- **Niye?** Bi-encoder hızlı ama coarse; cross-encoder yavaş ama precise. Top-50'yi top-5'e indir.

### C. Query rewriting (öncelik: orta)
- LLM kullanıcı sorgusunu "retrieval için" optimize eder.
- Örnek: `"user auth issue"` → `"JWT token expiration middleware problem"`
- **Niye?** Kısa/belirsiz sorgular recall'u öldürür.

### D. Multi-query / RAG-Fusion (öncelik: orta)
- LLM aynı sorgunun 3-5 varyantını üretir, hepsiyle ara, RRF ile birleştir.
- **Niye?** Tek query'nin ıskaladığı belge varyantlarla yakalanır.

### E. Parent-child retrieval (öncelik: yüksek-orta)
- Küçük chunk'lar üzerinden ara (precise match), büyük parent chunk'ları dön (semantic context).
- **Niye?** Production'da çok güçlü — precision + context aynı anda.

### F. Hierarchical retrieval (öncelik: düşük)
- Document → Section → Chunk seviyesinde kademeli arama.
- **Niye?** Büyük dokümanlar için (PDF, kitap). Mevcut JSON yapısı için kısmen ihtiyaç.

### G. Self-query retrieval (öncelik: düşük)
- LLM metadata filter üretir: `{"category": "factual", "source": "Algoritma.json"}`
- **Niye?** Yapılandırılmış sorgular için. Önce metadata zenginleştir.

**Plugin contract:**

```python
class BaseRetriever(ABC):
    def retrieve(self, query: str, k: int, filters: dict | None = None) -> list[RetrievedDoc]: ...
    def supports_filters(self) -> bool: ...

class HybridRetriever(BaseRetriever):
    def __init__(self, dense: BaseRetriever, sparse: BaseRetriever, k_rrf: int = 60): ...
```

---

## 3. Context engineering

**Durum:** Retrieved chunk'lar olduğu gibi sistem prompt'a yapıştırılıyor. Sıralama yok, compression yok.

**Strateji:**

### A. Context compression
- **Extractive:** Cümle bazında relevance score, top sentences only.
- **Semantic filtering:** Query ile cosine sim < threshold cümleleri at.
- **Redundancy removal:** Yüksek benzerlikli chunk'ları dedupe et.
- **LLM summarization (opt-in):** Çok büyük context için özet (latency maliyeti var).

### B. Context ordering (lost-in-the-middle)
- En relevant chunk'ları **başa ve sona** koy.
- Modeller orta kısımdaki bilgiyi unutur (Anthropic/LongChat research).

### C. Dynamic context allocation
- Basit soru → 1-2 chunk, küçük prompt.
- Research query → 5-10 chunk, büyük prompt.
- Routing: query complexity classifier (heuristic veya küçük LLM).

### D. Token budgeting
- Her provider'ın context limiti farklı. Otomatik kırp.
- Maliyet hesabı: input + expected output toplamı limit'i aşmasın.

**Plugin contract:**

```python
class BaseContextProcessor(ABC):
    def process(self, query: str, docs: list[RetrievedDoc], budget: int) -> str: ...
```

---

## 4. Memory management

**Durum:** Memory yok — her soru izole.

**Strateji — 4 farklı memory tipi:**

| Tip | Ne saklar? | Nasıl çalışır? | Maliyet |
|---|---|---|---|
| **Short-term (conversation buffer)** | Son N turn | Liste, FIFO | Düşük |
| **Summary buffer** | Tüm sohbetin LLM özeti | Her N turn'de bir özet güncelle | Orta (LLM call) |
| **Vector retrieval** | Tüm önceki turn'ler embed'lenir | Yeni sorudan benzer turn'leri çek | Orta |
| **Token-budget aware** | Token sınırına göre dinamik | Önemli turn'leri tut, eskilerini özetle | Karma |

**Kullanıcı settings'ten seçebilir:**
- `memory.strategy`: `none` | `window` | `summary` | `vector` | `budget`
- `memory.window_size`: 5 (default)
- `memory.summary_threshold`: 10 turn

**Plugin contract:**

```python
class BaseMemory(ABC):
    def add(self, turn: ConversationTurn) -> None: ...
    def get_context(self, query: str, max_tokens: int) -> str: ...
    def clear(self) -> None: ...
```

**Long-term/episodic/semantic** memory (agent için): v3'te.

---

## 5. Latency optimization

**Durum:** Sync, sıralı, batching yok, cache yok, streaming yok.

**Strateji — öncelik sırasıyla:**

### A. Streaming generation (öncelik: yüksek)
- SSE (Server-Sent Events) ile token-token akıt.
- Kullanıcı TTFT (Time To First Token) ile mutlu olur.

### B. Async orchestration (öncelik: yüksek)
- Retrieval paralel: dense + sparse aynı anda.
- Tool calls (agent v3) paralel.
- Flask → Quart veya FastAPI'ye geçiş düşünülmeli (uzun vadede).

### C. Embedding batching
- Reindex sırasında 32-64'lük batch'ler.
- Mevcut tek-tek embedding 10-20x yavaş.

### D. KV cache reuse (provider-bağımlı)
- OpenAI/Anthropic prompt caching API'leri.
- Sabit system prompt + değişken user message yapısı.

### E. Speculative decoding (uzun vade)
- Küçük model tahmin, büyük model doğrular.
- Provider-side feature (Ollama yakında destekleyecek).

**Ölçüm:** Her endpoint için p50/p95/p99 latency, structured trace.

---

## 6. Caching

**Durum:** Hiç yok.

**Strateji — katmanlı cache:**

| Katman | Ne cache'lenir? | Anahtar | TTL | Niye? |
|---|---|---|---|---|
| **Embedding cache** | `(model, text) → vector` | sha256(text) | ∞ | Aynı dokümanı tekrar embed'leme |
| **Retrieval cache** | `(query, params) → docs` | hash | 1h | Sık sorulan sorular |
| **Semantic cache** | Benzer query → önceki cevap | embedding sim > 0.95 | 1d | Production'da %20-40 hit oranı |
| **LLM response cache** | Aynı prompt → aynı cevap | sha256(prompt+params) | 1d | Deterministic temp=0 isteklerde |
| **Tool call cache** (agent) | `(tool, args) → result` | hash | tool-specific | Web search, API call'lar |

**Backend:** İlk versiyon SQLite (zero-dep). Production'da Redis swap edilebilir (plugin).

**Plugin contract:**

```python
class BaseCache(ABC):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
```

**Önemli:** Semantic cache her zaman opt-in olmalı (yanlış hit'ler kötü UX).

---

## 7. Evaluation

**Durum:** L1-L4 katmanlı harness (`tests/evaluation/`) tamamlandı. Golden dataset (12 örnek). RAGAS henüz yok.

**Strateji:**

### A. Retrieval evaluation (eklenecek)
- **Recall@k:** Doğru kaynak top-k'da var mı?
- **Precision@k:** Top-k'daki belgelerin yüzde kaçı relevant?
- **nDCG:** Sıralama kalitesi (ideal sıraya göre).
- **MRR (Mean Reciprocal Rank):** İlk doğru sonucun yeri.

### B. Generation evaluation (mevcut + iyileştirme)
- L1 (rules) ✓, L2 (vector) ✓, L3 (BLEU/ROUGE) ✓, L4 (Groq judge) ✓
- **Faithfulness, groundedness, hallucination detection:** L4'te ölçülüyor.
- **Answer relevance:** L4'te.

### C. Agent evaluation (v3)
- Tool correctness (doğru tool çağrıldı mı?)
- Planning quality (plan adımları mantıklı mı?)
- Task completion (sonuç ulaşıldı mı?)

### D. RAGAS entegrasyonu (major milestone'larda)
- Sadece release-candidate baseline'larda.
- Industry-standard rapor olarak README'de.

### E. Eval dashboard (UI)
- Her run'ın sonucu UI'da: skor trendi, hangi sorularda regression, hangi katman zayıf.
- Production framework için kritik özellik.

---

## 8. Plugin contracts (özet)

```python
# Embedding
class BaseEmbedder(ABC):
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: ...

# Retriever (dense, sparse, hybrid, parent-child, ...)
class BaseRetriever(ABC):
    def retrieve(self, query: str, k: int, filters: dict | None = None) -> list[RetrievedDoc]: ...

# Reranker
class BaseReranker(ABC):
    def rerank(self, query: str, docs: list[RetrievedDoc], top_k: int) -> list[RetrievedDoc]: ...

# Context processor
class BaseContextProcessor(ABC):
    def process(self, query: str, docs: list[RetrievedDoc], budget: int) -> str: ...

# Memory
class BaseMemory(ABC):
    def add(self, turn: ConversationTurn) -> None: ...
    def get_context(self, query: str, max_tokens: int) -> str: ...

# LLM provider
class BaseLLMProvider(ABC):
    def generate(self, prompt: str, **params) -> str: ...
    def stream(self, prompt: str, **params) -> Iterator[str]: ...

# Cache
class BaseCache(ABC):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...

# Evaluator
class BaseEvaluator(ABC):
    def evaluate(self, item: GoldenItem, output: RAGOutput) -> LayerResult: ...

# Agent (v3)
class BaseAgent(ABC):
    def run(self, query: str, tools: list[Tool], memory: BaseMemory) -> AgentResult: ...
```

---

## 9. Tech stack kararları

| Bileşen | Seçim | Niye? | Alternatif |
|---|---|---|---|
| **Web framework** | Flask (kısa vade), FastAPI (uzun vade) | Mevcut, hızlı; FastAPI async + auto-docs için | Quart, Litestar |
| **Vector DB** | ChromaDB (default), Qdrant plugin | Embed-only mode ile local-first | Qdrant, Pinecone, Weaviate |
| **Embedder** | sentence-transformers (local) + OpenAI/Cohere (cloud plugin) | Türkçe için multilingual MiniLM | bge-m3, e5-mistral |
| **Reranker** | bge-reranker-v2-m3 | Multilingual, CPU-friendly | cohere-rerank, jina-reranker |
| **LLM (cloud)** | DeepSeek, OpenAI, Anthropic, Groq | BYOK, free tier var | Together, Replicate |
| **LLM (local)** | Ollama | Plug-and-play, GGUF | llama.cpp, vLLM |
| **Sparse retrieval** | rank_bm25 | Single-file dep | Tantivy, Elasticsearch |
| **Cache** | SQLite (default), Redis (plugin) | Zero-dep başlangıç | DiskCache, Memcached |
| **Eval (custom)** | Pure Python L1-L4 (mevcut) | Bağımlılık-bağımsız | RAGAS, DeepEval (eklenecek) |
| **Eval (industry)** | RAGAS (milestone'larda) | Standard, README için | DeepEval, TruLens |
| **UI** | Vanilla JS + CSS vars (mevcut) | Build-step-free, açık kaynak için ideal | React/Vue (gerekirse) |
| **Frontend bundling** | Hiç | Tek HTML, kolay deploy | Vite (UI büyürse) |
| **Pipeline config** | YAML | Okunabilir, standart | TOML, JSON |
| **Observability** | structlog + Flask request logs | Structured logging | OpenTelemetry (uzun vade) |
| **Testing** | pytest + unittest.mock | Standart | hypothesis (property-based) |
| **E2E** | Playwright | Modern, hızlı, cross-browser | Cypress, Selenium |

---

## 10. Roadmap

### Kısa vade (mevcut sprint + 2-3 branch)
- [x] Layered eval harness (L1-L4) — `feat/eval-harness`
- [ ] Settings UI: API key, provider, LLM params — `feat/settings`
- [ ] Conversation memory (4 strategies) — `feat/memory`
- [ ] BYOK: frontend-only key — `feat/byok` (settings ile birleşebilir)

### Orta vade
- [ ] Hybrid retrieval (BM25 + dense + RRF) — `feat/hybrid-retrieval`
- [ ] Reranker (bge-reranker-v2-m3) — `feat/reranker`
- [ ] Streaming (SSE) — `feat/streaming`
- [ ] Embedding + retrieval + response cache (SQLite) — `feat/caching`
- [ ] Context compression + ordering — `feat/context-engineering`
- [ ] Query rewriting + multi-query — `feat/query-expansion`
- [ ] Pipeline config (YAML) — `feat/pipeline-config`
- [ ] E2E Playwright suite — `test/e2e-playwright`

### Tamamlanan (workspaces + vector DB)
- [x] **Workspaces** (NotebookLM-style) — `feat/workspaces-and-vector-db`
  - Her workspace izole: kendi dosya klasörü + kendi vector DB
  - `data/workspaces/{id}/` + `data/workspaces/{id}/meta.json`
  - Lazy RAG-per-workspace cache; embedder paylaşılır
  - Legacy `data/*.json` otomatik default workspace'e migrate
- [x] **BaseVectorStore plugin family** — Chroma + Qdrant adapters
  - Plugin sözleşmesi: `upsert_documents`, `similarity_search`, `count`,
    `delete_collection`
  - VectorStoreFactory ile UI dropdown'u otomatik üretiliyor
  - Yeni DB eklemek: BaseVectorStore implement et + Factory'ye register

### Sıradaki — Prompt engineering hooks (mimari, ileride implementasyon)

**Vizyon:** Kullanıcı UI'dan farklı prompt stratejilerini seçsin
(zero-shot, few-shot, chain-of-thought, self-consistency, ReAct...) veya
kendi system prompt'larını yönetebilsin.

**Plugin contract (taslak):**
```python
class BasePromptStrategy(ABC):
    name: str
    description_tr: str

    @abstractmethod
    def build(
        self,
        *,
        question: str,
        context: str,
        memory_context: str = "",
        history: list = None,
    ) -> str:
        """Final LLM prompt string'i üret."""

    def metadata(self) -> dict:
        """UI için açıklayıcı bilgi (temperature önerisi vb.)"""
        return {}
```

**Önceden tanımlı stratejiler (ileride):**
| ID | İsim | Açıklama |
|---|---|---|
| `direct` | Direkt cevap | Mevcut TURKISH_SYSTEM_PROMPT (default) |
| `few_shot` | Birkaç örnekli | Verilen 2-3 örnek Q&A + sonra asıl soru |
| `cot` | Adım adım düşün (CoT) | "Önce mantığını yaz, sonra cevabı ver" |
| `self_consistency` | Tutarlılık | N×CoT cevap üret, çoğunluğu seç |
| `react` | ReAct (Reasoning + Acting) | Tool routing için (Agentic RAG'da kullanılır) |
| `custom` | Kullanıcı tanımlı | UI'dan Markdown template yazılabilir, `{context}` `{question}` placeholder'ları |

**API tasarımı:**
- Header: `X-Prompt-Strategy: cot`
- Header: `X-Custom-Prompt: <base64-encoded template>` (custom için)
- Workspace meta'sına opsiyonel `default_prompt_strategy: str` alanı

**Eval etkisi (tahmin):**
- CoT: medium/hard kategorisinde L4 (judge) skorunda +5-10%
- Few-shot: easy kategorisinde L3 (lexical) +10-15% (referans örneğine yakın yazım)
- Self-consistency: faithfulness +5% (multi-vote effect)

**Şu an'ki implementation durumu:** YOK. Sadece architecture doc'ta plan.
Plugin pattern'i hazır olduğu için (BaseRetriever, BaseMemory, vs.) yeni
plugin family eklemek 1-2 saatlik iş.

### Uzun vade
- [ ] **Prompt engineering strategies** — `feat/prompt-strategies` (bu doc'ta sketch'i var)
- [ ] Agent loop (ReAct + tools) — `feat/agent`
- [ ] Parent-child retrieval — `feat/parent-child`
- [ ] Eval dashboard (UI) — `feat/eval-dashboard`
- [ ] RAGAS integration — `feat/ragas`
- [ ] Multi-tenant + auth (opsiyonel) — `feat/multi-tenant`
- [ ] OpenTelemetry tracing — `feat/observability`
- [ ] FastAPI migration — `chore/fastapi-migration`
- [ ] Docker compose — `chore/docker`

---

## 11. Karar günlüğü (ADR — Architecture Decision Records)

Önemli mimari kararlar için `docs/adr/NNN-title.md` formatında ayrı dosyalar tutulur:
- ADR-001: Why ChromaDB over Qdrant for v1?
- ADR-002: Why Flask over FastAPI for v1?
- ADR-003: Why frontend-only BYOK?
- ADR-004: Why layered evaluation (L1-L4) over single LLM judge?

Bu format, gelecekteki katkıcıların **niye böyle yapıldı?** sorusunu kolayca cevaplamasını sağlar.

---

## 12. Katkı yapmak isteyenler için

Her feature branch şu yapıyı içermeli:
1. **Implementation** — kod (modüler, plugin contract'a uygun).
2. **Unit tests** — `tests/test_<feature>.py`.
3. **Integration test** — full pipeline'da çalışıyor mu?
4. **Eval impact** — `tests/evaluation/` ile baseline'a göre delta. Regresyon varsa açıklama.
5. **Docs update** — bu doküman + README.
6. **ADR (varsa)** — ciddi mimari karar varsa kayıt.
