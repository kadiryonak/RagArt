# Evaluation History

Bu doküman, her major branch'in eval harness skorlarını **baseline'a göre delta** olarak izler. Mimari kararları sayılarla destekleyen showcase için kritik.

## Skor okuma rehberi

- **L1 (rules):** Cevabın yüzeysel sağlığı. ≥0.6 → pass.
- **L2 (vector):** Anlamsal yakınlık. ≥0.65 → pass.
- **L3 (lexical):** Kelime örtüşmesi (BLEU + ROUGE). ≥0.25 → pass. Türkçe ve paraphrase için tipik olarak düşük.
- **L4 (judge):** LLM-as-a-judge (Groq). Sadece `critical=true` örnekler. ≥0.6 → pass.
- **Overall:** Çalıştırılan katmanların eşit ağırlıklı ortalaması.

Pass threshold geçilmemiş olması "kötü cevap" demek değil; "katmanın eşiğinin altı" demek. Karşılaştırma için **trend** ve **delta** önemli.

---

## v0 — Naive RAG baseline (2026-05-18)

**Branch:** `feat/baseline-measurement`
**Commit:** TBD
**Komut:** `python scripts/run_eval.py --layers L1,L2,L3 --name baseline --limit 12`
**Konfig:** `MODEL_TYPE=local`, embedder = `paraphrase-multilingual-MiniLM-L12-v2`, dense-only retrieval, top-k=5, chunk_size=800, chunk_overlap=150.

### Özet

| Katman | Avg Score | Pass Rate | Avg Latency |
|---|---|---|---|
| L1 (rules) | **0.793** | 100% | 0.2 ms |
| L2 (vector) | **0.820** | 92% | 109.9 ms |
| L3 (lexical) | **0.116** | 25% | 0.9 ms |
| **Overall** | **0.5762** | 3/12 PASS | — |

### Zorluk bazında

| Zorluk | N | Avg Score |
|---|---|---|
| easy | 4 | 0.595 |
| medium | 4 | 0.558 |
| hard | 4 | 0.575 |

### Pass/Fail dağılımı

```
PASS (3/12):
  easy-02-python-tanim             0.694
  medium-01-algoritma-tarih        0.711
  hard-02-algoritma-elestirisi     0.762

FAIL (9/12):  ortalama 0.526
  easy-01-algoritma-tanim          0.612
  easy-03-yapay-zeka               0.570
  medium-02-python-isim-kaynagi    0.512
  medium-03-veri-yapilari          0.488   ← retrieval bias: Veri_bilimi.json (Veri_yapıları.json olmalıydı)
  medium-04-derin-ogrenme          0.521
  hard-01-python-kullanim-yerleri  0.481
  hard-03-multi-hop                0.494
  edge-01-out-of-domain            0.564   ← out-of-domain reddetme zayıf
  edge-02-bos-soru                 0.504   ← boş soru reddetme zayıf
```

### Yorumlar

1. **L1 (rules) yüksek, L2 (vector) yüksek, L3 (lexical) çok düşük** → sistem semantik olarak iyi rota tutuyor ama kelime kelime referansa benzemiyor. Local LLM cevapları kısa ve template-bazlı; reference cevaplar uzun ve detaylı.
2. **L3 problemini çözecek değişiklikler:**
   - Cloud LLM (DeepSeek / Groq) ile detaylı cevap üretimi
   - Daha güçlü retrieval (hybrid + reranker) → daha iyi context → daha iyi cevap
3. **Retrieval bias örneği:** `medium-03-veri-yapilari` → `Veri_bilimi.json` kaynak gelmiş, oysa doğrusu `Veri_yapıları.json`. Embedding model "veri yapıları" ile "veri bilimi"ni karıştırıyor. **Hybrid retrieval (BM25)** bu tip exact-match sorunları çözer.
4. **Edge case zayıflığı:** Out-of-domain ve boş soru ideal red mesajı vermiyor. **Query understanding / input validation** katmanı gerekli.

### İyileştirme hedefleri (gelecek branch'ler)

| Branch | Beklenen kazanım | Niye? |
|---|---|---|
| `feat/settings` (cloud LLM seçeneği) | L3 +0.20-0.30 (en büyük tek kazanım) | Detaylı cevap → kelime örtüşmesi |
| `feat/hybrid-retrieval` | L2 +0.05, retrieval doğruluğu | BM25 exact-match sorunları çözer |
| `feat/reranker` | L2 +0.05, edge cases iyi | Cross-encoder relevance |
| `feat/context-engineering` | Genel +0.05 | Compression + ordering |
| `feat/query-understanding` | edge cases +0.20 | OOD detection |

---

## Şablon: yeni branch sonrası ekleme formatı

```markdown
## v<N> — <Branch adı> (<tarih>)

**Branch:** `feat/...`
**Commit:** abc1234
**Komut:** `python scripts/run_eval.py ...`
**Değişen:** Bu branch ne ekledi/değiştirdi.

### Özet

| Katman | Score | Δ vs v<N-1> | Pass |
|---|---|---|---|
| L1 | x.xxx | +/-x.xxx | y/12 |
| L2 | x.xxx | +/-x.xxx | y/12 |
| L3 | x.xxx | +/-x.xxx | y/12 |
| Overall | x.xxxx | +/-x.xxxx | y/12 |

### Önemli regression veya kazanımlar
- ...

### Bir sonraki hedef
- ...
```
