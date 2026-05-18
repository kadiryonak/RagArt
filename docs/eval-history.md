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

## v1 — Naive RAG + Groq LLM (2026-05-18)

**Branch:** `feat/baseline-v1-groq`
**Komut:** `python scripts/run_eval.py --provider groq --layers L1,L2,L3 --name baseline-v1-groq`
**Değişen:** Sadece LLM provider — local fallback → Groq `llama-3.3-70b-versatile`. Retrieval, embedder, chunking, prompt aynı.

### Özet

| Katman | v0 score | v1 score | Δ | v0 pass | v1 pass |
|---|---|---|---|---|---|
| L1 (rules)    | 0.793 | **0.884** | **+0.092** | 100% | 100% |
| L2 (vector)   | 0.820 | **0.843** | +0.023 | 92% | 83% |
| L3 (lexical)  | 0.116 | **0.266** | **+0.150** ⭐ | 25% | 58% |
| **Overall**   | **0.5762** | **0.6645** | **+0.0883** | 3/12 | 7/12 |

### Zorluk bazında

| Zorluk | v0 | v1 | Δ |
|---|---|---|---|
| easy   | 0.595 | 0.650 | +0.055 |
| medium | 0.558 | 0.679 | **+0.122** ← en büyük kategori kazancı |
| hard   | 0.575 | 0.665 | +0.089 |

### Per-item delta (en çok kazanan ve kaybeden)

| Item | v0 | v1 | Δ |
|---|---|---|---|
| **hard-01-python-kullanim-yerleri** | 0.481 | 0.798 | **+0.316** |
| **medium-02-python-isim-kaynagi**   | 0.512 | 0.763 | **+0.251** |
| **medium-04-derin-ogrenme**         | 0.521 | 0.714 | **+0.193** |
| **easy-01-algoritma-tanim**         | 0.612 | 0.773 | +0.161 |
| hard-03-multi-hop                   | 0.494 | 0.566 | +0.071 |
| medium-03-veri-yapilari             | 0.488 | 0.518 | +0.030 ← hâlâ retrieval bias |
| **edge-01-out-of-domain**           | 0.564 | 0.523 | **−0.041** ⚠ |

### Yorumlar

**1. Hipotez doğrulandı: cloud LLM, L3'te en büyük tekil kazancı verdi.**
v0 baseline analizinde "L3 düşüklüğü en büyük tek hedef, cloud LLM +0.20-0.30 bekleniyor" demiştik. **+0.150** geldi — beklenenin biraz altı ama yön doğru. Lexical pass oranı %25 → %58'e çıktı.

**2. L1 de iyileşti (+0.092).** Bu beklenmedik bir kazançtı: Groq cevapları daha keyword-rich + bad-pattern içermiyor (local LLM bazen kalıp "Bu konuda…" döndürüyordu).

**3. L2 (vector) pass rate aslında düştü (%92 → %83).** Average yükseldi ama eşik aşımı azaldı; bu, Groq'un bazı sorularda biraz farklı anlamsal yörünge takip ettiğini gösteriyor (örn. medium-03 hâlâ yanlış kaynak çekiyor → cevap context'inden uzaklaşıyor).

**4. medium kategorisi en büyük kazancı aldı (+0.122).** Medium sorular detay ister; Groq detaylı cevap üretiyor.

**5. edge-01 (out-of-domain) gerilemesi: ⚠️**
Local fallback "bilgi yok" diyebiliyordu; Groq "yetersiz veri ama genel bilgilerim şu…" diyerek yarı-halüsinasyon yapıyor. Bu, faithfulness sorununu işaret ediyor. **L4 (LLM judge)** bu örneği critical=true ile işaretliyor; gelecek branch'te ölçeceğiz.

**6. medium-03-veri-yapilari hâlâ failing.** Retrieval bias problemi: `Veri_bilimi.json` çekiliyor, `Veri_yapıları.json` olmalı. Cloud LLM bunu çözemiyor çünkü context yanlış. **Hybrid retrieval (BM25)** gerekli.

### İyileştirme hedefleri (güncellenmiş)

| Hedef | Beklenen kazanım | Niye? |
|---|---|---|
| `feat/hybrid-retrieval` | medium-03 +0.20, L2 pass +%10 | BM25 "veri yapıları" terimini kesin yakalar |
| `feat/reranker` | L2 +0.05, edge-01 daha iyi | Cross-encoder relevance |
| `feat/query-understanding` | edge-01 +0.20, halüsinasyon ↓ | OOD detection net red |
| `feat/memory` | Çoklu turn senaryolarında | Mevcut testset'i etkilemez |

### Rate limit notu

Groq free tier 12K TPM (token-per-minute). İlk çalıştırmada 3 hard soru 429'a takıldı. `GroqProvider` artık 429'da error mesajındaki "try again in Xs" ipucunu okuyup tek-seferlik backoff retry yapıyor (`MAX_RETRY_WAIT_S=75`). Test: `tests/test_groq_provider.py::TestRateLimitRetry`.

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
