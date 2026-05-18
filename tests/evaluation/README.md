# Layered Evaluation Harness

Bu klasör, RAG sisteminin **kalite ölçümünü** üstlenen değerlendirme altyapısıdır. Standart yazılım testlerinden (unit/integration/e2e — `tests/`'te) ayrıdır: testler "kod çalışıyor mu?", eval ise "cevap iyi mi?" sorusunu cevaplar.

## Felsefe: Katmanlı yaklaşım

Her sorgu için 4 farklı maliyet/hassasiyet seviyesinde değerlendirme yapılır. Daha pahalı katmanlar daha az çalıştırılır.

```
                 ┌─────────────────────┐
   maliyet ↑    │  L4: LLM-as-Judge   │  dk    | API call  | sadece kritik örnek
                ├─────────────────────┤
                │  L3: BLEU / ROUGE   │  sn    | bedava    | her örnek
                ├─────────────────────┤
                │  L2: Vector sim.    │  sn    | minimal   | her örnek
                ├─────────────────────┤
   hassasiyet ↓ │  L1: Rule-based     │  ms    | bedava    | her örnek
                 └─────────────────────┘
```

## Katmanların detayı

### L1 — Rule-based (kural tabanlı)

**Ne ölçer?** Cevabın "yüzeysel sağlığı" — formatı, dili, anahtar kelime varlığı, yasaklı kalıplar.

**Nasıl çalışır?** 4 alt-check'in ağırlıklı toplamı:
- **Uzunluk** (%20): Çok kısa (<20 char) veya çok uzun (>4000) cevaplar cezalandırılır.
- **Dil** (%20): Türkçe soru → Türkçe cevap (Türkçeye özel karakterler oranı).
- **Anahtar kelime** (%40): Golden item'da listelenen `keywords` cevapta geçiyor mu.
- **Yasaklı kalıp** (%20): "I don't know", "as an AI" gibi tipik başarısızlık ifadelerinin yokluğu.

**Ne zaman?** Her test koşumunda. Sıfıra yakın skor → pipeline bozuk.

**Sınırı:** Format doğru olsa bile cevap yanlış olabilir; L1 başarı SEMANTİK doğruluk garantilemez.

### L2 — Vector similarity (anlamsal)

**Ne ölçer?** Üretilen cevap ile referans cevap arasındaki **anlamsal yakınlık**. Kelime tutmasa bile aynı anlamı taşıyorsa yüksek skor verir.

**Nasıl çalışır?**
1. Üretilen cevap ve referans aynı embedding modeli ile vektöre çevrilir.
2. Cosine similarity hesaplanır: `cos(θ) = (u·v) / (||u||·||v||)`
3. `[-1, 1]` → `[0, 1]`'e normalize: `(cos + 1) / 2`

İsteğe bağlı: cevap ↔ retrieved_context benzerliği (groundedness sinyali).

**Ne zaman?** Genel kalite ölçümünde her zaman. Mevcut embedder zaten yüklü olduğundan ek maliyet minimal.

**Sınırı:** Embedding modeli kalitesine bağlı. Multilingual modellerle Türkçe için "iyi ama mükemmel değil" sonuç verir.

### L3 — Lexical (BLEU + ROUGE-L)

**Ne ölçer?** Üretilen cevap ile referans arasındaki **kelime örtüşmesi**. Anlamı değil, "kelimeleri ne kadar tuttu" sorusunu cevaplar.

**Nasıl çalışır?**
- **BLEU-n**: cevaptaki n-gram'ların referansta bulunma oranı. Brevity penalty ile kısa cevaplar cezalandırılır.
- **ROUGE-L**: en uzun ortak alt dizi (longest common subsequence) bazlı F1.

Bizim skor: `0.4·BLEU-1 + 0.2·BLEU-2 + 0.4·ROUGE-L_F1`

**Türkçe ipucu:** Hafif suffix stripping uygulanır (`algoritmaları` → `algoritma`) — çekim cezasını azaltır.

**Ne zaman?** Her test koşumunda. Hızlı, deterministik, sıfır dış bağımlılık.

**Sınırı:** Parafraz cezalandırılır ("oto" ≠ "araba"). Sadece L3'e güvenme.

### L4 — LLM-as-a-Judge (Groq)

**Ne ölçer?** Üç boyut, her biri 1–5 Likert ile değerlendirilir:
1. **Faithfulness** — Cevap, bağlamdaki bilgilere sadık mı? (Halüsinasyon detektörü)
2. **Relevance** — Cevap, soruyla alakalı mı?
3. **Completeness** — Cevap, referansa göre yeterli mi?

**Nasıl çalışır?** Tek bir Groq chat completion isteği, JSON formatında skor döndürür. Default model: `llama-3.3-70b-versatile` (Groq free tier).

**Ne zaman?** **SADECE** `critical=True` işaretli golden item'larda + major branch baseline/final ölçümlerinde. Her commit'te değil.

**Sınırı:**
- LLM judge tutarsızlığı (%5-10 varyans).
- Self-bias: LLM kendi tarzına yakın cevaplara yüksek puan verebilir.
- Groq rate limit (free tier).

**Config:** `GROQ_API_KEY` env değişkeni yoksa katman atlanır (hata değil).

## Kullanım

### Hızlı sanity check (gerçek RAG'a ihtiyaç yok)

```bash
python scripts/run_eval.py --mock-rag --layers L1,L3
```

Bu komut RAG sistemini başlatmaz, mock bir RAG ile harness'ı doğrular. Sadece L1+L3 (dış bağımlılığı sıfır olanlar) çalışır.

### Gerçek RAG ile baseline

```bash
python scripts/run_eval.py --layers L1,L2,L3 --name baseline
```

L4 (LLM judge) hariç tüm katmanlar. Çıktı: `evaluation/reports/baseline_<timestamp>.{md,json}`.

### Critical örnekler için LLM judge dahil

```bash
export GROQ_API_KEY=...   # PowerShell: $env:GROQ_API_KEY="..."
python scripts/run_eval.py --with-judge --name baseline-with-judge
```

L4 sadece `critical=true` olan örneklerde çalışır.

### CLI seçenekleri

| Bayrak | Açıklama |
|---|---|
| `--dataset PATH` | Alternatif golden dataset (default: `evaluation/datasets/golden_qa.json`) |
| `--layers L1,L2,L3` | Çalıştırılacak katmanlar (virgülle ayrılmış) |
| `--with-judge` | L4'ü dahil et (`GROQ_API_KEY` gerekli) |
| `--limit N` | İlk N örneği değerlendir (debug için) |
| `--name baseline` | Rapor dosya öneki |
| `--mock-rag` | RAG yerine mock kullan (harness sanity check) |

## Golden dataset

`evaluation/datasets/golden_qa.json` — 12 hand-crafted Türkçe Q&A çifti, üç zorluk seviyesinde ve dört kategoride. Her item:

| Alan | Tip | Açıklama |
|---|---|---|
| `id` | str | Benzersiz ID |
| `question` | str | Sorulacak soru |
| `reference_answer` | str | İnsan tarafından yazılmış ideal cevap |
| `keywords` | list[str] | L1'in arayacağı anahtar terimler |
| `expected_sources` | list[str] | Bu sorunun cevabı hangi JSON'larda olmalı (gelecekte retrieval eval) |
| `difficulty` | easy/medium/hard | |
| `category` | factual/analytical/synthesis/edge_case | |
| `critical` | bool | L4'ün çalışacağı kritik örnekler |

## Rapor formatı

Her çalıştırma iki dosya üretir:

- **Markdown** (`*.md`): İnsan dostu özet — toplam skor, katman bazında, zorluk bazında, her örneğin detayı.
- **JSON** (`*.json`): Otomasyon için tam yapı. CI gerilemesi yakalama, blog grafikleri vb. için işlenebilir.

## Pyramid'deki yeri

```
                   /\
                  /  \      Playwright e2e (en sonda, ayrı branch)
                 /----\
                /      \    pytest integration (her feature branch'te)
               /--------\
              /          \  pytest unit (her feature branch'te)
             /------------\
            /              \
           / ============   \   ← Bu klasör (kalite metrikleri)
          /    EVALUATION    \
         /     (L1-L4)        \
        /-----------------------\
```

- **Unit/integration/e2e** → kod doğruluğu (CI'da otomatik)
- **Eval (L1-L4)** → cevap kalitesi (geliştirme döngüsünde, branch baseline'larda)
- **RAGAS** → major release'lerde tek seferlik, "endüstri standardı" rapor olarak (henüz eklenmedi)
