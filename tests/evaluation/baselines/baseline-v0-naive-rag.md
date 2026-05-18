# RAG Evaluation Report

_Generated: 2026-05-18 17:42:32_

## Özet

- **Toplam örnek:** 12
- **Toplam süre:** 1.67 sn
- **Çalıştırılan katmanlar:** L1_rules, L2_vector, L3_lexical
- **Ortalama skor:** 0.5762

## Katman bazında özet

| Katman | N | Ort. skor | Pass oranı | Ort. latency (ms) |
|---|---|---|---|---|
| L1_rules | 12 | 0.7928 | 100.00% | 0.2 |
| L2_vector | 12 | 0.8199 | 91.67% | 109.9 |
| L3_lexical | 12 | 0.1158 | 25.00% | 0.9 |

## Zorluk bazında özet

| Zorluk | N | Ort. skor |
|---|---|---|
| easy | 4 | 0.5951 |
| medium | 4 | 0.5579 |
| hard | 4 | 0.5755 |

## Örnek detayları

### ✗ `easy-01-algoritma-tanim` (skor: 0.612, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Algoritma nedir?

**Cevap:** Algorithm information found: [Source 1 - Algoritma.json] . Algoritmik yaklaşımda da çözüm için olası yöntemlerden en uygun olan seçilir ve yapılması gerekenler adım adım ortaya konulur. Algoritmayı belirtmek için; metinsel olarak düz ifade ve akış diyagramı olmak üzere 2 yöntem kullanılır. Algoritma…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.800 | ✓ | 0.9ms |
| L2_vector | 0.908 | ✓ | 128.1ms |
| L3_lexical | 0.129 | ✗ | 1.4ms |

### ✓ `easy-02-python-tanim` (skor: 0.694, zorluk: easy, kategori: factual)

**Soru:** Python nedir ve hangi alanlarda kullanılır?

**Cevap:** Definition: title: Python (programlama dili). content: Python, nesne yönelimli, yorumlamalı, birimsel (modüler) ve etkileşimli yüksek seviyeli bir programlama dilidir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.829 | ✓ | 112.7ms |
| L3_lexical | 0.253 | ✓ | 0.7ms |

### ✗ `easy-03-yapay-zeka` (skor: 0.570, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Yapay zeka nedir?

**Cevap:** Available information: [Source 1 - Yapay_zeka.json]
. Önde gelen yapay zeka ders kitapları, alanı zeki etmenlerin çalışması olarak tanımlar: Çevresini algılayan ve hedeflerine başarıyla ulaşma şansını en üst düzeye çıkaran ...

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.733 | ✓ | 0.1ms |
| L2_vector | 0.863 | ✓ | 122.4ms |
| L3_lexical | 0.114 | ✗ | 0.6ms |

### ✓ `medium-01-algoritma-tarih` (skor: 0.711, zorluk: medium, kategori: factual, **CRITICAL**)

**Soru:** Algoritma kelimesi nereden gelir ve ilk algoritma kim tarafından yazılmıştır?

**Cevap:** Algorithm information found: [Source 1 - Algoritma.json] Algoritma sözcüğü, Özbekistan'ın Harezm bölgesindeki Hive kentinde doğmuş olan Ebu Abdullah Muhammed İbn Musa el Harezmi'den gelir. Bu alim 9. yüzyılda cebir alanındaki algoritmik çalışmalarını kitaba dökerek matematiğe çok büyük bir katkı sağ…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.900 | ✓ | 0.3ms |
| L2_vector | 0.956 | ✓ | 193.8ms |
| L3_lexical | 0.276 | ✓ | 2.7ms |

### ✗ `medium-02-python-isim-kaynagi` (skor: 0.512, zorluk: medium, kategori: factual)

**Soru:** Python programlama dilinin ismi nereden gelir?

**Cevap:** Definition: Python programlama dilinin tasarımı ve felsefesi, diğer birçok programlama diline de ilham vermiş veya bu dillere etki etmiştir:. title: Python (programlama dili)

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.600 | ✓ | 0.1ms |
| L2_vector | 0.869 | ✓ | 119.2ms |
| L3_lexical | 0.066 | ✗ | 0.7ms |

### ✗ `medium-03-veri-yapilari` (skor: 0.489, zorluk: medium, kategori: factual)

**Soru:** Veri yapıları nedir ve hangi temel türleri vardır?

**Cevap:** Available information: [Source 1 - Veri_bilimi.json]
Veri bilimi ile ilgili önemli bazı kavramlardan;
Yapılandırılmış veri, en basit anlamıyla bir Excel tablosu olarak düşünülebilir. Başka bir deyişle buradaki her bir sütun...

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.600 | ✓ | 0.1ms |
| L2_vector | 0.818 | ✓ | 112.5ms |
| L3_lexical | 0.048 | ✗ | 0.8ms |

### ✗ `medium-04-derin-ogrenme` (skor: 0.521, zorluk: medium, kategori: analytical, **CRITICAL**)

**Soru:** Derin öğrenme ile makine öğrenmesi arasındaki fark nedir?

**Cevap:** Definition: title: Derin öğrenme. title: Makine öğrenmesi

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.800 | ✓ | 0.1ms |
| L2_vector | 0.706 | ✓ | 116.6ms |
| L3_lexical | 0.056 | ✗ | 0.6ms |

### ✗ `hard-01-python-kullanim-yerleri` (skor: 0.481, zorluk: hard, kategori: synthesis)

**Soru:** Hangi büyük kurumlar ve projeler Python kullanır?

**Cevap:** Found items: title: Python (programlama dili), content: Python, nesne yönelimli, yorumlamalı, birimsel (modüler) ve etkileşimli yüksek seviyeli bir programlama dilidir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.600 | ✓ | 0.1ms |
| L2_vector | 0.792 | ✓ | 111.8ms |
| L3_lexical | 0.052 | ✗ | 0.6ms |

### ✓ `hard-02-algoritma-elestirisi` (skor: 0.762, zorluk: hard, kategori: analytical)

**Soru:** Algoritmalara yöneltilen toplumsal eleştiriler nelerdir?

**Cevap:** Algorithm information found: [Source 1 - Algoritma.json] Algoritmalara eleştirel yaklaşımlar  Algoritmaların kullanımı hayatın her alanında giderek yaygınlaşmaktadır. İş yerlerindeki performans değerlendirmelerinden bankaların kime kredi vereceğine, güvenlik sistemlerinden sosyal medya platformların…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.2ms |
| L2_vector | 0.920 | ✓ | 101.0ms |
| L3_lexical | 0.366 | ✓ | 1.5ms |

### ✗ `hard-03-multi-hop` (skor: 0.494, zorluk: hard, kategori: synthesis, **CRITICAL**)

**Soru:** Python'dan etkilenen programlama dilleri hangileridir ve bu diller hangi özellikleri ödünç almıştır?

**Cevap:** Definition: Python programlama dilinin tasarımı ve felsefesi, diğer birçok programlama diline de ilham vermiş veya bu dillere etki etmiştir:. 

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.600 | ✓ | 0.1ms |
| L2_vector | 0.852 | ✓ | 110.7ms |
| L3_lexical | 0.031 | ✗ | 0.7ms |

### ✗ `edge-01-out-of-domain` (skor: 0.564, zorluk: hard, kategori: edge_case, **CRITICAL**)

**Soru:** Türkiye'de 2023 deprem felaketinin etkileri nelerdir?

**Cevap:** Insufficient information in the knowledge base. Relevance score: 0.033. Consider expanding your data.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.940 | ✓ | 0.1ms |
| L2_vector | 0.752 | ✓ | 45.9ms |
| L3_lexical | 0.000 | ✗ | 0.4ms |

### ✗ `edge-02-bos-soru` (skor: 0.504, zorluk: easy, kategori: edge_case)

**Soru:** ?

**Cevap:** Insufficient information in the knowledge base. Relevance score: 0.000. Consider expanding your data.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.940 | ✓ | 0.1ms |
| L2_vector | 0.573 | ✗ | 43.9ms |
| L3_lexical | 0.000 | ✗ | 0.2ms |
