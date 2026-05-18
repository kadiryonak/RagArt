# RAG Evaluation Report

_Generated: 2026-05-18 19:51:07_

## Özet

- **Toplam örnek:** 12
- **Toplam süre:** 99.59 sn
- **Çalıştırılan katmanlar:** L1_rules, L2_vector, L3_lexical, L4_judge
- **Ortalama skor:** 0.7384

## Katman bazında özet

| Katman | N | Ort. skor | Pass oranı | Ort. latency (ms) |
|---|---|---|---|---|
| L1_rules | 12 | 0.9389 | 100.00% | 0.3 |
| L2_vector | 12 | 0.8804 | 91.67% | 99.8 |
| L3_lexical | 12 | 0.3226 | 75.00% | 0.9 |
| L4_judge | 6 | 0.8958 | 100.00% | 448.6 |

## Zorluk bazında özet

| Zorluk | N | Ort. skor |
|---|---|---|
| easy | 4 | 0.6891 |
| medium | 4 | 0.7529 |
| hard | 4 | 0.7731 |

## Örnek detayları

### ✓ `easy-01-algoritma-tanim` (skor: 0.818, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Algoritma nedir?

**Cevap:** Algoritma, "belli bir problemi çözmek veya belirli bir amaca ulaşmak için tasarlanan yol" olarak tanımlanır. Ayrıca, "bir başlangıç durumundan başladığında, açıkça belirlenmiş bir son durumunda sonlanan, sonlu işlemler kümesidir."

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.900 | ✓ | 0.5ms |
| L2_vector | 0.950 | ✓ | 86.4ms |
| L3_lexical | 0.485 | ✓ | 0.7ms |
| L4_judge | 0.938 | ✓ | 376.9ms |

### ✓ `easy-02-python-tanim` (skor: 0.791, zorluk: easy, kategori: factual)

**Soru:** Python nedir ve hangi alanlarda kullanılır?

**Cevap:** Python, "nesne yönelimli, yorumlamalı, birimsel (modüler) ve etkileşimli yüksek seviyeli bir programlama dilidir." Python ile sistem programlama, kullanıcı arabirimi programlama, ağ programlama, web programlama, uygulama ve veritabanı yazılımı programlama gibi birçok alanda yazılım geliştirebilirsin…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.936 | ✓ | 95.3ms |
| L3_lexical | 0.438 | ✓ | 1.1ms |
| L4_judge | 0.000 | ✗ | 0.0ms — skipped |

### ✗ `easy-03-yapay-zeka` (skor: 0.629, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Yapay zeka nedir?

**Cevap:** Yapay zeka, "insanlar da dahil olmak üzere hayvanlar tarafından, doğal zekânın aksine makineler tarafından görüntülenen zekâ çeşididir."

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.867 | ✓ | 0.1ms |
| L2_vector | 0.900 | ✓ | 97.9ms |
| L3_lexical | 0.124 | ✗ | 0.4ms |
| L4_judge | 0.625 | ✓ | 643.0ms |

### ✓ `medium-01-algoritma-tarih` (skor: 0.776, zorluk: medium, kategori: factual, **CRITICAL**)

**Soru:** Algoritma kelimesi nereden gelir ve ilk algoritma kim tarafından yazılmıştır?

**Cevap:** Algoritma kelimesi, Özbekistan'ın Harezm bölgesindeki Hive kentinde doğmuş olan Ebu Abdullah Muhammed İbn Musa el Harezmi'den gelir. İlk algoritma, el-Hârizmî tarafından "Hisab el-cebir ve el-mukabala" kitabında sunulmuştur.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.800 | ✓ | 0.1ms |
| L2_vector | 0.938 | ✓ | 89.9ms |
| L3_lexical | 0.428 | ✓ | 0.8ms |
| L4_judge | 0.938 | ✓ | 427.8ms |

### ✓ `medium-02-python-isim-kaynagi` (skor: 0.763, zorluk: medium, kategori: factual)

**Soru:** Python programlama dilinin ismi nereden gelir?

**Cevap:** Python'un ismi, dilin yaratıcısı olan Guido van Rossum'un dili geliştirirken keyif aldığı İngiliz komedi grubu Monty Python'dan gelmektedir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.954 | ✓ | 82.1ms |
| L3_lexical | 0.336 | ✓ | 0.6ms |
| L4_judge | 0.000 | ✗ | 0.0ms — skipped |

### ✓ `medium-03-veri-yapilari` (skor: 0.721, zorluk: medium, kategori: factual)

**Soru:** Veri yapıları nedir ve hangi temel türleri vardır?

**Cevap:** Veri yapısı, bilgisayar ortamında verilerin etkin olarak saklanması ve işlenmesi için kullanılan yapıdır. Farklı türdeki veri yapıları, farklı türdeki uygulamalara uygundur ve bazıları belirli görevlere oldukça uzmanlaşmıştır. Örneğin, ilişkisel veritabanları genellikle veri alımı için B-ağacı dizin…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.897 | ✓ | 126.6ms |
| L3_lexical | 0.268 | ✓ | 1.1ms |
| L4_judge | 0.000 | ✗ | 0.0ms — skipped |

### ✓ `medium-04-derin-ogrenme` (skor: 0.751, zorluk: medium, kategori: analytical, **CRITICAL**)

**Soru:** Derin öğrenme ile makine öğrenmesi arasındaki fark nedir?

**Cevap:** Derin öğrenme, bir veya daha fazla gizli katman içeren yapay sinir ağları ve benzeri makine öğrenme algoritmalarını kapsayan bir çalışma alanıdır. Makine öğrenimi ise, veriden öğrenebilen ve görünmeyen verilere genelleştirebilen istatistiksel algoritmaların geliştirilmesi ve incelenmesiyle ilgilenen…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.900 | ✓ | 0.1ms |
| L2_vector | 0.879 | ✓ | 128.6ms |
| L3_lexical | 0.289 | ✓ | 1.2ms |
| L4_judge | 0.938 | ✓ | 427.6ms |

### ✓ `hard-01-python-kullanim-yerleri` (skor: 0.798, zorluk: hard, kategori: synthesis)

**Soru:** Hangi büyük kurumlar ve projeler Python kullanır?

**Cevap:** Google, NASA, CERN, Wikipedia, Facebook, Amazon, Instagram ve Spotify gibi büyük kurumlar ve YouTube, orijinal BitTorrent istemcisi, Django, Zope uygulama sunucuları gibi projeler Python kullanır.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.942 | ✓ | 75.3ms |
| L3_lexical | 0.451 | ✓ | 0.7ms |
| L4_judge | 0.000 | ✗ | 0.0ms — skipped |

### ✓ `hard-02-algoritma-elestirisi` (skor: 0.772, zorluk: hard, kategori: analytical)

**Soru:** Algoritmalara yöneltilen toplumsal eleştiriler nelerdir?

**Cevap:** Algoritmaların toplumsal gruplar arasındaki eşitsizlikleri derinleştirdikleri, yeni güç dengeleri oluşturdukları ve farklı otoriteler tarafından toplumun belli kesimlerini baskılamak için kullanıldıklarının altı çizilmektedir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.882 | ✓ | 158.0ms |
| L3_lexical | 0.435 | ✓ | 0.6ms |
| L4_judge | 0.000 | ✗ | 0.0ms — skipped |

### ✓ `hard-03-multi-hop` (skor: 0.815, zorluk: hard, kategori: synthesis, **CRITICAL**)

**Soru:** Python'dan etkilenen programlama dilleri hangileridir ve bu diller hangi özellikleri ödünç almıştır?

**Cevap:** Python'dan etkilenen programlama dilleri arasında Boo, Cobra, CoffeeScript, ECMAScript/JavaScript, GDScript, Go, Groovy, Swift, Julia, Nim ve Ruby bulunur. Bu diller, Python'dan çeşitli özellikleri ödünç almışlardır. Örneğin:

- Boo ve Cobra, Python'un girintileme ve sözdizimini kullanır.
- CoffeeSc…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.2ms |
| L2_vector | 0.943 | ✓ | 124.8ms |
| L3_lexical | 0.379 | ✓ | 2.8ms |
| L4_judge | 0.938 | ✓ | 454.2ms |

### ✗ `edge-01-out-of-domain` (skor: 0.707, zorluk: hard, kategori: edge_case, **CRITICAL**)

**Soru:** Türkiye'de 2023 deprem felaketinin etkileri nelerdir?

**Cevap:** Bu konuda verilen bilgilerde yeterli detay bulunmuyor

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.800 | ✓ | 0.0ms |
| L2_vector | 0.789 | ✓ | 78.4ms |
| L3_lexical | 0.241 | ✗ | 0.3ms |
| L4_judge | 1.000 | ✓ | 362.4ms |

### ✗ `edge-02-bos-soru` (skor: 0.518, zorluk: easy, kategori: edge_case)

**Soru:** ?

**Cevap:** Bu konuda mevcut verilerimde yeterli detay bulunamadı, ancak genel bilgilerim şunlar:

Soru sormak için buradasın, ancak henüz bir soru sormadın. Lütfen sana nasıl yardımcı olabileceğimi söyleyerek bir soru sor.

---
📚 **Available Data Topics:**
Topics: Veri Bilimi, Biyoloji, Nasa, Genetik, Bilgisay…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 1.8ms |
| L2_vector | 0.555 | ✗ | 54.8ms |
| L3_lexical | 0.000 | ✗ | 0.3ms |
| L4_judge | 0.000 | ✗ | 0.0ms — skipped |
