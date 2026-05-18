# RAG Evaluation Report

_Generated: 2026-05-19 00:26:56_

## Özet

- **Toplam örnek:** 12
- **Toplam süre:** 100.18 sn
- **Çalıştırılan katmanlar:** L1_rules, L2_vector, L3_lexical
- **Ortalama skor:** 0.7038

## Katman bazında özet

| Katman | N | Ort. skor | Pass oranı | Ort. latency (ms) |
|---|---|---|---|---|
| L1_rules | 12 | 0.9139 | 100.00% | 0.1 |
| L2_vector | 12 | 0.8842 | 91.67% | 71.2 |
| L3_lexical | 12 | 0.3134 | 66.67% | 0.7 |

## Zorluk bazında özet

| Zorluk | N | Ort. skor |
|---|---|---|
| easy | 4 | 0.6654 |
| medium | 4 | 0.7019 |
| hard | 4 | 0.7441 |

## Örnek detayları

### ✓ `easy-01-algoritma-tanim` (skor: 0.778, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Algoritma nedir?

**Cevap:** Algoritma, "belli bir problemi çözmek veya belirli bir amaca ulaşmak için tasarlanan yol" olarak tanımlanır. Ayrıca, "bir başlangıç durumundan başladığında, açıkça belirlenmiş bir son durumunda sonlanan, sonlu işlemler kümesidir."

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.900 | ✓ | 0.5ms |
| L2_vector | 0.950 | ✓ | 66.3ms |
| L3_lexical | 0.485 | ✓ | 0.6ms |

### ✓ `easy-02-python-tanim` (skor: 0.733, zorluk: easy, kategori: factual)

**Soru:** Python nedir ve hangi alanlarda kullanılır?

**Cevap:** Python, nesne yönelimli, yorumlamalı, birimsel (modüler) ve etkileşimli yüksek seviyeli bir programlama dilidir. Python'un kullanımı hakkında verilen bilgilerde yeterli detay bulunmuyor. Ancak, Python'un programlama dilleri arasında yer aldığı ve yazılımcının bir algoritmayı ifade etmek amacıyla, bi…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.935 | ✓ | 79.5ms |
| L3_lexical | 0.265 | ✓ | 0.7ms |

### ✗ `easy-03-yapay-zeka` (skor: 0.630, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Yapay zeka nedir?

**Cevap:** Yapay zeka, "insanlar da dahil olmak üzere hayvanlar tarafından, doğal zekânın aksine makineler tarafından görüntülenen zekâ çeşididir."

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.867 | ✓ | 0.1ms |
| L2_vector | 0.900 | ✓ | 64.4ms |
| L3_lexical | 0.124 | ✗ | 0.5ms |

### ✓ `medium-01-algoritma-tarih` (skor: 0.753, zorluk: medium, kategori: factual, **CRITICAL**)

**Soru:** Algoritma kelimesi nereden gelir ve ilk algoritma kim tarafından yazılmıştır?

**Cevap:** Algoritma sözcüğü, Özbekistan'ın Harezm bölgesindeki Hive kentinde doğmuş olan Ebu Abdullah Muhammed İbn Musa el Harezmi'den gelir. Bu alim 9. yüzyılda cebir alanındaki algoritmik çalışmalarını kitaba dökerek matematiğe çok büyük bir katkı sağlamıştır. "Hisab el-cebir ve el-mukabala (حساب الجبر و ال…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.900 | ✓ | 0.1ms |
| L2_vector | 0.956 | ✓ | 84.8ms |
| L3_lexical | 0.402 | ✓ | 1.0ms |

### ✓ `medium-02-python-isim-kaynagi` (skor: 0.763, zorluk: medium, kategori: factual)

**Soru:** Python programlama dilinin ismi nereden gelir?

**Cevap:** Python'un ismi, dilin yaratıcısı olan Guido van Rossum'un dili geliştirirken keyif aldığı İngiliz komedi grubu Monty Python'dan gelmektedir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.954 | ✓ | 58.7ms |
| L3_lexical | 0.336 | ✓ | 0.5ms |

### ✗ `medium-03-veri-yapilari` (skor: 0.613, zorluk: medium, kategori: factual)

**Soru:** Veri yapıları nedir ve hangi temel türleri vardır?

**Cevap:** Veri yapıları, "verileri verimli bir şekilde organize etmek ve depolamak" gibi ortak bir hedefi paylaşır. Temel türleri arasında dizi, kayıt ve bağlantılı veri yapıları bulunur. Dizi ve kayıt veri yapıları, "veri öğelerinin adreslerinin aritmetik işlemlerle hesaplanmasına" dayanırken, bağlantılı ver…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.680 | ✓ | 0.1ms |
| L2_vector | 0.909 | ✓ | 74.5ms |
| L3_lexical | 0.249 | ✗ | 0.9ms |

### ✓ `medium-04-derin-ogrenme` (skor: 0.679, zorluk: medium, kategori: analytical, **CRITICAL**)

**Soru:** Derin öğrenme ile makine öğrenmesi arasındaki fark nedir?

**Cevap:** Derin öğrenme ile makine öğrenmesi arasındaki fark, derin öğrenmenin bir veya daha fazla gizli katman içeren yapay sinir ağları ve benzeri makine öğrenme algoritmalarını kapsayan bir çalışma alanı olmasıdır. Makine öğrenimi ise, veriden öğrenebilen ve görünmeyen verilere genelleştirebilen ve dolayıs…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.900 | ✓ | 0.1ms |
| L2_vector | 0.886 | ✓ | 70.5ms |
| L3_lexical | 0.251 | ✓ | 1.0ms |

### ✓ `hard-01-python-kullanim-yerleri` (skor: 0.829, zorluk: hard, kategori: synthesis)

**Soru:** Hangi büyük kurumlar ve projeler Python kullanır?

**Cevap:** Google, NASA, CERN, Wikipedia, Yahoo!, Facebook, Amazon, Instagram ve Spotify gibi büyük kurumlar ve Reddit, YouTube, Django, Zope uygulama sunucuları ve orijinal BitTorrent istemcisi gibi projeler Python kullanır.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.947 | ✓ | 80.1ms |
| L3_lexical | 0.538 | ✓ | 0.6ms |

### ✓ `hard-02-algoritma-elestirisi` (skor: 0.772, zorluk: hard, kategori: analytical)

**Soru:** Algoritmalara yöneltilen toplumsal eleştiriler nelerdir?

**Cevap:** Algoritmaların toplumsal gruplar arasındaki eşitsizlikleri derinleştirdikleri, yeni güç dengeleri oluşturdukları ve farklı otoriteler tarafından toplumun belli kesimlerini baskılamak için kullanıldıklarının altı çizilmektedir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.882 | ✓ | 74.7ms |
| L3_lexical | 0.435 | ✓ | 0.5ms |

### ✓ `hard-03-multi-hop` (skor: 0.769, zorluk: hard, kategori: synthesis, **CRITICAL**)

**Soru:** Python'dan etkilenen programlama dilleri hangileridir ve bu diller hangi özellikleri ödünç almıştır?

**Cevap:** Python'dan etkilenen programlama dilleri arasında Boo, Cobra, CoffeeScript, JavaScript, GDScript, Go, Groovy ve Swift bulunur. Bu diller, Python'dan çeşitli özellikler ödünç almışlardır. Örneğin:

- Boo, Python'un benzer girintileme, sözdizimi ve nesne modelini kullanır.
- Cobra, Python'un benzer gi…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.920 | ✓ | 0.1ms |
| L2_vector | 0.953 | ✓ | 88.9ms |
| L3_lexical | 0.435 | ✓ | 1.7ms |

### ✗ `edge-01-out-of-domain` (skor: 0.606, zorluk: hard, kategori: edge_case, **CRITICAL**)

**Soru:** Türkiye'de 2023 deprem felaketinin etkileri nelerdir?

**Cevap:** Bu konuda verilen bilgilerde yeterli detay bulunmuyor.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.800 | ✓ | 0.1ms |
| L2_vector | 0.777 | ✓ | 65.6ms |
| L3_lexical | 0.241 | ✗ | 0.3ms |

### ✗ `edge-02-bos-soru` (skor: 0.520, zorluk: easy, kategori: edge_case)

**Soru:** ?

**Cevap:** Bu konuda mevcut verilerimde yeterli detay bulunamadı, ancak genel bilgilerim şunlar:

Soru sormak için buradasın, ancak henüz bir soru sormadın. Lütfen sana nasıl yardımcı olabileceğimi söyleyerek bir soru sor.

---
📚 **Available Data Topics:**
Topics: Biyoloji, Veri Bilimi, Nasa, Bilgisayarla Görm…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.2ms |
| L2_vector | 0.560 | ✗ | 45.9ms |
| L3_lexical | 0.000 | ✗ | 0.4ms |
