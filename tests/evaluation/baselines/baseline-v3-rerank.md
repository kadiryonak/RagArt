# RAG Evaluation Report

_Generated: 2026-05-18 18:37:31_

## Özet

- **Toplam örnek:** 12
- **Toplam süre:** 237.15 sn
- **Çalıştırılan katmanlar:** L1_rules, L2_vector, L3_lexical
- **Ortalama skor:** 0.7096

## Katman bazında özet

| Katman | N | Ort. skor | Pass oranı | Ort. latency (ms) |
|---|---|---|---|---|
| L1_rules | 12 | 0.9256 | 100.00% | 0.2 |
| L2_vector | 12 | 0.8828 | 91.67% | 86.2 |
| L3_lexical | 12 | 0.3203 | 66.67% | 0.9 |

## Zorluk bazında özet

| Zorluk | N | Ort. skor |
|---|---|---|
| easy | 4 | 0.6808 |
| medium | 4 | 0.7083 |
| hard | 4 | 0.7396 |

## Örnek detayları

### ✓ `easy-01-algoritma-tanim` (skor: 0.778, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Algoritma nedir?

**Cevap:** Algoritma, "belli bir problemi çözmek veya belirli bir amaca ulaşmak için tasarlanan yol" olarak tanımlanır. Ayrıca, "bir başlangıç durumundan başladığında, açıkça belirlenmiş bir son durumunda sonlanan, sonlu işlemler kümesidir."

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.900 | ✓ | 0.5ms |
| L2_vector | 0.950 | ✓ | 91.5ms |
| L3_lexical | 0.485 | ✓ | 0.9ms |

### ✓ `easy-02-python-tanim` (skor: 0.791, zorluk: easy, kategori: factual)

**Soru:** Python nedir ve hangi alanlarda kullanılır?

**Cevap:** Python, "nesne yönelimli, yorumlamalı, birimsel (modüler) ve etkileşimli yüksek seviyeli bir programlama dilidir." Python ile sistem programlama, kullanıcı arabirimi programlama, ağ programlama, web programlama, uygulama ve veritabanı yazılımı programlama gibi birçok alanda yazılım geliştirebilirsin…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.936 | ✓ | 90.9ms |
| L3_lexical | 0.438 | ✓ | 1.1ms |

### ✗ `easy-03-yapay-zeka` (skor: 0.630, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Yapay zeka nedir?

**Cevap:** Yapay zeka, "insanlar da dahil olmak üzere hayvanlar tarafından, doğal zekânın aksine makineler tarafından görüntülenen zekâ çeşididir."

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.867 | ✓ | 0.1ms |
| L2_vector | 0.900 | ✓ | 86.6ms |
| L3_lexical | 0.124 | ✗ | 0.4ms |

### ✓ `medium-01-algoritma-tarih` (skor: 0.722, zorluk: medium, kategori: factual, **CRITICAL**)

**Soru:** Algoritma kelimesi nereden gelir ve ilk algoritma kim tarafından yazılmıştır?

**Cevap:** Algoritma kelimesi, Özbekistan'ın Harezm bölgesindeki Hive kentinde doğmuş olan Ebu Abdullah Muhammed İbn Musa el Harezmi'den gelir. İlk algoritma, el-Hârizmî tarafından "Hisab el-cebir ve el-mukabala" kitabında sunulmuştur.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.800 | ✓ | 0.1ms |
| L2_vector | 0.938 | ✓ | 93.4ms |
| L3_lexical | 0.428 | ✓ | 0.8ms |

### ✓ `medium-02-python-isim-kaynagi` (skor: 0.763, zorluk: medium, kategori: factual)

**Soru:** Python programlama dilinin ismi nereden gelir?

**Cevap:** Python'un ismi, dilin yaratıcısı olan Guido van Rossum'un dili geliştirirken keyif aldığı İngiliz komedi grubu Monty Python'dan gelmektedir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.954 | ✓ | 91.8ms |
| L3_lexical | 0.336 | ✓ | 0.7ms |

### ✗ `medium-03-veri-yapilari` (skor: 0.661, zorluk: medium, kategori: factual)

**Soru:** Veri yapıları nedir ve hangi temel türleri vardır?

**Cevap:** Veri yapısı, bilgisayar ortamında verilerin etkin olarak saklanması ve işlenmesi için kullanılan yapıdır. Farklı türdeki veri yapıları, farklı türdeki uygulamalara uygundur ve bazıları belirli görevlere oldukça uzmanlaşmıştır. Örneğin, ilişkisel veritabanları genellikle veri alımı için B-ağacı dizin…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.840 | ✓ | 0.1ms |
| L2_vector | 0.897 | ✓ | 104.2ms |
| L3_lexical | 0.247 | ✗ | 1.1ms |

### ✓ `medium-04-derin-ogrenme` (skor: 0.687, zorluk: medium, kategori: analytical, **CRITICAL**)

**Soru:** Derin öğrenme ile makine öğrenmesi arasındaki fark nedir?

**Cevap:** Derin öğrenme, "bir veya daha fazla gizli katman içeren yapay sinir ağları ve benzeri makine öğrenme algoritmalarını kapsayan çalışma alanıdır." Makine öğrenmesi ise, "veriden öğrenebilen ve görünmeyen verilere genelleştirebilen ve dolayısıyla açık talimatlar olmadan görevleri yerine getirebilen ist…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.900 | ✓ | 0.1ms |
| L2_vector | 0.900 | ✓ | 103.1ms |
| L3_lexical | 0.260 | ✓ | 1.4ms |

### ✓ `hard-01-python-kullanim-yerleri` (skor: 0.798, zorluk: hard, kategori: synthesis)

**Soru:** Hangi büyük kurumlar ve projeler Python kullanır?

**Cevap:** Google, NASA, CERN, Wikipedia, Facebook, Amazon, Instagram ve Spotify gibi büyük kurumlar ve YouTube, orijinal BitTorrent istemcisi, Django, Zope uygulama sunucuları gibi projeler Python kullanır.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.942 | ✓ | 74.6ms |
| L3_lexical | 0.451 | ✓ | 0.7ms |

### ✓ `hard-02-algoritma-elestirisi` (skor: 0.772, zorluk: hard, kategori: analytical)

**Soru:** Algoritmalara yöneltilen toplumsal eleştiriler nelerdir?

**Cevap:** Algoritmaların toplumsal gruplar arasındaki eşitsizlikleri derinleştirdikleri, yeni güç dengeleri oluşturdukları ve farklı otoriteler tarafından toplumun belli kesimlerini baskılamak için kullanıldıklarının altı çizilmektedir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.882 | ✓ | 83.6ms |
| L3_lexical | 0.435 | ✓ | 0.6ms |

### ✓ `hard-03-multi-hop` (skor: 0.781, zorluk: hard, kategori: synthesis, **CRITICAL**)

**Soru:** Python'dan etkilenen programlama dilleri hangileridir ve bu diller hangi özellikleri ödünç almıştır?

**Cevap:** Python'dan etkilenen programlama dilleri arasında Boo, Cobra, CoffeeScript, ECMAScript/JavaScript, GDScript, Go, Groovy, Julia, Nim, Ruby ve Swift bulunur. Bu diller, Python'dan çeşitli özellikler ödünç almıştır. Örneğin:

- Boo ve Cobra, Python'un benzer girintileme ve sözdizimini kullanır.
- Coffe…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.2ms |
| L2_vector | 0.942 | ✓ | 96.2ms |
| L3_lexical | 0.401 | ✓ | 2.5ms |

### ✗ `edge-01-out-of-domain` (skor: 0.607, zorluk: hard, kategori: edge_case, **CRITICAL**)

**Soru:** Türkiye'de 2023 deprem felaketinin etkileri nelerdir?

**Cevap:** Bu konuda verilen bilgilerde yeterli detay bulunmuyor.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.800 | ✓ | 0.0ms |
| L2_vector | 0.781 | ✓ | 63.1ms |
| L3_lexical | 0.241 | ✗ | 0.2ms |

### ✗ `edge-02-bos-soru` (skor: 0.524, zorluk: easy, kategori: edge_case)

**Soru:** ?

**Cevap:** Bu konuda mevcut verilerimde yeterli detay bulunamadı, ancak genel bilgilerim şunlar:

Soru sormak için buradasın, ancak henüz bir soru sormadın. Lütfen sana nasıl yardımcı olabileceğimi söyleyerek bir soru sor.

---
📚 **Available Data Topics:**
Topics: Bilgisayarla Görme, Veri Bilimi, Biyoloji, Nas…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.6ms |
| L2_vector | 0.571 | ✗ | 55.4ms |
| L3_lexical | 0.000 | ✗ | 0.3ms |
