# RAG Evaluation Report

_Generated: 2026-05-18 18:11:20_

## Özet

- **Toplam örnek:** 12
- **Toplam süre:** 8.53 sn
- **Çalıştırılan katmanlar:** L1_rules, L2_vector, L3_lexical
- **Ortalama skor:** 0.6645

## Katman bazında özet

| Katman | N | Ort. skor | Pass oranı | Ort. latency (ms) |
|---|---|---|---|---|
| L1_rules | 12 | 0.8844 | 100.00% | 0.2 |
| L2_vector | 12 | 0.8429 | 83.33% | 71.2 |
| L3_lexical | 12 | 0.2663 | 58.33% | 0.8 |

## Zorluk bazında özet

| Zorluk | N | Ort. skor |
|---|---|---|
| easy | 4 | 0.6496 |
| medium | 4 | 0.6794 |
| hard | 4 | 0.6647 |

## Örnek detayları

### ✓ `easy-01-algoritma-tanim` (skor: 0.773, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Algoritma nedir?

**Cevap:** Algoritma, "belli bir problemi çözmek veya belirli bir amaca ulaşmak için tasarlanan yol" olarak tanımlanır. Ayrıca, "bir başlangıç durumundan başladığında, açıkça belirlenmiş bir son durumunda sonlanan, sonlu işlemler kümesidir."

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.900 | ✓ | 0.5ms |
| L2_vector | 0.934 | ✓ | 78.1ms |
| L3_lexical | 0.485 | ✓ | 1.3ms |

### ✓ `easy-02-python-tanim` (skor: 0.722, zorluk: easy, kategori: factual)

**Soru:** Python nedir ve hangi alanlarda kullanılır?

**Cevap:** Python, "nesne yönelimli, yorumlamalı, birimsel (modüler) ve etkileşimli yüksek seviyeli bir programlama dilidir." Python'un kullanımı hakkında verilen bilgilerde yeterli detay bulunmuyor.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.2ms |
| L2_vector | 0.898 | ✓ | 76.0ms |
| L3_lexical | 0.267 | ✓ | 0.6ms |

### ✗ `easy-03-yapay-zeka` (skor: 0.580, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Yapay zeka nedir?

**Cevap:** Yapay zeka, "insan zekâsına özgü yüksek bilişsel fonksiyonları veya otonom davranışları sergileyen bir yapay işletim sistemidir" ve "çevresini algılayan ve hedeflerine başarıyla ulaşma şansını en üst düzeye çıkaran eylemleri gerçekleştiren herhangi bir cihaz" olarak tanımlanabilir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.733 | ✓ | 0.1ms |
| L2_vector | 0.903 | ✓ | 85.2ms |
| L3_lexical | 0.103 | ✗ | 0.7ms |

### ✓ `medium-01-algoritma-tarih` (skor: 0.722, zorluk: medium, kategori: factual, **CRITICAL**)

**Soru:** Algoritma kelimesi nereden gelir ve ilk algoritma kim tarafından yazılmıştır?

**Cevap:** Algoritma kelimesi, Özbekistan'ın Harezm bölgesindeki Hive kentinde doğmuş olan Ebu Abdullah Muhammed İbn Musa el Harezmi'den gelir. İlk algoritma, el-Hârizmî tarafından "Hisab el-cebir ve el-mukabala" kitabında sunulmuştur.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.800 | ✓ | 0.1ms |
| L2_vector | 0.938 | ✓ | 73.7ms |
| L3_lexical | 0.428 | ✓ | 0.7ms |

### ✓ `medium-02-python-isim-kaynagi` (skor: 0.763, zorluk: medium, kategori: factual)

**Soru:** Python programlama dilinin ismi nereden gelir?

**Cevap:** Python'un ismi, dilin yaratıcısı olan Guido van Rossum'un dili geliştirirken keyif aldığı İngiliz komedi grubu Monty Python'dan gelmektedir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.954 | ✓ | 68.8ms |
| L3_lexical | 0.336 | ✓ | 0.6ms |

### ✗ `medium-03-veri-yapilari` (skor: 0.518, zorluk: medium, kategori: factual)

**Soru:** Veri yapıları nedir ve hangi temel türleri vardır?

**Cevap:** Veri yapısı, "verilere verimli erişim için seçilen bir veri organizasyonu ve depolama biçimidir." Temel türleri hakkında bilgi verilmemiştir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.600 | ✓ | 0.1ms |
| L2_vector | 0.827 | ✓ | 65.6ms |
| L3_lexical | 0.128 | ✗ | 0.5ms |

### ✓ `medium-04-derin-ogrenme` (skor: 0.714, zorluk: medium, kategori: analytical, **CRITICAL**)

**Soru:** Derin öğrenme ile makine öğrenmesi arasındaki fark nedir?

**Cevap:** Derin öğrenme, bir veya daha fazla gizli katman içeren yapay sinir ağları ve benzeri makine öğrenme algoritmalarını kapsayan çalışma alanıdır. Makine öğrenmesi ise, veriden öğrenebilen ve görünmeyen verilere genelleştirebilen istatistiksel algoritmaların geliştirilmesi ve incelenmesiyle ilgilenen bi…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.900 | ✓ | 0.1ms |
| L2_vector | 0.926 | ✓ | 82.6ms |
| L3_lexical | 0.317 | ✓ | 1.0ms |

### ✓ `hard-01-python-kullanim-yerleri` (skor: 0.798, zorluk: hard, kategori: synthesis)

**Soru:** Hangi büyük kurumlar ve projeler Python kullanır?

**Cevap:** Google, NASA, CERN, Wikipedia, Facebook, Amazon, Instagram ve Spotify gibi büyük kurumlar ve YouTube, orijinal BitTorrent istemcisi, Django, Zope uygulama sunucuları gibi projeler Python kullanır.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.942 | ✓ | 74.5ms |
| L3_lexical | 0.451 | ✓ | 0.5ms |

### ✓ `hard-02-algoritma-elestirisi` (skor: 0.772, zorluk: hard, kategori: analytical)

**Soru:** Algoritmalara yöneltilen toplumsal eleştiriler nelerdir?

**Cevap:** Algoritmaların toplumsal gruplar arasındaki eşitsizlikleri derinleştirdikleri, yeni güç dengeleri oluşturdukları ve farklı otoriteler tarafından toplumun belli kesimlerini baskılamak için kullanıldıklarının altı çizilmektedir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.882 | ✓ | 59.8ms |
| L3_lexical | 0.435 | ✓ | 0.5ms |

### ✗ `hard-03-multi-hop` (skor: 0.566, zorluk: hard, kategori: synthesis, **CRITICAL**)

**Soru:** Python'dan etkilenen programlama dilleri hangileridir ve bu diller hangi özellikleri ödünç almıştır?

**Cevap:** Python'dan etkilenen programlama dilleri arasında Swift yer almaktadır. Swift, Python'dan ilham alan bir sözdizimine sahiptir. Ayrıca, Python'un geliştirme uygulamaları diğer diller tarafından da taklit edilmiştir. Örneğin, Tcl, Erlang ve Swift, Python'da kullanılan "PEP" (Python Enhancement Proposa…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.680 | ✓ | 0.1ms |
| L2_vector | 0.807 | ✓ | 83.0ms |
| L3_lexical | 0.210 | ✗ | 0.9ms |

### ✗ `edge-01-out-of-domain` (skor: 0.523, zorluk: hard, kategori: edge_case, **CRITICAL**)

**Soru:** Türkiye'de 2023 deprem felaketinin etkileri nelerdir?

**Cevap:** Bu konuda mevcut verilerimde yeterli detay bulunamadı, ancak genel bilgilerim şunlar:

Türkiye'de 2023 yılında yaşanan deprem felaketi, ülkenin güneydoğu bölgesini vurmuş ve büyük bir yıkıma neden olmuştur. Depremin etkileri arasında:

1. **Can kayıpları**: Binlerce insan hayatını kaybetmiş, çok say…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.4ms |
| L2_vector | 0.533 | ✗ | 60.6ms |
| L3_lexical | 0.037 | ✗ | 1.9ms |

### ✗ `edge-02-bos-soru` (skor: 0.524, zorluk: easy, kategori: edge_case)

**Soru:** ?

**Cevap:** Bu konuda mevcut verilerimde yeterli detay bulunamadı, ancak genel bilgilerim şunlar:

Soru sormak için buradasın, ancak henüz bir soru sormadın. Lütfen sana nasıl yardımcı olabileceğimi söyleyerek bir soru sor.

---
📚 **Available Data Topics:**
Topics: Nasa, Genetik, Biyoloji, Veri Bilimi, Bilgisay…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.2ms |
| L2_vector | 0.572 | ✗ | 46.2ms |
| L3_lexical | 0.000 | ✗ | 0.6ms |
