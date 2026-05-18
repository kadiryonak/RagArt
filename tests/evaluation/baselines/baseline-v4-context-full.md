# RAG Evaluation Report

_Generated: 2026-05-19 00:24:38_

## Özet

- **Toplam örnek:** 12
- **Toplam süre:** 114.88 sn
- **Çalıştırılan katmanlar:** L1_rules, L2_vector, L3_lexical
- **Ortalama skor:** 0.6859

## Katman bazında özet

| Katman | N | Ort. skor | Pass oranı | Ort. latency (ms) |
|---|---|---|---|---|
| L1_rules | 12 | 0.9111 | 100.00% | 0.2 |
| L2_vector | 12 | 0.8500 | 83.33% | 85.5 |
| L3_lexical | 12 | 0.2965 | 75.00% | 0.7 |

## Zorluk bazında özet

| Zorluk | N | Ort. skor |
|---|---|---|
| easy | 4 | 0.6457 |
| medium | 4 | 0.7067 |
| hard | 4 | 0.7052 |

## Örnek detayları

### ✓ `easy-01-algoritma-tanim` (skor: 0.778, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Algoritma nedir?

**Cevap:** Algoritma, "belli bir problemi çözmek veya belirli bir amaca ulaşmak için tasarlanan yol" olarak tanımlanır. Ayrıca, "bir başlangıç durumundan başladığında, açıkça belirlenmiş bir son durumunda sonlanan, sonlu işlemler kümesidir."

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.900 | ✓ | 0.9ms |
| L2_vector | 0.950 | ✓ | 141.3ms |
| L3_lexical | 0.485 | ✓ | 0.8ms |

### ✓ `easy-02-python-tanim` (skor: 0.725, zorluk: easy, kategori: factual)

**Soru:** Python nedir ve hangi alanlarda kullanılır?

**Cevap:** Python, "nesne yönelimli, yorumlamalı, birimsel (modüler) ve etkileşimli yüksek seviyeli bir programlama dilidir." Python'un kullanımı hakkında verilen bilgilerde yeterli detay bulunmuyor.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.908 | ✓ | 86.4ms |
| L3_lexical | 0.267 | ✓ | 0.5ms |

### ✗ `easy-03-yapay-zeka` (skor: 0.557, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Yapay zeka nedir?

**Cevap:** Yapay zekâ (YZ), insanlar da dahil olmak üzere hayvanlar tarafından, doğal zekânın aksine makineler tarafından görüntülenen zekâ çeşididir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.733 | ✓ | 0.1ms |
| L2_vector | 0.851 | ✓ | 75.1ms |
| L3_lexical | 0.087 | ✗ | 0.5ms |

### ✓ `medium-01-algoritma-tarih` (skor: 0.726, zorluk: medium, kategori: factual, **CRITICAL**)

**Soru:** Algoritma kelimesi nereden gelir ve ilk algoritma kim tarafından yazılmıştır?

**Cevap:** Algoritma sözcüğü, Özbekistan'ın Harezm bölgesindeki Hive kentinde doğmuş olan Ebu Abdullah Muhammed İbn Musa el Harezmi'den gelir. "Hisab el-cebir ve el-mukabala (حساب الجبر و المقابلة)" kitabı dünyanın ilk cebir kitabı ve aynı zamanda ilk algoritma koleksiyonunu oluşturur.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.800 | ✓ | 0.1ms |
| L2_vector | 0.939 | ✓ | 81.8ms |
| L3_lexical | 0.440 | ✓ | 0.8ms |

### ✓ `medium-02-python-isim-kaynagi` (skor: 0.763, zorluk: medium, kategori: factual)

**Soru:** Python programlama dilinin ismi nereden gelir?

**Cevap:** Python'un ismi, dilin yaratıcısı olan Guido van Rossum'un dili geliştirirken keyif aldığı İngiliz komedi grubu Monty Python'dan gelmektedir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.954 | ✓ | 60.6ms |
| L3_lexical | 0.336 | ✓ | 0.6ms |

### ✓ `medium-03-veri-yapilari` (skor: 0.639, zorluk: medium, kategori: factual)

**Soru:** Veri yapıları nedir ve hangi temel türleri vardır?

**Cevap:** Veri yapıları, "verileri verimli bir şekilde organize etmek ve depolamak" gibi ortak bir hedefi paylaşır. Temel türleri arasında dizi, kayıt ve bağlantılı veri yapıları bulunur.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.680 | ✓ | 0.1ms |
| L2_vector | 0.906 | ✓ | 62.9ms |
| L3_lexical | 0.331 | ✓ | 0.5ms |

### ✓ `medium-04-derin-ogrenme` (skor: 0.698, zorluk: medium, kategori: analytical, **CRITICAL**)

**Soru:** Derin öğrenme ile makine öğrenmesi arasındaki fark nedir?

**Cevap:** Derin öğrenme, makine öğrenmesinin bir alt kümesidir ve bir veya daha fazla gizli katman içeren yapay sinir ağları ve benzeri makine öğrenme algoritmalarını kapsar. Yani, derin öğrenme, makine öğrenmesinin bir özel türüdür, özellikle yapay sinir ağlarını kullanır.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.900 | ✓ | 0.1ms |
| L2_vector | 0.868 | ✓ | 64.4ms |
| L3_lexical | 0.327 | ✓ | 0.5ms |

### ✓ `hard-01-python-kullanim-yerleri` (skor: 0.746, zorluk: hard, kategori: synthesis)

**Soru:** Hangi büyük kurumlar ve projeler Python kullanır?

**Cevap:** Google, NASA, CERN, Wikipedia, Yahoo!, Facebook, Amazon, Instagram ve Spotify gibi büyük kurumlar ve YouTube, Reddit gibi projeler Python kullanır.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.903 | ✓ | 101.5ms |
| L3_lexical | 0.336 | ✓ | 0.5ms |

### ✓ `hard-02-algoritma-elestirisi` (skor: 0.772, zorluk: hard, kategori: analytical)

**Soru:** Algoritmalara yöneltilen toplumsal eleştiriler nelerdir?

**Cevap:** Algoritmaların toplumsal gruplar arasındaki eşitsizlikleri derinleştirdikleri, yeni güç dengeleri oluşturdukları ve farklı otoriteler tarafından toplumun belli kesimlerini baskılamak için kullanıldıklarının altı çizilmektedir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.882 | ✓ | 67.8ms |
| L3_lexical | 0.435 | ✓ | 0.4ms |

### ✓ `hard-03-multi-hop` (skor: 0.779, zorluk: hard, kategori: synthesis, **CRITICAL**)

**Soru:** Python'dan etkilenen programlama dilleri hangileridir ve bu diller hangi özellikleri ödünç almıştır?

**Cevap:** Python'dan etkilenen programlama dilleri arasında Boo, Cobra, CoffeeScript, ECMAScript/JavaScript, GDScript, Go, Groovy ve Swift yer alır. Bu diller, Python'dan çeşitli özellikler ödünç almışlardır. Örneğin, Cobra ve Boo, Python'un girintileme ve sözdizimini kullanırken, CoffeeScript, Python'dan ilh…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.920 | ✓ | 0.1ms |
| L2_vector | 0.944 | ✓ | 86.4ms |
| L3_lexical | 0.472 | ✓ | 1.0ms |

### ✗ `edge-01-out-of-domain` (skor: 0.523, zorluk: hard, kategori: edge_case, **CRITICAL**)

**Soru:** Türkiye'de 2023 deprem felaketinin etkileri nelerdir?

**Cevap:** Bu konuda mevcut verilerimde yeterli detay bulunamadı, ancak genel bilgilerim şunlar:

Merhaba, 2023 yılında yaşanan deprem felaketi Türkiye'de büyük bir yıkıma neden oldu. Depremin etkileri arasında binlerce insanın hayatını kaybetmesi, çok sayıda kişinin yaralanması ve evsiz kalması yer alıyor. Ay…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.8ms |
| L2_vector | 0.528 | ✗ | 111.4ms |
| L3_lexical | 0.041 | ✗ | 1.8ms |

### ✗ `edge-02-bos-soru` (skor: 0.522, zorluk: easy, kategori: edge_case)

**Soru:** ?

**Cevap:** Bu konuda mevcut verilerimde yeterli detay bulunamadı, ancak genel bilgilerim şunlar:

Soru sormak için buradasın, ancak henüz bir soru sormadın. Lütfen sana nasıl yardımcı olabileceğimi söyleyerek bir soru sor.

---
📚 **Available Data Topics:**
Topics: Veri Bilimi, Genetik, Bilgisayarla Görme, Biyo…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.2ms |
| L2_vector | 0.567 | ✗ | 86.8ms |
| L3_lexical | 0.000 | ✗ | 0.6ms |
