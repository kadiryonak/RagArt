# RAG Evaluation Report

_Generated: 2026-05-18 18:25:27_

## Özet

- **Toplam örnek:** 12
- **Toplam süre:** 13.97 sn
- **Çalıştırılan katmanlar:** L1_rules, L2_vector, L3_lexical
- **Ortalama skor:** 0.6946

## Katman bazında özet

| Katman | N | Ort. skor | Pass oranı | Ort. latency (ms) |
|---|---|---|---|---|
| L1_rules | 12 | 0.9283 | 100.00% | 0.2 |
| L2_vector | 12 | 0.8492 | 83.33% | 66.1 |
| L3_lexical | 12 | 0.3064 | 66.67% | 0.8 |

## Zorluk bazında özet

| Zorluk | N | Ort. skor |
|---|---|---|
| easy | 4 | 0.6678 |
| medium | 4 | 0.7040 |
| hard | 4 | 0.7121 |

## Örnek detayları

### ✓ `easy-01-algoritma-tanim` (skor: 0.763, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Algoritma nedir?

**Cevap:** Algoritma, "belli bir problemi çözmek veya belirli bir amaca ulaşmak için tasarlanan yol" olarak tanımlanır. Ayrıca, "bir başlangıç durumundan başladığında, açıkça belirlenmiş bir son durumunda sonlanan, sonlu işlemler kümesidir."

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.900 | ✓ | 0.5ms |
| L2_vector | 0.903 | ✓ | 80.9ms |
| L3_lexical | 0.485 | ✓ | 1.0ms |

### ✗ `easy-02-python-tanim` (skor: 0.652, zorluk: easy, kategori: factual)

**Soru:** Python nedir ve hangi alanlarda kullanılır?

**Cevap:** Python, Guido van Rossum tarafından geliştirilmiş bir programlama dilidir. "Python'un son derece kolay okunabilir olması düşünülmüştür." Ayrıca, Django, YouTube, Google, NASA ve CERN gibi büyük kurumlar tarafından kullanılan önemli projelerden bazılarıdır. Python, 2D oyun geliştirme, blockchain uygu…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.800 | ✓ | 0.1ms |
| L2_vector | 0.913 | ✓ | 84.5ms |
| L3_lexical | 0.242 | ✗ | 1.1ms |

### ✓ `easy-03-yapay-zeka` (skor: 0.732, zorluk: easy, kategori: factual, **CRITICAL**)

**Soru:** Yapay zeka nedir?

**Cevap:** Diyalog bazlı yapay zeka, doğal dil işleme ve makine öğrenmesi gibi yapay zeka teknolojileri kullanılarak insan ve makine (yazılım) arasında bir diyaloğun sürdürülmesini sağlar.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.922 | ✓ | 70.0ms |
| L3_lexical | 0.274 | ✓ | 0.5ms |

### ✓ `medium-01-algoritma-tarih` (skor: 0.722, zorluk: medium, kategori: factual, **CRITICAL**)

**Soru:** Algoritma kelimesi nereden gelir ve ilk algoritma kim tarafından yazılmıştır?

**Cevap:** Algoritma kelimesi, Özbekistan'ın Harezm bölgesindeki Hive kentinde doğmuş olan Ebu Abdullah Muhammed İbn Musa el Harezmi'den gelir. İlk algoritma, el-Hârizmî tarafından "Hisab el-cebir ve el-mukabala" kitabında sunulmuştur.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.800 | ✓ | 0.1ms |
| L2_vector | 0.938 | ✓ | 68.7ms |
| L3_lexical | 0.428 | ✓ | 0.8ms |

### ✓ `medium-02-python-isim-kaynagi` (skor: 0.762, zorluk: medium, kategori: factual)

**Soru:** Python programlama dilinin ismi nereden gelir?

**Cevap:** Python'un ismi, dilin yaratıcısı olan Guido van Rossum'un dili geliştirirken keyif aldığı İngiliz komedi grubu Monty Python'dan gelmektedir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.950 | ✓ | 64.7ms |
| L3_lexical | 0.336 | ✓ | 0.6ms |

### ✓ `medium-03-veri-yapilari` (skor: 0.753, zorluk: medium, kategori: factual)

**Soru:** Veri yapıları nedir ve hangi temel türleri vardır?

**Cevap:** Veri yapısı, bilgisayar ortamında verilerin etkin olarak saklanması ve işlenmesi için kullanılan yapıdır. Temel türleri arasında dizi, kayıt, bağlantılı veri yapıları, listeler, yığıtlar, kuyruklar, ağaçlar ve çizgeler gibi veri modelleri bulunur.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.920 | ✓ | 0.1ms |
| L2_vector | 0.906 | ✓ | 77.7ms |
| L3_lexical | 0.432 | ✓ | 0.5ms |

### ✗ `medium-04-derin-ogrenme` (skor: 0.579, zorluk: medium, kategori: analytical, **CRITICAL**)

**Soru:** Derin öğrenme ile makine öğrenmesi arasındaki fark nedir?

**Cevap:** Bu konuda verilen bilgilerde yeterli detay bulunmuyor. Ancak, "Optimizasyon ve makine öğrenimi arasındaki fark, genelleştirme hedefinden kaynaklanır" ifadesi bulunmakta olup, derin öğrenme ve makine öğrenmesi arasındaki fark açıkça belirtilmemektedir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.800 | ✓ | 0.1ms |
| L2_vector | 0.806 | ✓ | 61.9ms |
| L3_lexical | 0.132 | ✗ | 0.6ms |

### ✓ `hard-01-python-kullanim-yerleri` (skor: 0.798, zorluk: hard, kategori: synthesis)

**Soru:** Hangi büyük kurumlar ve projeler Python kullanır?

**Cevap:** Google, NASA, CERN, Wikipedia, Facebook, Amazon, Instagram ve Spotify gibi büyük kurumlar ve YouTube, orijinal BitTorrent istemcisi, Django, Zope uygulama sunucuları gibi projeler Python kullanır.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.942 | ✓ | 64.2ms |
| L3_lexical | 0.451 | ✓ | 0.6ms |

### ✓ `hard-02-algoritma-elestirisi` (skor: 0.772, zorluk: hard, kategori: analytical)

**Soru:** Algoritmalara yöneltilen toplumsal eleştiriler nelerdir?

**Cevap:** Algoritmaların toplumsal gruplar arasındaki eşitsizlikleri derinleştirdikleri, yeni güç dengeleri oluşturdukları ve farklı otoriteler tarafından toplumun belli kesimlerini baskılamak için kullanıldıklarının altı çizilmektedir.

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.1ms |
| L2_vector | 0.882 | ✓ | 66.4ms |
| L3_lexical | 0.435 | ✓ | 0.5ms |

### ✓ `hard-03-multi-hop` (skor: 0.760, zorluk: hard, kategori: synthesis, **CRITICAL**)

**Soru:** Python'dan etkilenen programlama dilleri hangileridir ve bu diller hangi özellikleri ödünç almıştır?

**Cevap:** Python'dan etkilenen programlama dilleri arasında Boo, Cobra, CoffeeScript, ECMAScript/JavaScript, GDScript, Go, Groovy ve Swift bulunur. Bu diller, Python'dan çeşitli özellikleri ödünç almışlardır. Örneğin:

- Boo, Python'dan benzer bir girintileme, sözdizimi ve nesne modeli kullanır.
- Cobra, Pyth…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 0.920 | ✓ | 0.2ms |
| L2_vector | 0.931 | ✓ | 77.1ms |
| L3_lexical | 0.428 | ✓ | 1.9ms |

### ✗ `edge-01-out-of-domain` (skor: 0.518, zorluk: hard, kategori: edge_case, **CRITICAL**)

**Soru:** Türkiye'de 2023 deprem felaketinin etkileri nelerdir?

**Cevap:** Bu konuda mevcut verilerimde yeterli detay bulunamadı, ancak genel bilgilerim şunlar:

Merhaba, 2023 yılında yaşanan deprem felaketi Türkiye'de büyük bir yıkıma neden oldu. Depremin etkileri arasında binlerce insanın hayatını kaybetmesi, çok sayıda kişinin yaralanması ve evsiz kalması yer alıyor. Ay…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.4ms |
| L2_vector | 0.522 | ✗ | 41.2ms |
| L3_lexical | 0.033 | ✗ | 1.7ms |

### ✗ `edge-02-bos-soru` (skor: 0.525, zorluk: easy, kategori: edge_case)

**Soru:** ?

**Cevap:** Bu konuda mevcut verilerimde yeterli detay bulunamadı, ancak genel bilgilerim şunlar:

Soru sormak için buradasın, ancak henüz bir soru sormadın. Lütfen sana nasıl yardımcı olabileceğimi söyleyerek bir soru sor.

---
📚 **Available Data Topics:**
Topics: Bilgisayarla Görme, Genetik, Nasa, Veri Bilimi…

| Katman | Skor | Geçti | Latency |
|---|---|---|---|
| L1_rules | 1.000 | ✓ | 0.2ms |
| L2_vector | 0.574 | ✗ | 36.0ms |
| L3_lexical | 0.000 | ✗ | 0.3ms |
