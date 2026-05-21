"""L3 — Lexical (kelime-bazlı) değerlendirme: BLEU + ROUGE-L.

NE ÖLÇER?
    Üretilen cevap ile referans cevap arasındaki KELİME ÖRTÜŞMESİ.
    Anlamı değil, "kelimeleri ne kadar tuttu" sorusunu cevaplar.

NASIL ÇALIŞIR?

    BLEU-n (n-gram precision):
        - Cevaptaki her n-gram referansta var mı?
        - precision_n = (eşleşen n-gram) / (cevaptaki toplam n-gram)
        - Brevity penalty: cevap referanstan kısaysa ceza (eksik cevap caydırma)
        - BLEU = BP · exp(Σ w_n · log(precision_n))
        Biz BLEU-1 ve BLEU-2 hesaplıyoruz (Türkçe için tatmin edici).

    ROUGE-L (longest common subsequence — LCS):
        - İki dizinin en uzun ortak alt dizisi (sıralı, bitişik olmak zorunda değil)
        - precision = LCS / len(candidate)
        - recall    = LCS / len(reference)
        - F1 = 2·P·R / (P+R)

    Bizim L3 skoru: 0.4·BLEU-1 + 0.2·BLEU-2 + 0.4·ROUGE-L_F1

NE ZAMAN KULLANILIR?
    Hızlı, deterministik, ücretsiz. Cevap referansa yakın FORMATTA olması
    beklenen sorularda iyi sinyal verir (özet, tanım, kısa cevap).

SINIRLAR
    - Parafraz cezalandırılır ("araba" vs. "oto" → düşük skor)
    - Türkçe çekimleri (-ler, -de, -den) skoru bozar; bizim çözümümüz:
      basit suffix stripping (lower-cost stemming heuristic).
    - Cevap doğru ama farklı kelimelerle yazıldıysa düşük skor verir.

PURE PYTHON
    Dış bağımlılık YOK. NLTK/sacrebleu/rouge_score gerektirmez —
    eval pipeline'ı kütüphane kurulum sorunlarından bağımsız çalışır.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

from tests.evaluation.dataset import GoldenItem
from tests.evaluation.layers.base import BaseEvaluator, RAGOutput

# Turkish tokenizer now lives in src/ (production code — BM25 uses it too).
from src.text_utils import tokenize


def _ngrams(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_n(candidate: List[str], reference: List[str], n: int) -> float:
    """Tek n için modifiye precision (clipping ile)."""
    cand_ngrams = _ngrams(candidate, n)
    ref_ngrams = _ngrams(reference, n)
    if not cand_ngrams:
        return 0.0

    clipped = 0
    for ngram, count in cand_ngrams.items():
        clipped += min(count, ref_ngrams.get(ngram, 0))
    total = sum(cand_ngrams.values())
    return clipped / total if total else 0.0


def brevity_penalty(candidate: List[str], reference: List[str]) -> float:
    c, r = len(candidate), len(reference)
    if c == 0:
        return 0.0
    if c > r:
        return 1.0
    return math.exp(1 - r / c)


def bleu_score(candidate: List[str], reference: List[str], max_n: int = 2) -> Dict[str, float]:
    """BLEU-1...BLEU-max_n. Geometric mean of precisions × brevity penalty."""
    bp = brevity_penalty(candidate, reference)
    precisions = [bleu_n(candidate, reference, n) for n in range(1, max_n + 1)]

    if any(p == 0 for p in precisions):
        geo = 0.0
    else:
        log_p = sum(math.log(p) for p in precisions) / max_n
        geo = math.exp(log_p)

    out: Dict[str, float] = {f"bleu_{i+1}_precision": p for i, p in enumerate(precisions)}
    out["brevity_penalty"] = bp
    out["bleu"] = bp * geo
    return out


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Klasik DP ile longest common subsequence uzunluğu. O(n·m)."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def rouge_l(candidate: List[str], reference: List[str], beta: float = 1.2) -> Dict[str, float]:
    """ROUGE-L F-measure (beta=1.2 standart)."""
    if not candidate or not reference:
        return {"rouge_l_p": 0.0, "rouge_l_r": 0.0, "rouge_l_f": 0.0, "lcs": 0}
    lcs = _lcs_length(candidate, reference)
    if lcs == 0:
        return {"rouge_l_p": 0.0, "rouge_l_r": 0.0, "rouge_l_f": 0.0, "lcs": 0}
    p = lcs / len(candidate)
    r = lcs / len(reference)
    f = (1 + beta**2) * p * r / (r + beta**2 * p)
    return {"rouge_l_p": p, "rouge_l_r": r, "rouge_l_f": f, "lcs": lcs}


class L3LexicalEvaluator(BaseEvaluator):
    """L3 — kelime-bazlı örtüşme (BLEU + ROUGE-L)."""

    name = "L3_lexical"
    threshold = 0.25  # NB: lexical eşleşme zor; eşik bilinçli düşük

    WEIGHTS = {"bleu_1": 0.4, "bleu_2": 0.2, "rouge_l_f": 0.4}

    def _evaluate(self, item: GoldenItem, output: RAGOutput) -> tuple[float, Dict[str, Any]]:
        cand = tokenize(output.answer or "")
        ref = tokenize(item.reference_answer)

        if not cand:
            return 0.0, {"reason": "empty_answer"}

        bleu = bleu_score(cand, ref, max_n=2)
        rouge = rouge_l(cand, ref)

        b1 = bleu["bleu_1_precision"] * bleu["brevity_penalty"]
        b2 = bleu["bleu"]  # full BLEU (n=2)
        rl = rouge["rouge_l_f"]

        score = (
            self.WEIGHTS["bleu_1"] * b1
            + self.WEIGHTS["bleu_2"] * b2
            + self.WEIGHTS["rouge_l_f"] * rl
        )

        return score, {
            "bleu_1": round(b1, 4),
            "bleu_2": round(b2, 4),
            "rouge_l_f": round(rl, 4),
            "rouge_l_p": round(rouge["rouge_l_p"], 4),
            "rouge_l_r": round(rouge["rouge_l_r"], 4),
            "brevity_penalty": round(bleu["brevity_penalty"], 4),
            "candidate_tokens": len(cand),
            "reference_tokens": len(ref),
        }
