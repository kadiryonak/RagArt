"""L2 — Vektörel benzerlik (semantik) değerlendirme.

NE ÖLÇER?
    Anlamsal yakınlık: "üretilen cevap, referans cevaba ANLAMCA ne kadar yakın?"
    Kelime tutmasa bile aynı anlamı taşıyorsa yüksek skor verir.

NASIL ÇALIŞIR?
    1. Hem üretilen cevap hem de referans cevap aynı embedding modeli ile
       vektöre çevrilir.
    2. İki vektör arasındaki cosine similarity hesaplanır:

            cos(θ) = (u · v) / (||u|| · ||v||)        ∈ [-1, 1]

    3. [0, 1] aralığına normalize edilir: (cos + 1) / 2

    Ekstra: cevap ile retrieved_context arasındaki benzerlik de hesaplanır
    (groundedness sinyali — cevap context'te mi yoksa havada mı?).

NE ZAMAN KULLANILIR?
    Genel kalite ölçümünde yararlı. BLEU/ROUGE'un kaçırdığı parafrazları
    yakalar. Embedding modeli zaten retrieval için yüklü olduğundan ek
    maliyeti minimaldir.

SINIRLAR
    - Embedding modeline bağımlı: model zayıfsa skor güvenilir değil.
    - "Doğru ama embedding'in farklı bulduğu" cevapları cezalandırabilir.
    - Türkçe için multilingual-MiniLM gibi modeller orta kalitedir;
      paraphrase-multilingual-MiniLM-L12-v2 (mevcut model) iyi bir başlangıç.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from tests.evaluation.dataset import GoldenItem
from tests.evaluation.layers.base import BaseEvaluator, RAGOutput

# Embedder protokolü: embed_query(text) -> List[float]
EmbedFn = Callable[[str], List[float]]


def _dot(u: List[float], v: List[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def _norm(u: List[float]) -> float:
    return sum(a * a for a in u) ** 0.5


def cosine_similarity(u: List[float], v: List[float]) -> float:
    """[-1, 1] aralığında cosine similarity. Sıfır vektörlere karşı dayanıklı."""
    nu, nv = _norm(u), _norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return _dot(u, v) / (nu * nv)


def normalize_to_unit(cos: float) -> float:
    """[-1, 1] → [0, 1]."""
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


class L2VectorEvaluator(BaseEvaluator):
    """L2 — embedding-based semantic similarity.

    Args:
        embed_fn: Tek bir string alıp embedding vektörü döndüren callable.
                  Üretimde EmbeddingManager.embed_query bağlanır;
                  testlerde mock embedder verilir.
        ctx_weight: Cevap–referans benzerliği vs. cevap–context benzerliği
                    ağırlığı. Default: referans daha önemli.
    """

    name = "L2_vector"
    threshold = 0.65  # 0.65 ≈ orta-yüksek semantik benzerlik

    def __init__(self, embed_fn: EmbedFn, ctx_weight: float = 0.3):
        assert 0.0 <= ctx_weight <= 1.0
        self.embed_fn = embed_fn
        self.ctx_weight = ctx_weight
        self.ref_weight = 1.0 - ctx_weight

    def _evaluate(self, item: GoldenItem, output: RAGOutput) -> tuple[float, Dict[str, Any]]:
        answer = (output.answer or "").strip()
        if not answer:
            return 0.0, {"reason": "empty_answer"}

        ans_vec = self.embed_fn(answer)
        ref_vec = self.embed_fn(item.reference_answer)

        ref_cos = cosine_similarity(ans_vec, ref_vec)
        ref_score = normalize_to_unit(ref_cos)

        details: Dict[str, Any] = {
            "answer_vs_reference": {"cosine": round(ref_cos, 4), "normalized": round(ref_score, 4)},
        }

        ctx_score: Optional[float] = None
        if output.retrieved_context:
            ctx_vec = self.embed_fn(output.retrieved_context)
            ctx_cos = cosine_similarity(ans_vec, ctx_vec)
            ctx_score = normalize_to_unit(ctx_cos)
            details["answer_vs_context"] = {
                "cosine": round(ctx_cos, 4),
                "normalized": round(ctx_score, 4),
            }
            total = self.ref_weight * ref_score + self.ctx_weight * ctx_score
        else:
            total = ref_score
            details["answer_vs_context"] = None

        return total, details
