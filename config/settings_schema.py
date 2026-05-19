"""Settings schema — frontend ↔ backend için kontrat.

BYOK (Bring Your Own Key) pattern'i uygular:
    - Kullanıcı tüm ayarları localStorage'da tutar (frontend)
    - Her API isteğinde header'larla server'a yollanır
    - Server stateless: key/ayar saklamaz, isteğe göre provider örnekler

Server bu modülü iki amaçla kullanır:
    1. /settings endpoint → frontend'e mevcut şemayı (provider listesi, default
       değerler, hangi param hangi provider'a ait) sunmak
    2. /ask endpoint → header'lardan LLM params'ı parse edip validate etmek
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# Tüm provider'larda ortak parametreler (cloud + local)
COMMON_LLM_PARAMS = {
    "temperature": {"type": "float", "min": 0.0, "max": 2.0, "default": 0.1, "step": 0.05},
    "top_p":       {"type": "float", "min": 0.0, "max": 1.0, "default": 0.9, "step": 0.05},
}

# Provider'a özel ek parametreler
PROVIDER_PARAMS: Dict[str, Dict[str, Any]] = {
    "deepseek": {
        **COMMON_LLM_PARAMS,
        "max_tokens": {"type": "int", "min": 64, "max": 8192, "default": 800, "step": 64},
    },
    "openai": {
        **COMMON_LLM_PARAMS,
        "max_tokens": {"type": "int", "min": 64, "max": 8192, "default": 500, "step": 64},
        "frequency_penalty": {"type": "float", "min": -2.0, "max": 2.0, "default": 0.0, "step": 0.1},
    },
    "groq": {
        **COMMON_LLM_PARAMS,
        "max_tokens": {"type": "int", "min": 64, "max": 8192, "default": 800, "step": 64},
    },
    "ollama": {
        **COMMON_LLM_PARAMS,
        "num_ctx":        {"type": "int",   "min": 512, "max": 32768, "default": 2048, "step": 512},
        "num_predict":    {"type": "int",   "min": 64,  "max": 8192,  "default": 500,  "step": 64},
        "repeat_penalty": {"type": "float", "min": 0.5, "max": 2.0,   "default": 1.1,  "step": 0.05},
    },
    "huggingface": {
        **COMMON_LLM_PARAMS,
        "max_tokens": {"type": "int", "min": 64, "max": 8192, "default": 500, "step": 64},
    },
    "local": {},  # local provider parametre kullanmıyor
}

# Her provider'ın default modeli (kullanıcı UI'dan değiştirebilir)
PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1:8b",
    "huggingface": "meta-llama/Llama-3.1-8B-Instruct",
    "local": "context-aware-fallback",
}

# Provider için API key gerekli mi?
PROVIDER_NEEDS_KEY: Dict[str, bool] = {
    "deepseek": True,
    "openai": True,
    "groq": True,
    "ollama": False,
    "huggingface": True,
    "local": False,
}


@dataclass
class LLMParams:
    """Bir LLM çağrısına geçirilen runtime parametreler.

    Sadece set edilen alanlar provider'a gönderilir; geri kalan default'lar
    provider tarafında tutulur. Bu sayede client kısmi override yapabilir.
    """

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    num_ctx: Optional[int] = None
    num_predict: Optional[int] = None
    repeat_penalty: Optional[float] = None

    def to_dict(self, *, drop_none: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        if drop_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d

    @classmethod
    def from_json_string(cls, raw: Optional[str]) -> "LLMParams":
        """Header'dan gelen JSON string'i parse et. Hatalı format → boş params."""
        if not raw:
            return cls()
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return cls()
        if not isinstance(data, dict):
            return cls()

        # Sadece tanıdığı alanları al; bilinmeyenleri sessizce at
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def validate(self, provider: str) -> List[str]:
        """Provider'a göre param sınırlarını kontrol et. Hata listesi döner."""
        errors: List[str] = []
        spec = PROVIDER_PARAMS.get(provider, {})
        for field_name, value in self.to_dict().items():
            if field_name not in spec:
                continue  # provider bu parametreyi desteklemiyor → sessizce yok say
            s = spec[field_name]
            if s["type"] == "int" and not isinstance(value, int):
                errors.append(f"{field_name} must be int")
                continue
            if s["type"] == "float" and not isinstance(value, (int, float)):
                errors.append(f"{field_name} must be number")
                continue
            if value < s["min"] or value > s["max"]:
                errors.append(
                    f"{field_name}={value} out of range [{s['min']}, {s['max']}]"
                )
        return errors


@dataclass
class RequestSettings:
    """Tek bir /ask isteğinin parsed ayarları (header'lardan)."""

    provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    llm_params: LLMParams = field(default_factory=LLMParams)
    retrieval_strategy: Optional[str] = None  # 'dense' | 'sparse' | 'hybrid' | None
    rerank: bool = False                       # cross-encoder rerank toggle
    rerank_fetch_k: int = 20                   # aday sayısı
    memory_strategy: Optional[str] = None      # 'none' | 'sliding_window' | 'summary_buffer' | 'vector'
    history: list = field(default_factory=list)  # list[dict] {role, content}
    deduplicate_context: bool = False           # redundancy filter (cosine sim)
    reorder_context: bool = False               # lost-in-the-middle reorder
    max_context_tokens: Optional[int] = None    # token budget cap
    allow_general_knowledge_fallback: bool = False  # opt-in hallucination risk


def parse_request_settings(headers) -> RequestSettings:
    """Flask request.headers'tan RequestSettings üret.

    Header isimleri (case-insensitive):
        X-Provider:    deepseek | openai | groq | ollama | huggingface | local
        X-API-Key:     cloud provider için key
        X-Model:       opsiyonel — default modeli override eder
        X-LLM-Params:  JSON string, LLMParams alanları
    """
    provider = (headers.get("X-Provider") or "").strip().lower() or None
    if provider == "":
        provider = None

    strategy = (headers.get("X-Retrieval-Strategy") or "").strip().lower() or None
    if strategy and strategy not in ("dense", "sparse", "hybrid"):
        strategy = None  # bilinmeyen değer → sessizce default

    rerank_raw = (headers.get("X-Rerank") or "").strip().lower()
    rerank = rerank_raw in ("1", "true", "yes", "on")

    rerank_fetch_k = 20
    try:
        rfk = headers.get("X-Rerank-Fetch-K")
        if rfk:
            rerank_fetch_k = max(1, min(200, int(rfk)))
    except (ValueError, TypeError):
        pass

    mem = (headers.get("X-Memory-Strategy") or "").strip().lower() or None
    if mem and mem not in ("none", "sliding_window", "summary_buffer", "vector"):
        mem = None

    history: list = []
    hist_raw = headers.get("X-Conversation-History")
    if hist_raw:
        # The client base64-encodes the UTF-8 JSON because HTTP header values
        # are restricted to ISO-8859-1 (browsers reject Turkish chars in raw
        # JSON headers). New clients send "b64:..." prefix; older clients
        # might still send plain JSON, so try both.
        decoded_raw: str | None = None
        if hist_raw.startswith("b64:"):
            import base64
            try:
                decoded_raw = base64.b64decode(hist_raw[4:]).decode("utf-8")
            except Exception:
                decoded_raw = None
        else:
            decoded_raw = hist_raw

        if decoded_raw is not None:
            try:
                data = json.loads(decoded_raw)
                if isinstance(data, list):
                    # Cap to a sensible max so a huge header can't OOM
                    history = data[-200:]
            except (ValueError, TypeError):
                history = []

    def _bool_header(name: str) -> bool:
        return (headers.get(name) or "").strip().lower() in ("1", "true", "yes", "on")

    dedup = _bool_header("X-Context-Deduplicate")
    reorder = _bool_header("X-Context-Reorder")
    allow_general_kb = _bool_header("X-Allow-General-Knowledge")

    max_ctx_tokens: Optional[int] = None
    raw_budget = headers.get("X-Context-Max-Tokens")
    if raw_budget:
        try:
            max_ctx_tokens = max(64, min(32000, int(raw_budget)))
        except (ValueError, TypeError):
            max_ctx_tokens = None

    return RequestSettings(
        provider=provider,
        api_key=(headers.get("X-API-Key") or "").strip() or None,
        model=(headers.get("X-Model") or "").strip() or None,
        llm_params=LLMParams.from_json_string(headers.get("X-LLM-Params")),
        retrieval_strategy=strategy,
        rerank=rerank,
        rerank_fetch_k=rerank_fetch_k,
        memory_strategy=mem,
        history=history,
        deduplicate_context=dedup,
        reorder_context=reorder,
        max_context_tokens=max_ctx_tokens,
        allow_general_knowledge_fallback=allow_general_kb,
    )


def get_settings_schema() -> Dict[str, Any]:
    """/settings GET için tam şema. Frontend bunu okuyup UI'ı dinamik üretir."""
    return {
        "providers": [
            {
                "id": pid,
                "needs_key": PROVIDER_NEEDS_KEY[pid],
                "default_model": PROVIDER_DEFAULT_MODELS[pid],
                "params": PROVIDER_PARAMS[pid],
            }
            for pid in ("deepseek", "openai", "groq", "ollama", "huggingface", "local")
        ],
        "retrieval_strategies": [
            {"id": "auto",   "label": "Otomatik (önerilen)",
             "desc": "Hybrid varsa hybrid, yoksa dense."},
            {"id": "dense",  "label": "Dense (embedding)",
             "desc": "Anlamsal arama. Paraphrase'lere güçlü, exact match'lere zayıf."},
            {"id": "sparse", "label": "Sparse (BM25)",
             "desc": "Term-bazlı. Özel isim/kısaltma/kod için güçlü."},
            {"id": "hybrid", "label": "Hybrid (RRF)",
             "desc": "Dense + BM25 birleşimi. Production standardı."},
        ],
        "rerank": {
            "available": True,
            "desc": "Cross-encoder ile son aşama yeniden sıralama. "
                    "İlk çağrıda ~400MB model indirir, sonra cache'lenir. "
                    "Her sorguya ~500-1000ms ekler ama relevance ciddi artar.",
            "default_fetch_k": 20,
        },
        "context_engineering": {
            "deduplicate": {
                "desc": "Cosine similarity > 0.92 olan chunk'ları temizle "
                        "(token israfı + lost-in-the-middle riskini azaltır).",
            },
            "reorder": {
                "desc": "Lost-in-the-middle: en alakalı chunk'ları context'in "
                        "başına ve sonuna yerleştir, zayıfları ortaya at.",
            },
            "max_tokens": {
                "desc": "Toplam context budget (token cinsinden). "
                        "Aşan kuyruktan kesilir; en az 1 chunk her zaman tutulur.",
                "min": 64,
                "max": 32000,
            },
        },
        "memory_strategies": [
            {"id": "none",          "label": "Hafıza yok",
             "desc": "Her soru izole. Tek-turn senaryolar için ideal."},
            {"id": "sliding_window", "label": "Kayar pencere (son N turn)",
             "desc": "Son 5 sohbet adımı ham olarak prompt'a eklenir. "
                     "Sıfır maliyet, deterministik."},
            {"id": "summary_buffer", "label": "Özetli buffer",
             "desc": "Eski turn'ler LLM ile özetlenir, son 4'ü ham tutulur. "
                     "Uzun sohbetlerde token tasarrufu."},
            {"id": "vector",         "label": "Semantik retrieval",
             "desc": "Soru ile embedding olarak en alakalı geçmiş turn'leri "
                     "çeker. Çok uzun sohbet için."},
        ],
        "param_descriptions_tr": {
            "temperature": "Yaratıcılık. Düşük → tutarlı/kuralcı, yüksek → çeşitli/yaratıcı.",
            "top_p": "Nucleus sampling. 1.0 = tüm seçenekler, 0.5 = en olası %50.",
            "max_tokens": "Maksimum cevap uzunluğu (token cinsinden).",
            "frequency_penalty": "Aynı kelime tekrarına ceza. Yüksek → çeşitlilik.",
            "num_ctx": "Context window büyüklüğü (Ollama).",
            "num_predict": "Üretilecek max token sayısı (Ollama).",
            "repeat_penalty": "Tekrar cezası (Ollama). 1.0 = ceza yok.",
        },
    }
