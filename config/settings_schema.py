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

    return RequestSettings(
        provider=provider,
        api_key=(headers.get("X-API-Key") or "").strip() or None,
        model=(headers.get("X-Model") or "").strip() or None,
        llm_params=LLMParams.from_json_string(headers.get("X-LLM-Params")),
        retrieval_strategy=strategy,
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
