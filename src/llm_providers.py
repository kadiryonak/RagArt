"""LLM provider implementations for the RAG system.

Each provider:
    - Accepts an optional ``model`` override at construction.
    - Implements ``generate(prompt, **runtime_params)`` where runtime_params
      override the provider's defaults for that single call.
    - Implements ``generate_general(question, **runtime_params)`` for the
      no-context fallback path.

Runtime params are a thin dict (see config.settings_schema.LLMParams) that
maps to each provider's native API field. The mapping is intentional:
clients pass standard names ("max_tokens", "num_predict", "repeat_penalty")
and the provider translates.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional

import requests

from src.utils import get_logger, StatusEmoji

logger = get_logger(__name__)


def _merge(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """defaults + overrides; None değerleri at."""
    out = dict(defaults)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, **params: Any) -> str:
        """Generate a response for the given prompt with optional runtime params."""

    @abstractmethod
    def generate_general(self, question: str, **params: Any) -> str:
        """Generate a response without RAG context."""

    def generate_stream(self, prompt: str, **params: Any) -> Iterator[str]:
        """Yield the response incrementally, chunk by chunk.

        Default implementation: a single chunk holding the full answer —
        providers with real token streaming (e.g. Groq) override this.
        Callers can always iterate generate_stream() regardless of whether
        the concrete provider streams natively.
        """
        yield self.generate(prompt, **params)


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API provider (OpenAI-compatible chat completions)."""

    API_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.defaults: Dict[str, Any] = {
            "temperature": 0.1,
            "max_tokens": 800,
            "top_p": 0.9,
        }
        self.timeout = 30

    def generate(self, prompt: str, **params: Any) -> str:
        cfg = _merge(self.defaults, params)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": params.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "top_p": cfg["top_p"],
            "stream": False,
        }
        try:
            r = requests.post(self.API_URL, headers=headers, json=payload, timeout=self.timeout)
            if r.status_code != 200:
                logger.error(f"{StatusEmoji.ERROR} DeepSeek API {r.status_code}")
                return f"API error: {r.status_code}"
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"{StatusEmoji.ERROR} DeepSeek error: {e}")
            return f"Connection error: {e}"

    def generate_general(self, question: str, **params: Any) -> str:
        general_prompt = (
            "Sen Türkçe konuşan bir yapay zeka asistanısın. Kullanıcının sorusunu "
            "Türkçe, faydalı ve özlü şekilde yanıtla (maks. 3-4 paragraf).\n\n"
            f"Soru: {question}\n\nYanıt:"
        )
        return self.generate(general_prompt, **params)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.defaults: Dict[str, Any] = {
            "temperature": 0.1,
            "max_tokens": 500,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
        }
        self.timeout = 30

    def generate(self, prompt: str, **params: Any) -> str:
        cfg = _merge(self.defaults, params)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": params.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "top_p": cfg["top_p"],
            "frequency_penalty": cfg["frequency_penalty"],
        }
        try:
            r = requests.post(self.API_URL, headers=headers, json=payload, timeout=self.timeout)
            if r.status_code != 200:
                return f"OpenAI API error: {r.status_code}"
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenAI connection error: {e}"

    def generate_general(self, question: str, **params: Any) -> str:
        return self.generate(f"Soruyu Türkçe yanıtla: {question}", **params)


class GroqProvider(BaseLLMProvider):
    """Groq API provider — OpenAI-compatible, ücretsiz katman ile.

    Default model llama-3.3-70b-versatile: çok hızlı, Türkçe yetkin.
    Rate limit (TPM) yendiğinde 429 üretir; biz error mesajındaki
    "try again in Xs" ipucunu okuyup tek seferlik retry yapıyoruz.
    """

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    # 429 retry için maksimum bekleme (free tier TPM reset penceresi 60s)
    MAX_RETRY_WAIT_S = 75.0

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.defaults: Dict[str, Any] = {
            "temperature": 0.1,
            "max_tokens": 800,
            "top_p": 0.9,
        }
        self.timeout = 30

    @staticmethod
    def _parse_retry_wait(error_text: str) -> Optional[float]:
        """Groq 429 mesajından "try again in 8.123s" gibi ipucu çıkar."""
        import re
        m = re.search(r"try again in ([\d.]+)s", error_text or "")
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    def generate(self, prompt: str, **params: Any) -> str:
        cfg = _merge(self.defaults, params)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": params.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "top_p": cfg["top_p"],
        }
        attempt = 0
        while attempt < 2:
            try:
                r = requests.post(self.API_URL, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                if r.status_code == 429 and attempt == 0:
                    wait = self._parse_retry_wait(r.text) or 30.0
                    wait = min(wait + 0.5, self.MAX_RETRY_WAIT_S)
                    logger.warning(
                        f"{StatusEmoji.WARNING} Groq 429 rate limit — sleeping {wait:.1f}s"
                    )
                    import time
                    time.sleep(wait)
                    attempt += 1
                    continue
                logger.error(f"{StatusEmoji.ERROR} Groq API {r.status_code}: {r.text[:200]}")
                return f"Groq API error: {r.status_code}"
            except Exception as e:
                logger.error(f"{StatusEmoji.ERROR} Groq error: {e}")
                return f"Groq connection error: {e}"
        return "Groq API error: rate limit (after retry)"

    def generate_stream(self, prompt: str, **params: Any) -> Iterator[str]:
        """Stream the answer token-by-token via Groq's SSE chat completions.

        On any non-200 (e.g. a 429 rate limit) we fall back to the regular
        generate() — it owns the retry logic — and yield its result as a
        single chunk, so the caller's streaming loop still works.
        """
        cfg = _merge(self.defaults, params)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": params.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "top_p": cfg["top_p"],
            "stream": True,
        }
        try:
            r = requests.post(
                self.API_URL, headers=headers, json=payload,
                timeout=self.timeout, stream=True,
            )
            if r.status_code != 200:
                logger.warning(
                    f"{StatusEmoji.WARNING} Groq stream {r.status_code} — "
                    "falling back to non-streamed generate()"
                )
                yield self.generate(prompt, **params)
                return
            # SSE responses omit a charset, so requests defaults to
            # ISO-8859-1 and decode_unicode would turn UTF-8 Turkish bytes
            # into mojibake ("çözmek" → "Ã§Ã¶zmek"). Pin UTF-8 explicitly.
            r.encoding = "utf-8"
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (ValueError, KeyError, IndexError):
                    continue
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"{StatusEmoji.ERROR} Groq stream error: {e}")
            yield f"Groq connection error: {e}"

    def generate_general(self, question: str, **params: Any) -> str:
        general_prompt = (
            "Sen Türkçe konuşan bir yapay zeka asistanısın. Kullanıcının sorusunu "
            "Türkçe, faydalı ve özlü şekilde yanıtla.\n\n"
            f"Soru: {question}\n\nYanıt:"
        )
        return self.generate(general_prompt, **params)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) provider — Messages API.

    Uses the REST API directly via `requests` (no SDK dependency), matching
    the other providers. Note: Anthropic requires `max_tokens` on every call
    and uses `x-api-key` + `anthropic-version` headers (not Bearer).
    """

    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key
        self.model = model
        # NB: Claude 4.x rejects temperature + top_p together — we send only
        # one (see _payload). Default to temperature.
        self.defaults: Dict[str, Any] = {
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        self.timeout = 60

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

    def _payload(self, prompt: str, params: Dict[str, Any], *, stream: bool) -> Dict[str, Any]:
        cfg = _merge(self.defaults, params)
        payload: Dict[str, Any] = {
            "model": params.get("model", self.model),
            "max_tokens": cfg.get("max_tokens", 1024),   # required by the API
            "messages": [{"role": "user", "content": prompt}],
        }
        # Anthropic rejects temperature + top_p together — send exactly one,
        # preferring temperature.
        if cfg.get("temperature") is not None:
            payload["temperature"] = cfg["temperature"]
        elif cfg.get("top_p") is not None:
            payload["top_p"] = cfg["top_p"]
        if stream:
            payload["stream"] = True
        return payload

    def generate(self, prompt: str, **params: Any) -> str:
        try:
            r = requests.post(
                self.API_URL, headers=self._headers(),
                json=self._payload(prompt, params, stream=False), timeout=self.timeout,
            )
            if r.status_code != 200:
                logger.error(f"{StatusEmoji.ERROR} Anthropic API {r.status_code}: {r.text[:200]}")
                return f"Anthropic API error: {r.status_code}"
            blocks = r.json().get("content", [])
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        except Exception as e:
            logger.error(f"{StatusEmoji.ERROR} Anthropic error: {e}")
            return f"Anthropic connection error: {e}"

    def generate_stream(self, prompt: str, **params: Any) -> Iterator[str]:
        """Token-by-token via Anthropic's SSE stream. Falls back to
        generate() (single chunk) on any non-200."""
        try:
            r = requests.post(
                self.API_URL, headers=self._headers(),
                json=self._payload(prompt, params, stream=True),
                timeout=self.timeout, stream=True,
            )
            if r.status_code != 200:
                logger.warning(
                    f"{StatusEmoji.WARNING} Anthropic stream {r.status_code} — "
                    "falling back to non-streamed generate()"
                )
                yield self.generate(prompt, **params)
                return
            # See Groq note: pin UTF-8 so Turkish characters aren't mangled
            # by requests' ISO-8859-1 fallback for charset-less SSE.
            r.encoding = "utf-8"
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if obj.get("type") == "content_block_delta":
                    delta = obj.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield delta["text"]
                elif obj.get("type") == "message_stop":
                    break
        except Exception as e:
            logger.error(f"{StatusEmoji.ERROR} Anthropic stream error: {e}")
            yield f"Anthropic connection error: {e}"

    def generate_general(self, question: str, **params: Any) -> str:
        general_prompt = (
            "Sen Türkçe konuşan bir yapay zeka asistanısın. Kullanıcının sorusunu "
            "Türkçe, faydalı ve özlü şekilde yanıtla.\n\n"
            f"Soru: {question}\n\nYanıt:"
        )
        return self.generate(general_prompt, **params)


class OllamaProvider(BaseLLMProvider):
    """Ollama local API provider (no API key)."""

    API_URL = "http://localhost:11434/api/generate"

    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        self.defaults: Dict[str, Any] = {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 2048,
            "num_predict": 500,
            "repeat_penalty": 1.1,
        }
        self.timeout = 60

    def generate(self, prompt: str, **params: Any) -> str:
        cfg = _merge(self.defaults, params)
        payload = {
            "model": params.get("model", self.model),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": cfg["temperature"],
                "top_p": cfg["top_p"],
                "num_ctx": cfg["num_ctx"],
                "num_predict": cfg["num_predict"],
                "repeat_penalty": cfg["repeat_penalty"],
            },
        }
        try:
            r = requests.post(self.API_URL, json=payload, timeout=self.timeout)
            if r.status_code != 200:
                return f"Ollama error: {r.status_code}"
            return r.json()["response"]
        except Exception as e:
            return f"Ollama connection failed: {e}"

    def generate_general(self, question: str, **params: Any) -> str:
        return self.generate(f"Soruyu Türkçe yanıtla: {question}", **params)


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Inference API (OpenAI-compatible endpoint)."""

    API_URL = "https://router.huggingface.co/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.api_key = api_key
        self.model = model
        self.defaults: Dict[str, Any] = {
            "temperature": 0.1,
            "max_tokens": 500,
            "top_p": 0.9,
        }

    def generate(self, prompt: str, **params: Any) -> str:
        cfg = _merge(self.defaults, params)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": params.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "top_p": cfg["top_p"],
        }
        try:
            r = requests.post(self.API_URL, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                body = r.json()
                if body.get("choices"):
                    return body["choices"][0]["message"]["content"].strip()
                return "No response generated."
            if r.status_code == 402:
                return "HuggingFace kredi limiti doldu. PRO hesaba geçin veya local mode kullanın."
            return f"HuggingFace API error: {r.status_code} - {r.text[:200]}"
        except Exception as e:
            return f"HuggingFace error: {e}"

    def generate_general(self, question: str, **params: Any) -> str:
        return self.generate(f"Soruyu Türkçe yanıtla: {question}", **params)


class LocalProvider(BaseLLMProvider):
    """Local context-aware response generator (no API required).

    Params yok-sayılır (template-tabanlı çıktı).
    """

    def generate(self, prompt: str, **params: Any) -> str:
        if "BAĞLAM:" in prompt and "SORU:" in prompt:
            context_start = prompt.find("BAĞLAM:") + len("BAĞLAM:")
            context_end = prompt.find("SORU:")
            context = prompt[context_start:context_end].strip()
            question_start = prompt.find("SORU:") + len("SORU:")
            question_end = prompt.find("YANITIN:")
            question = prompt[question_start:question_end].strip()
            return self._generate_contextual_answer(context, question)
        return "Prompt format error."

    def generate_general(self, question: str, **params: Any) -> str:
        return "Bu konuda daha detaylı yanıt için bir API anahtarı gereklidir."

    def _generate_contextual_answer(self, context: str, question: str) -> str:
        if not context or len(context.strip()) < 10:
            return "Bu konuda verilen bilgilerde yeterli detay bulunmuyor."

        question_lower = question.lower()
        context_lower = context.lower()

        if "algoritma" in question_lower and "algoritma" in context_lower:
            lines = context.split("\n")
            relevant = [l for l in lines if "algoritma" in l.lower()]
            if relevant:
                return f"Algoritma bilgisi: {' '.join(relevant[:3])}"

        if any(w in question_lower for w in ["nedir", "ne"]):
            lines = context.split("\n")
            definitions = [l for l in lines if ":" in l and len(l) < 200]
            if definitions:
                return f"Tanım: {definitions[0]}. {definitions[1] if len(definitions) > 1 else ''}"
            return f"Mevcut bilgi: {context[:200]}{'...' if len(context) > 200 else ''}"

        if any(w in question_lower for w in ["hangi", "ne gibi", "nasıl"]):
            lines = context.split("\n")
            items = [l.strip() for l in lines if l.strip() and (l.startswith("-") or ":" in l)]
            if items:
                return f"İlgili maddeler: {', '.join(items[:5])}"

        relevant = [l for l in context.split("\n") if l.strip() and len(l) > 20][:3]
        if relevant:
            return f"İlgili içerik: {' '.join(relevant)}"
        return f"Bağlam: {context[:300]}{'...' if len(context) > 300 else ''}"


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

    PROVIDERS = {
        "deepseek": DeepSeekProvider,
        "openai": OpenAIProvider,
        "groq": GroqProvider,
        "anthropic": AnthropicProvider,
        "huggingface": HuggingFaceProvider,
        "ollama": OllamaProvider,
        "local": LocalProvider,
    }

    PROVIDERS_REQUIRING_KEY = {"deepseek", "openai", "groq", "anthropic", "huggingface"}

    @classmethod
    def create(
        cls,
        provider_type: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        provider_type = provider_type.lower()
        if provider_type not in cls.PROVIDERS:
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Available: {sorted(cls.PROVIDERS.keys())}"
            )
        cls_ = cls.PROVIDERS[provider_type]

        if provider_type in cls.PROVIDERS_REQUIRING_KEY:
            if not api_key:
                raise ValueError(f"{provider_type} requires an API key")
            if model:
                return cls_(api_key, model=model)
            return cls_(api_key)

        if provider_type == "ollama":
            if model:
                return cls_(model=model)
            return cls_()

        return cls_()

    @classmethod
    def get_available_providers(cls) -> list:
        return list(cls.PROVIDERS.keys())
