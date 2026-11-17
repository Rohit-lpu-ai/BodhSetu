"""LLM providers for grounded answer generation.

Two real providers, both spoken to over plain HTTP with ``httpx`` — which the
backend already pins, so grounded generation adds **no new dependency**:

* :class:`OpenAICompatibleProvider` — the ``/chat/completions`` shape. That one
  wire format covers OpenAI, Groq, Together, OpenRouter, DeepSeek, Mistral's
  compatible endpoint, Azure OpenAI, LM Studio and vLLM. Pointing at a
  different vendor is a base URL and a key, not a new SDK.
* :class:`OllamaProvider` — a local Ollama daemon. No key, no account, nothing
  leaves the machine, which matters for a project whose corpus is public but
  whose users' questions need not be.

and one for tests:

* :class:`MockProvider` — returns a scripted reply. Used only in unit tests;
  it is never selected by :func:`resolve_provider` unless explicitly asked for,
  so a mock can never be mistaken for a configured model.

What this module will not do
----------------------------
It will not guess that a provider is available because a package is installed,
and it will not quietly downgrade to something that invents text. If nothing is
configured, :func:`resolve_provider` returns ``None`` and the caller is
expected to say so — see ``ProviderUnavailable`` for the exact message the user
should see.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "LLMError",
    "ProviderUnavailable",
    "GenerationTimeout",
    "InvalidModelResponse",
    "BaseLLMProvider",
    "OpenAICompatibleProvider",
    "OllamaProvider",
    "MockProvider",
    "resolve_provider",
    "describe_configuration",
]


class LLMError(RuntimeError):
    """Base class for generation failures. Carries a stable ``code``."""

    code = "LLM_ERROR"


class ProviderUnavailable(LLMError):
    """No provider is configured, or the configured one cannot be reached."""

    code = "LLM_PROVIDER_UNAVAILABLE"


class GenerationTimeout(LLMError):
    """The provider did not answer within the configured timeout."""

    code = "LLM_TIMEOUT"


class InvalidModelResponse(LLMError):
    """The provider answered, but not with anything usable."""

    code = "LLM_INVALID_RESPONSE"


@dataclass
class LLMResult:
    """One completion, plus what it cost and who produced it."""

    text: str
    model: str
    provider: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None


class BaseLLMProvider(ABC):
    """A chat-completion endpoint."""

    name: str = "base"

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_seconds: float = 60.0,
    ) -> LLMResult:
        """Return one completion, or raise an :class:`LLMError` subclass."""

    def health(self) -> bool:
        """Cheap reachability check. False rather than raising."""
        return True


def _httpx():
    """Import httpx lazily so importing this module never needs it."""
    try:
        import httpx  # noqa: PLC0415 - deliberately lazy
    except ImportError as exc:  # pragma: no cover - httpx is a pinned dep
        raise ProviderUnavailable(
            "httpx is required for HTTP LLM providers but is not installed. "
            "Install it with: .venv\\Scripts\\python -m pip install httpx"
        ) from exc
    return httpx


class OpenAICompatibleProvider(BaseLLMProvider):
    """Any endpoint speaking the OpenAI ``/chat/completions`` shape."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        api_key: Optional[str],
        model: str,
        base_url: str = "https://api.openai.com/v1",
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.extra_headers = dict(extra_headers or {})

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_seconds: float = 60.0,
    ) -> LLMResult:
        httpx = _httpx()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        import time

        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise GenerationTimeout(
                f"the model did not answer within {timeout_seconds:.0f}s"
            ) from exc
        except httpx.HTTPError as exc:
            # The URL can contain a host but never a key: keys live in headers.
            raise ProviderUnavailable(
                f"could not reach the model endpoint at {self.base_url}: "
                f"{type(exc).__name__}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code == 401 or response.status_code == 403:
            raise ProviderUnavailable(
                "the model endpoint rejected the credentials "
                f"(HTTP {response.status_code}). Check NIYAMSETU_RAG_LLM_API_KEY."
            )
        if response.status_code == 429:
            raise ProviderUnavailable(
                "the model endpoint is rate limiting this key (HTTP 429). "
                "Wait and retry, or use a different provider."
            )
        if response.status_code >= 400:
            raise ProviderUnavailable(
                f"the model endpoint returned HTTP {response.status_code}. "
                f"{_safe_error_detail(response.text)}"
            )

        try:
            body = response.json()
            choices = body["choices"]
            text = choices[0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise InvalidModelResponse(
                "the model endpoint answered with a body this client could not "
                f"read ({type(exc).__name__})."
            ) from exc

        if not isinstance(text, str) or not text.strip():
            raise InvalidModelResponse("the model returned an empty answer.")

        usage = body.get("usage") or {}
        return LLMResult(
            text=text.strip(),
            model=str(body.get("model") or self.model),
            provider=self.name,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
        )

    def health(self) -> bool:
        if not self.api_key and "127.0.0.1" not in self.base_url and "localhost" not in self.base_url:
            # A hosted endpoint with no key will only fail later, louder.
            return False
        try:
            httpx = _httpx()
            response = httpx.get(
                f"{self.base_url}/models", headers=self._headers(), timeout=5.0
            )
            return response.status_code < 400
        except Exception:  # noqa: BLE001 - health must never raise
            return False


class OllamaProvider(BaseLLMProvider):
    """A local Ollama daemon (default ``http://127.0.0.1:11434``)."""

    name = "ollama"

    def __init__(
        self,
        *,
        model: str = "llama3.1",
        base_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_seconds: float = 120.0,
    ) -> LLMResult:
        httpx = _httpx()
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        import time

        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat", json=payload, timeout=timeout_seconds
            )
        except httpx.TimeoutException as exc:
            raise GenerationTimeout(
                f"Ollama did not answer within {timeout_seconds:.0f}s. A larger "
                "model on CPU can exceed this; raise NIYAMSETU_RAG_LLM_TIMEOUT."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"could not reach Ollama at {self.base_url}. Is `ollama serve` "
                f"running? ({type(exc).__name__})"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code == 404:
            raise ProviderUnavailable(
                f"Ollama has no model named {self.model!r}. "
                f"Pull it first: ollama pull {self.model}"
            )
        if response.status_code >= 400:
            raise ProviderUnavailable(
                f"Ollama returned HTTP {response.status_code}. "
                f"{_safe_error_detail(response.text)}"
            )

        try:
            body = response.json()
            text = body["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise InvalidModelResponse(
                f"Ollama answered with an unreadable body ({type(exc).__name__})."
            ) from exc

        if not isinstance(text, str) or not text.strip():
            raise InvalidModelResponse("Ollama returned an empty answer.")

        return LLMResult(
            text=text.strip(),
            model=str(body.get("model") or self.model),
            provider=self.name,
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            latency_ms=latency_ms,
        )

    def health(self) -> bool:
        try:
            httpx = _httpx()
            response = httpx.get(f"{self.base_url}/api/tags", timeout=3.0)
            return response.status_code < 400
        except Exception:  # noqa: BLE001
            return False


class MockProvider(BaseLLMProvider):
    """A scripted provider. Tests only — never auto-selected."""

    name = "mock"

    def __init__(self, reply: str = "", *, fail_with: Optional[Exception] = None) -> None:
        self.reply = reply
        self.fail_with = fail_with
        self.calls: List[Dict[str, Any]] = []

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout_seconds: float = 60.0,
    ) -> LLMResult:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.fail_with is not None:
            raise self.fail_with
        return LLMResult(text=self.reply, model="mock-model", provider=self.name)


def _safe_error_detail(body: str, limit: int = 200) -> str:
    """A short, credential-free fragment of an error body for the user.

    Error bodies routinely echo request fields. Anything that looks like a key
    is removed rather than relayed into a log line or an API response.
    """
    fragment = (body or "").strip().replace("\n", " ")[:limit]
    for marker in ("sk-", "Bearer ", "api_key", "apikey", "api-key"):
        if marker.lower() in fragment.lower():
            return "(error detail withheld: it contained credential-like text)"
    return fragment


def resolve_provider(settings: Any) -> Optional[BaseLLMProvider]:
    """Build the configured provider, or ``None`` if none is configured.

    ``None`` is not a failure to handle quietly — it means the deployment has
    no model and the caller must say so. The one thing this function will never
    do is return a provider that makes text up.

    Recognised ``llm_provider`` values:

    * ``openai`` / ``openai_compatible`` / ``groq`` / ``together`` /
      ``openrouter`` / ``deepseek`` / ``azure`` — the chat-completions shape.
      Needs ``llm_api_key``; ``llm_base_url`` defaults per vendor.
    * ``ollama`` — a local daemon; no key.
    * ``mock`` — the scripted provider, for tests.
    * ``none`` / ``off`` / anything unset — no provider.
    * ``auto`` — Ollama if it answers, else the compatible shape if a key is
      present, else ``None``.
    """
    provider = str(getattr(settings, "llm_provider", "") or "").strip().lower()
    model = str(getattr(settings, "llm_model_name", "") or "").strip()
    api_key = getattr(settings, "llm_api_key", None)
    base_url = getattr(settings, "llm_base_url", None)

    if provider in {"", "none", "off", "disabled"}:
        return None

    if provider == "mock":
        return MockProvider(reply=os.environ.get("NIYAMSETU_RAG_MOCK_REPLY", ""))

    if provider == "ollama":
        return OllamaProvider(
            model=model or "llama3.1",
            base_url=base_url or "http://127.0.0.1:11434",
        )

    if provider in _OPENAI_COMPATIBLE_DEFAULTS:
        resolved_base = base_url or _OPENAI_COMPATIBLE_DEFAULTS[provider]["base_url"]
        resolved_model = model or _OPENAI_COMPATIBLE_DEFAULTS[provider]["model"]
        is_local = "127.0.0.1" in resolved_base or "localhost" in resolved_base
        if not api_key and not is_local:
            raise ProviderUnavailable(
                f"provider {provider!r} needs an API key. Set "
                "NIYAMSETU_RAG_LLM_API_KEY in .env (it is git-ignored)."
            )
        return OpenAICompatibleProvider(
            api_key=api_key, model=resolved_model, base_url=resolved_base
        )

    if provider == "auto":
        ollama = OllamaProvider(
            model=model or "llama3.1", base_url=base_url or "http://127.0.0.1:11434"
        )
        if ollama.health():
            return ollama
        if api_key:
            return OpenAICompatibleProvider(
                api_key=api_key,
                model=model or "gpt-4o-mini",
                base_url=base_url or "https://api.openai.com/v1",
            )
        return None

    raise ProviderUnavailable(
        f"unknown llm_provider {provider!r}. Supported: "
        f"{', '.join(sorted(_OPENAI_COMPATIBLE_DEFAULTS))}, ollama, auto, mock, none."
    )


#: Base URL and a sensible small model per vendor speaking the OpenAI shape.
_OPENAI_COMPATIBLE_DEFAULTS: Dict[str, Dict[str, str]] = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "openai_compatible": {"base_url": "http://127.0.0.1:1234/v1", "model": "local-model"},
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct",
    },
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "mistral": {"base_url": "https://api.mistral.ai/v1", "model": "mistral-small-latest"},
}


def describe_configuration(settings: Any) -> Dict[str, Any]:
    """What is configured, for a status endpoint. Never includes the key."""
    provider = str(getattr(settings, "llm_provider", "") or "none").strip().lower()
    has_key = bool(getattr(settings, "llm_api_key", None))
    return {
        "provider": provider,
        "model": str(getattr(settings, "llm_model_name", "") or ""),
        "base_url": getattr(settings, "llm_base_url", None),
        "api_key_present": has_key,
        "supported_providers": sorted(
            set(_OPENAI_COMPATIBLE_DEFAULTS) | {"ollama", "auto", "none"}
        ),
    }
