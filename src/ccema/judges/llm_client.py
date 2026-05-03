"""Provider-agnostic LLM client interface used by judges and pi_agents.

We define a Protocol so the rest of the code can be tested with a
deterministic MockLLMClient and ship with thin adapters for Anthropic,
OpenAI, and Google. Concrete adapters are implemented lazily — they only
import their respective SDK at construction time so tests don't need
network or API keys.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    text: str
    model: str
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class LLMError(RuntimeError):
    """Common error type that all real adapters raise on provider failures.

    Wraps the underlying SDK exception in `__cause__` so callers can
    introspect, while presenting a uniform interface for retry / logging.
    """


class LLMClient(Protocol):
    name: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Best-effort detection of rate-limit / overloaded / 429 errors."""
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "overloaded" in name or "toomanyrequests" in name:
        return True
    msg = str(exc).lower()
    if "rate limit" in msg or "429" in msg or "overloaded" in msg or "quota" in msg:
        return True
    # SDKs commonly expose a status_code attribute
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status == 429


def _retry_call(fn, *, max_retries: int = 3, base_delay: float = 1.0):
    """Call `fn()` retrying only on rate-limit-shaped errors."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise non-retryable below
            if attempt >= max_retries or not _is_rate_limit_error(exc):
                raise
            # Exponential backoff with jitter
            delay = base_delay * (2**attempt) + random.uniform(0, 0.25)
            time.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------------
# Mock client (for tests + dry runs)
# ---------------------------------------------------------------------------


@dataclass
class MockLLMClient:
    """Deterministic mock LLM that returns canned responses by hashing inputs.

    Used for unit tests and offline pipeline verification.
    """

    name: str = "mock"
    canned_response: str | None = None
    return_alpha_beta: tuple[float, float] | None = None

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        if self.canned_response is not None:
            return LLMResponse(text=self.canned_response, model=self.name)

        # Default: produce a valid debate-judge JSON or pi_agent JSON
        if json_mode:
            if self.return_alpha_beta is not None:
                a, b = self.return_alpha_beta
                payload = {"rationale": "mock", "alpha": a, "beta": b}
            else:
                # Default favourable score
                payload = {"rationale": "mock", "alpha": 7.0, "beta": 3.0}
            return LLMResponse(text=json.dumps(payload), model=self.name)

        return LLMResponse(text="mock response", model=self.name)

    def health_check(self) -> dict[str, Any]:
        """Mock health check always succeeds."""
        return {"ok": True, "model": self.name, "error": None}


# ---------------------------------------------------------------------------
# Real providers — thin adapters, lazy SDK import
# ---------------------------------------------------------------------------


@dataclass
class AnthropicClient:
    model: str = "claude-opus-4-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    name: str = "anthropic"
    timeout: float = 60.0
    max_retries: int = 3

    def _get_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    def _client(self):
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("anthropic SDK not installed; pip install anthropic") from e
        key = self._get_key()
        if not key:
            raise LLMError(
                f"missing {self.api_key_env}; set the env var to use AnthropicClient"
            )
        return anthropic.Anthropic(api_key=key, timeout=self.timeout)

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        client = self._client()

        def _do() -> LLMResponse:
            try:
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limit_error(exc):
                    raise  # let _retry_call see it
                raise LLMError(f"Anthropic API error: {exc}") from exc
            text = "".join(block.text for block in resp.content if hasattr(block, "text"))
            return LLMResponse(
                text=text,
                model=self.model,
                finish_reason=resp.stop_reason or "stop",
                usage={"in": resp.usage.input_tokens, "out": resp.usage.output_tokens},
                raw=resp,
            )

        try:
            return _retry_call(_do, max_retries=self.max_retries)
        except LLMError:
            raise
        except Exception as exc:  # rate-limit raised through after retries
            raise LLMError(f"Anthropic API error after retries: {exc}") from exc

    def health_check(self) -> dict[str, Any]:
        """Ping the provider with a 1-token request."""
        if not self._get_key():
            return {
                "ok": False,
                "model": self.model,
                "error": f"missing {self.api_key_env}",
            }
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            return {"ok": False, "model": self.model, "error": f"SDK not installed: {e}"}
        try:
            resp = self.complete(
                system="reply with the single character: ok",
                user="ping",
                max_tokens=1,
                temperature=0.0,
            )
            return {"ok": True, "model": resp.model, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "model": self.model, "error": str(exc)}


@dataclass
class OpenAIClient:
    model: str = "gpt-5"
    api_key_env: str = "OPENAI_API_KEY"
    name: str = "openai"
    seed: int | None = 42
    timeout: float = 60.0
    max_retries: int = 3

    def _get_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMError("openai SDK not installed; pip install openai") from e
        key = self._get_key()
        if not key:
            raise LLMError(
                f"missing {self.api_key_env}; set the env var to use OpenAIClient"
            )
        return OpenAI(api_key=key, timeout=self.timeout)

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        client = self._client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        def _do() -> LLMResponse:
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limit_error(exc):
                    raise
                raise LLMError(f"OpenAI API error: {exc}") from exc
            choice = resp.choices[0]
            return LLMResponse(
                text=choice.message.content or "",
                model=self.model,
                finish_reason=choice.finish_reason or "stop",
                usage={
                    "in": resp.usage.prompt_tokens if resp.usage else 0,
                    "out": resp.usage.completion_tokens if resp.usage else 0,
                },
                raw=resp,
            )

        try:
            return _retry_call(_do, max_retries=self.max_retries)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"OpenAI API error after retries: {exc}") from exc

    def health_check(self) -> dict[str, Any]:
        if not self._get_key():
            return {
                "ok": False,
                "model": self.model,
                "error": f"missing {self.api_key_env}",
            }
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as e:
            return {"ok": False, "model": self.model, "error": f"SDK not installed: {e}"}
        try:
            resp = self.complete(
                system="reply with the single character: ok",
                user="ping",
                max_tokens=1,
                temperature=0.0,
            )
            return {"ok": True, "model": resp.model, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "model": self.model, "error": str(exc)}


@dataclass
class GoogleClient:
    model: str = "gemini-2.5-pro"
    api_key_env: str = "GOOGLE_API_KEY"
    name: str = "google"
    timeout: float = 60.0
    max_retries: int = 3

    def _get_key(self) -> str | None:
        return os.environ.get(self.api_key_env)

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise LLMError(
                "google-generativeai SDK not installed; pip install google-generativeai"
            ) from e

        key = self._get_key()
        if not key:
            raise LLMError(
                f"missing {self.api_key_env}; set the env var to use GoogleClient"
            )

        genai.configure(api_key=key)
        model = genai.GenerativeModel(self.model, system_instruction=system)
        cfg: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        # Some versions of google-generativeai accept request_options for timeout
        request_options = {"timeout": self.timeout}

        def _do() -> LLMResponse:
            try:
                try:
                    resp = model.generate_content(
                        user, generation_config=cfg, request_options=request_options
                    )
                except TypeError:
                    # Older SDK signature without request_options kwarg
                    resp = model.generate_content(user, generation_config=cfg)
            except Exception as exc:  # noqa: BLE001
                if _is_rate_limit_error(exc):
                    raise
                raise LLMError(f"Google API error: {exc}") from exc
            return LLMResponse(text=resp.text or "", model=self.model, raw=resp)

        try:
            return _retry_call(_do, max_retries=self.max_retries)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Google API error after retries: {exc}") from exc

    def health_check(self) -> dict[str, Any]:
        if not self._get_key():
            return {
                "ok": False,
                "model": self.model,
                "error": f"missing {self.api_key_env}",
            }
        try:
            import google.generativeai as genai  # noqa: F401
        except ImportError as e:
            return {"ok": False, "model": self.model, "error": f"SDK not installed: {e}"}
        try:
            resp = self.complete(
                system="reply with the single character: ok",
                user="ping",
                max_tokens=1,
                temperature=0.0,
            )
            return {"ok": True, "model": resp.model, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "model": self.model, "error": str(exc)}


def make_client(vendor: str, model: str, **kwargs) -> LLMClient:
    """Factory used by configs."""
    vendor = vendor.lower()
    if vendor == "anthropic":
        return AnthropicClient(model=model, **kwargs)
    if vendor == "openai":
        return OpenAIClient(model=model, **kwargs)
    if vendor == "google":
        return GoogleClient(model=model, **kwargs)
    if vendor == "mock":
        return MockLLMClient(name=model or "mock", **kwargs)
    raise ValueError(f"unknown vendor {vendor!r}")
