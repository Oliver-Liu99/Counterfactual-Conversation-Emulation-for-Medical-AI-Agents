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
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    text: str
    model: str
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


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


# ---------------------------------------------------------------------------
# Real providers — thin adapters, lazy SDK import
# ---------------------------------------------------------------------------


@dataclass
class AnthropicClient:
    model: str = "claude-opus-4-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    name: str = "anthropic"

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
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic SDK not installed; pip install anthropic") from e

        client = anthropic.Anthropic(api_key=os.environ.get(self.api_key_env))
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return LLMResponse(
            text=text,
            model=self.model,
            finish_reason=resp.stop_reason or "stop",
            usage={"in": resp.usage.input_tokens, "out": resp.usage.output_tokens},
            raw=resp,
        )


@dataclass
class OpenAIClient:
    model: str = "gpt-5"
    api_key_env: str = "OPENAI_API_KEY"
    name: str = "openai"
    seed: int | None = 42

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
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai SDK not installed; pip install openai") from e

        client = OpenAI(api_key=os.environ.get(self.api_key_env))
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

        resp = client.chat.completions.create(**kwargs)
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


@dataclass
class GoogleClient:
    model: str = "gemini-2.5-pro"
    api_key_env: str = "GOOGLE_API_KEY"
    name: str = "google"

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
            raise RuntimeError(
                "google-generativeai SDK not installed; pip install google-generativeai"
            ) from e

        genai.configure(api_key=os.environ.get(self.api_key_env))
        model = genai.GenerativeModel(self.model, system_instruction=system)
        cfg: dict[str, Any] = {"max_output_tokens": max_tokens, "temperature": temperature}
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        resp = model.generate_content(user, generation_config=cfg)
        return LLMResponse(text=resp.text or "", model=self.model, raw=resp)


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
