"""Tests for LLM client adapters.

These tests never make real API calls — they verify:
  * MockLLMClient.health_check returns ok=True
  * Real adapters without env keys return a clean ok=False (no exception)
  * make_client factory dispatches to the correct concrete class
"""

from __future__ import annotations

import pytest

from ccema.judges.llm_client import (
    AnthropicClient,
    GoogleClient,
    LLMError,
    MockLLMClient,
    OpenAIClient,
    make_client,
)


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


def test_mock_health_check_ok() -> None:
    res = MockLLMClient().health_check()
    assert res["ok"] is True
    assert res["error"] is None
    assert "model" in res


@pytest.mark.parametrize(
    "cls,env_var",
    [
        (AnthropicClient, "ANTHROPIC_API_KEY"),
        (OpenAIClient, "OPENAI_API_KEY"),
        (GoogleClient, "GOOGLE_API_KEY"),
    ],
)
def test_real_client_health_check_without_key(monkeypatch, cls, env_var) -> None:
    monkeypatch.delenv(env_var, raising=False)
    client = cls()
    res = client.health_check()
    assert res["ok"] is False
    assert res["error"]
    # Error must be informative — mention the env var name
    assert env_var in res["error"]


@pytest.mark.parametrize(
    "cls,env_var",
    [
        (AnthropicClient, "ANTHROPIC_API_KEY"),
        (OpenAIClient, "OPENAI_API_KEY"),
        (GoogleClient, "GOOGLE_API_KEY"),
    ],
)
def test_real_client_complete_without_key_raises_llm_error(monkeypatch, cls, env_var) -> None:
    monkeypatch.delenv(env_var, raising=False)
    client = cls()
    with pytest.raises(LLMError) as ei:
        client.complete(system="s", user="u", max_tokens=1)
    assert env_var in str(ei.value)


# ---------------------------------------------------------------------------
# make_client factory
# ---------------------------------------------------------------------------


def test_make_client_mock() -> None:
    c = make_client("mock", "mock-v0")
    assert isinstance(c, MockLLMClient)
    assert c.name == "mock-v0"


def test_make_client_anthropic() -> None:
    c = make_client("anthropic", "claude-opus-4-5")
    assert isinstance(c, AnthropicClient)
    assert c.model == "claude-opus-4-5"
    assert c.timeout == 60.0
    assert c.max_retries == 3


def test_make_client_openai() -> None:
    c = make_client("openai", "gpt-5")
    assert isinstance(c, OpenAIClient)
    assert c.model == "gpt-5"
    assert c.timeout == 60.0
    assert c.max_retries == 3


def test_make_client_google() -> None:
    c = make_client("google", "gemini-2.5-pro")
    assert isinstance(c, GoogleClient)
    assert c.model == "gemini-2.5-pro"
    assert c.timeout == 60.0
    assert c.max_retries == 3


def test_make_client_case_insensitive() -> None:
    assert isinstance(make_client("Anthropic", "claude-opus-4-5"), AnthropicClient)
    assert isinstance(make_client("OPENAI", "gpt-5"), OpenAIClient)


def test_make_client_unknown_vendor() -> None:
    with pytest.raises(ValueError):
        make_client("acme", "model-x")


# ---------------------------------------------------------------------------
# Timeout / retry config plumbing
# ---------------------------------------------------------------------------


def test_real_clients_accept_timeout_and_retries() -> None:
    a = AnthropicClient(timeout=10.0, max_retries=1)
    o = OpenAIClient(timeout=10.0, max_retries=1)
    g = GoogleClient(timeout=10.0, max_retries=1)
    assert a.timeout == 10.0 and a.max_retries == 1
    assert o.timeout == 10.0 and o.max_retries == 1
    assert g.timeout == 10.0 and g.max_retries == 1
