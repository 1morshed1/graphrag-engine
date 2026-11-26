"""AnthropicLLMProvider request shape, with the SDK faked out (no network)."""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def calls(monkeypatch):
    sent: list[dict] = []

    class _Messages:
        def create(self, **kwargs):
            sent.append(kwargs)
            blocks = [
                types.SimpleNamespace(type="thinking", thinking=""),
                types.SimpleNamespace(type="text", text="answer"),
            ]
            return types.SimpleNamespace(content=blocks)

    module = types.ModuleType("anthropic")
    module.Anthropic = lambda api_key=None: types.SimpleNamespace(messages=_Messages())
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return sent


def test_omits_sampling_params_and_keeps_text_only(calls):
    from graphrag_engine.providers.anthropic_provider import AnthropicLLMProvider

    provider = AnthropicLLMProvider(api_key="k")
    assert provider.complete("q", system="be terse") == "answer"
    sent = calls[0]
    for param in ("temperature", "top_p", "top_k"):
        assert param not in sent  # current Claude models reject these with 400
    assert sent["system"] == "be terse"
    assert sent["max_tokens"] >= 4096  # room for thinking + answer


def test_no_system_field_when_none(calls):
    from graphrag_engine.providers.anthropic_provider import AnthropicLLMProvider

    AnthropicLLMProvider(api_key="k").complete("q")
    assert "system" not in calls[0]
