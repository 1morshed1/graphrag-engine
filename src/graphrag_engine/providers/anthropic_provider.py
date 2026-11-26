"""Anthropic (Claude) LLM provider. Lazy-imported.

Install with ``pip install 'graphrag-engine[anthropic]'`` and set ``ANTHROPIC_API_KEY``.
Claude has no first-party embedding model, so pair this with the local sentence-transformers
embedder or the OpenAI embedder.
"""

from __future__ import annotations

import os

from .base import LLMProvider


class AnthropicLLMProvider(LLMProvider):
    def __init__(
        self, model: str = "claude-sonnet-5", api_key: str | None = None, max_tokens: int = 16000
    ):
        # max_tokens is generous because current Claude models think before
        # answering, and that thinking counts against the same budget.
        from anthropic import Anthropic  # lazy

        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, prompt: str, system: str | None = None) -> str:
        # No `temperature`: current Claude models reject sampling parameters
        # with HTTP 400. Determinism for extraction comes from the prompt.
        kwargs = {"system": system} if system else {}
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        # Concatenate any text blocks in the response.
        return "".join(block.text for block in resp.content if block.type == "text")
