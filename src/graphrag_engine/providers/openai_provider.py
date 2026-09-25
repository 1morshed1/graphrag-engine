"""OpenAI-backed providers. Imported lazily so the package works without the SDK installed.

Install with ``pip install 'graphrag-engine[openai]'`` and set ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import os

from .base import EmbeddingProvider, LLMProvider


class OpenAILLMProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        from openai import OpenAI  # lazy: only needed when this provider is used

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model = model

    def complete(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0,  # deterministic extraction/answers
        )
        return resp.choices[0].message.content or ""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model = model
        self.dim = 1536  # text-embedding-3-small

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]
