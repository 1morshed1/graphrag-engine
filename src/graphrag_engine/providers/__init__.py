"""Provider registry + factory.

``make_providers("mock")`` returns a ready (llm, embedder) pair. Named factories keep the
lazy imports contained so importing the package never drags in optional heavy SDKs.
"""

from __future__ import annotations

from .base import EmbeddingProvider, LLMProvider
from .mock import MockEmbeddingProvider, MockLLMProvider

__all__ = [
    "LLMProvider",
    "EmbeddingProvider",
    "MockLLMProvider",
    "MockEmbeddingProvider",
    "make_providers",
]


def make_providers(name: str = "mock") -> tuple[LLMProvider, EmbeddingProvider]:
    """Build an (llm, embedder) pair by preset name.

    Presets:
      - "mock"          : fully offline, deterministic (default).
      - "openai"        : OpenAI LLM + OpenAI embeddings (needs OPENAI_API_KEY).
      - "anthropic"     : Claude LLM + local sentence-transformers embeddings.
      - "local"         : mock LLM + local sentence-transformers embeddings.
    """
    if name == "mock":
        return MockLLMProvider(), MockEmbeddingProvider()

    if name == "openai":
        from .openai_provider import OpenAIEmbeddingProvider, OpenAILLMProvider

        return OpenAILLMProvider(), OpenAIEmbeddingProvider()

    if name == "anthropic":
        from .anthropic_provider import AnthropicLLMProvider
        from .sentence_transformer_provider import SentenceTransformerEmbeddingProvider

        return AnthropicLLMProvider(), SentenceTransformerEmbeddingProvider()

    if name == "local":
        from .sentence_transformer_provider import SentenceTransformerEmbeddingProvider

        return MockLLMProvider(), SentenceTransformerEmbeddingProvider()

    raise ValueError(f"Unknown provider preset: {name!r}")
