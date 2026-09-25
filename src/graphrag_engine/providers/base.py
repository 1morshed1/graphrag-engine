"""Provider interfaces.

Every call to an LLM or an embedding model in this codebase goes through one of these two
abstract classes. This is the seam that lets the whole engine run offline (via
``MockProvider``) or against OpenAI/Anthropic/local models by swapping one object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """A text-in, text-out large language model."""

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> str:
        """Return the model's completion for ``prompt``."""
        raise NotImplementedError


class EmbeddingProvider(ABC):
    """Turns text into fixed-length vectors."""

    #: Dimensionality of the vectors this provider returns.
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Batching is part of the interface because real APIs
        are far cheaper per-item when called in batches."""
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        """Convenience for embedding a single string."""
        return self.embed([text])[0]
