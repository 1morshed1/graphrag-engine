"""Local embedding model via sentence-transformers. Lazy-imported.

Install with ``pip install 'graphrag-engine[embeddings]'``. Runs on CPU, no API key,
much higher quality than the mock embedder. Good default for real offline use.
"""

from __future__ import annotations

from .base import EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]
