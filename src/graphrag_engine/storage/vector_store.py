"""A brute-force NumPy vector store.

Stores embeddings in a single matrix and answers nearest-neighbor queries with exact cosine
similarity. For the corpus sizes this engine targets (thousands of items), exact search is
fast and keeps the *retrieval logic* readable instead of hidden inside an ANN index's
tuning. The interface leaves room to drop in FAISS/pgvector later without touching callers.

We keep three logical namespaces in separate stores (chunks, entities, communities) so a
query can search each independently — see the retriever in docs/07.
"""

from __future__ import annotations

import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[str] = []
        self._payloads: list[str] = []  # the text behind each vector
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, id: str, embedding: list[float], payload: str) -> None:
        self._ids.append(id)
        self._payloads.append(payload)
        row = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        self._matrix = np.vstack([self._matrix, row]) if len(self._ids) > 1 else row

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[str, str, float]]:
        """Return up to ``top_k`` (id, payload, similarity) triples, best first."""
        if len(self._ids) == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        # Normalize both sides so the dot product is cosine similarity.
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm
        mat_norms = np.linalg.norm(self._matrix, axis=1)
        mat_norms[mat_norms == 0] = 1.0
        normalized = self._matrix / mat_norms[:, None]
        sims = normalized @ q
        k = min(top_k, len(self._ids))
        # argpartition for the top-k, then sort just those k.
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [(self._ids[i], self._payloads[i], float(sims[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self._ids)

    # --- persistence helpers (used by GraphRAGEngine.save/load) -----------------------

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "ids": self._ids,
            "payloads": self._payloads,
            "matrix": self._matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> VectorStore:
        store = cls(dim=data["dim"])
        store._ids = list(data["ids"])
        store._payloads = list(data["payloads"])
        store._matrix = np.asarray(data["matrix"], dtype=np.float32).reshape(-1, data["dim"])
        return store
