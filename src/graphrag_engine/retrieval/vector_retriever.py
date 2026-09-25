"""Vector retrieval: the classic semantic search half of GraphRAG.

Embeds the query and pulls the nearest chunks (fine-grained facts) and the nearest community
summaries (global context) from their vector stores. This is exactly what a vanilla RAG
system does — GraphRAG keeps it and *adds* graph traversal alongside. See docs/07.
"""

from __future__ import annotations

from ..config import Config
from ..models import RetrievedItem
from ..providers.base import EmbeddingProvider
from ..storage.vector_store import VectorStore


class VectorRetriever:
    def __init__(self, embedder: EmbeddingProvider, config: Config):
        self.embedder = embedder
        self.config = config

    def retrieve(
        self, query: str, chunk_store: VectorStore, community_store: VectorStore
    ) -> list[RetrievedItem]:
        q = self.embedder.embed_one(query)
        items: list[RetrievedItem] = []

        for cid, text, score in chunk_store.search(q, self.config.vector_top_k):
            items.append(
                RetrievedItem(id=cid, text=text, kind="chunk", score=score, source="vector")
            )

        for cid, text, score in community_store.search(q, self.config.community_top_k):
            items.append(
                RetrievedItem(
                    id=cid, text=text, kind="community", score=score, source="vector"
                )
            )
        return items
