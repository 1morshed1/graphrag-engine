"""GraphRAGEngine — the public facade that ties every stage together.

Ingest builds all the indexes once; query then retrieves and synthesizes. The engine
supports three retrieval modes so you can compare them directly (this is what the evaluation
harness uses to prove GraphRAG beats vector-only on multi-hop questions):

    "hybrid"  graph traversal + vector search, fused with RRF   (the full system)
    "vector"  vector search only                                (the RAG baseline)
    "graph"   graph traversal only                              (ablation)

See docs/08 and docs/10.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

from .config import Config
from .models import Answer, Chunk
from .pipeline.chunker import Chunker
from .pipeline.community import CommunityBuilder
from .pipeline.kg_builder import KnowledgeGraphBuilder
from .providers import make_providers
from .providers.base import EmbeddingProvider, LLMProvider
from .retrieval.fusion import reciprocal_rank_fusion
from .retrieval.graph_retriever import GraphRetriever
from .retrieval.synthesizer import AnswerSynthesizer
from .retrieval.vector_retriever import VectorRetriever
from .storage.graph_store import GraphStore
from .storage.vector_store import VectorStore

_TEXT_EXTENSIONS = {".txt", ".md"}


class GraphRAGEngine:
    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        config: Config | None = None,
    ):
        self.llm = llm
        self.embedder = embedder
        self.config = config or Config()

        # Populated by ingest().
        self.chunks: list[Chunk] = []
        self.chunk_texts: dict[str, str] = {}
        self.graph = GraphStore()
        self.communities: list = []
        self.chunk_store = VectorStore(embedder.dim)
        self.community_store = VectorStore(embedder.dim)

        # Retrieval components.
        self._graph_retriever = GraphRetriever(llm, self.config)
        self._vector_retriever = VectorRetriever(embedder, self.config)
        self._synthesizer = AnswerSynthesizer(llm, self.config)

    # --- construction shortcuts -------------------------------------------------------

    @classmethod
    def default(cls, config: Config | None = None) -> GraphRAGEngine:
        """Fully offline engine using the deterministic mock providers."""
        llm, embedder = make_providers("mock")
        return cls(llm, embedder, config)

    @classmethod
    def from_preset(cls, preset: str, config: Config | None = None) -> GraphRAGEngine:
        llm, embedder = make_providers(preset)
        return cls(llm, embedder, config)

    # --- ingestion --------------------------------------------------------------------

    def ingest_documents(self, documents: dict[str, str]) -> None:
        """Ingest a mapping of ``doc_id -> text``. Builds chunks, graph, and all indexes."""
        chunker = Chunker(self.config.chunk_size, self.config.chunk_overlap)

        # 1) Chunk every document.
        for doc_id, text in documents.items():
            self.chunks.extend(chunker.chunk_document(text, doc_id))
        self.chunk_texts = {c.id: c.text for c in self.chunks}

        # 2) Embed and index the chunks (vector-search half).
        embeddings = self.embedder.embed([c.text for c in self.chunks])
        for chunk, emb in zip(self.chunks, embeddings, strict=True):
            chunk.embedding = emb
            self.chunk_store.add(chunk.id, emb, chunk.text)

        # 3) Build the knowledge graph (LLM extraction + dedup).
        kg = KnowledgeGraphBuilder(self.llm, self.embedder, self.config)
        graph_data = kg.build(self.chunks)
        for entity in graph_data.entities.values():
            self.graph.add_entity(entity)
        for relation in graph_data.relations:
            self.graph.add_relation(relation)

        # 4) Detect + summarize communities (global-context half).
        community_builder = CommunityBuilder(self.llm, self.embedder, self.config)
        self.communities = community_builder.build(self.graph)
        for community in self.communities:
            if community.embedding is not None:
                self.community_store.add(
                    f"community::{community.id}",
                    community.embedding,
                    f"{community.title}: {community.summary}",
                )

    def ingest_directory(self, path: str) -> None:
        """Ingest every .txt/.md file under ``path`` (recursively). doc_id = relative path."""
        base = Path(path)
        documents: dict[str, str] = {}
        for file in sorted(base.rglob("*")):
            if file.suffix.lower() in _TEXT_EXTENSIONS and file.is_file():
                documents[str(file.relative_to(base))] = file.read_text(encoding="utf-8")
        if not documents:
            raise ValueError(f"No .txt/.md documents found under {path!r}")
        self.ingest_documents(documents)

    # --- querying ---------------------------------------------------------------------

    def query(self, question: str, mode: str = "hybrid") -> Answer:
        """Answer ``question`` using the chosen retrieval ``mode``."""
        context = self.retrieve(question, mode)
        return self._synthesizer.synthesize(question, context)

    def retrieve(self, question: str, mode: str = "hybrid"):
        """Return the fused context items (without synthesizing) — handy for evaluation."""
        graph_items = []
        vector_items = []
        if mode in ("hybrid", "graph"):
            graph_items = self._graph_retriever.retrieve(question, self.graph, self.chunk_texts)
        if mode in ("hybrid", "vector"):
            vector_items = self._vector_retriever.retrieve(
                question,
                self.chunk_store,
                self.community_store,
                include_communities=(mode == "hybrid"),
            )

        if mode == "graph":
            lists = [graph_items]
        elif mode == "vector":
            lists = [vector_items]
        else:
            lists = [graph_items, vector_items]

        return reciprocal_rank_fusion(lists, self.config.rrf_k, self.config.final_top_k)

    # --- stats & persistence ----------------------------------------------------------

    def stats(self) -> dict:
        return {
            "documents": len({c.doc_id for c in self.chunks}),
            "chunks": len(self.chunks),
            "entities": self.graph.num_nodes(),
            "relations": self.graph.num_edges(),
            "communities": len(self.communities),
        }

    def save(self, path: str) -> None:
        """Persist all built state to a single file so you don't re-run extraction.

        We pickle the graph, chunks, communities, vector stores, and config. Pickle keeps
        this simple for a learning project; a production system would use a real database
        (the storage interfaces leave room for that).
        """
        payload = {
            "config": self.config,
            "chunks": self.chunks,
            "chunk_texts": self.chunk_texts,
            "graph": self.graph,
            "communities": self.communities,
            "chunk_store": self.chunk_store.to_dict(),
            "community_store": self.community_store.to_dict(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.config = payload["config"]
        self.chunks = payload["chunks"]
        self.chunk_texts = payload["chunk_texts"]
        self.graph = payload["graph"]
        self.communities = payload["communities"]
        self.chunk_store = VectorStore.from_dict(payload["chunk_store"])
        self.community_store = VectorStore.from_dict(payload["community_store"])
        # Rebuild retrieval components against the loaded config.
        self._graph_retriever = GraphRetriever(self.llm, self.config)
        self._vector_retriever = VectorRetriever(self.embedder, self.config)
        self._synthesizer = AnswerSynthesizer(self.llm, self.config)
