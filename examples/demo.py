"""End-to-end demo: build the graph and ask questions in each retrieval mode.

    python examples/demo.py

Offline, no keys. Shows the retrieved context so you can see *why* hybrid retrieval finds
the multi-hop answer that pure vector search misses.
"""

from __future__ import annotations

from graphrag_engine import Config, GraphRAGEngine

QUESTION = "Which chip company, founded by a Stanford alum, makes the GPUs OpenAI uses?"


def main() -> None:
    engine = GraphRAGEngine.default(Config(chunk_size=180, chunk_overlap=40, vector_top_k=4))
    engine.ingest_directory("examples/corpus")
    print("Built:", engine.stats(), "\n")
    print("Q:", QUESTION, "\n")

    for mode in ("vector", "graph", "hybrid"):
        context = engine.retrieve(QUESTION, mode=mode)
        print(f"[{mode}] retrieved {len(context)} items:")
        for item in context[:5]:
            print(f"   ({item.source}/{item.kind}) {item.text[:90].strip()}")
        print()

    answer = engine.query(QUESTION, mode="hybrid")
    print("Hybrid answer:\n", answer.text)


if __name__ == "__main__":
    main()
