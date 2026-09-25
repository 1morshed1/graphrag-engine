# GraphRAG Engine

Knowledge-graph-enhanced retrieval — the next step past vanilla vector RAG. Builds an
explicit knowledge graph from your documents (entities + relationships extracted by an
LLM), then answers questions by combining **graph traversal**, **vector search**, and
**community summaries**.

It **runs offline out of the box**: the default providers are deterministic and need no API
key, no database, and no GPU. Swap in OpenAI, Anthropic, or a local embedding model with a
one-line change when you want real quality.

> This repo is also a **learning project**. The [`docs/`](docs/) folder is a numbered
> series that explains every design decision as it was made. Start with
> [`docs/00-architecture.md`](docs/00-architecture.md).

## Why GraphRAG?

Vector RAG retrieves chunks by semantic similarity. It cannot answer questions whose answer
must be *assembled across documents by relationships*:

> "Which companies founded by Stanford alumni also received Sequoia funding?"

That needs graph traversal (`Stanford --alumni--> people --founded--> companies`, then
intersect with `Sequoia --invested_in--> companies`). GraphRAG builds that graph and walks
it. See [`docs/00-architecture.md`](docs/00-architecture.md) for the full rationale.

## Install

```bash
uv venv && uv pip install -e ".[dev]"
# optional extras: .[embeddings]  .[openai]  .[anthropic]  .[leiden]
```

## Quickstart (offline)

```bash
# Build a graph from the sample corpus and ask a multi-hop question:
graphrag build examples/corpus --out data/demo.graphrag
graphrag ask data/demo.graphrag "Which founders studied at Stanford and got Sequoia funding?"
```

Or from Python:

```python
from graphrag_engine import GraphRAGEngine

engine = GraphRAGEngine.default()          # offline mock providers
engine.ingest_directory("examples/corpus")
answer = engine.query("What themes connect these companies?")
print(answer.text)
```

## How it works

```
Documents → Chunker → KG Builder (LLM entity/relation extraction + dedup)
          → Graph Store + Vector Store
          → Community Detection + Summarization
          → Hybrid Retrieval (graph + vector, fused with RRF)
          → Answer Synthesis
```

Each stage is a separate, documented module. Read the [`docs/`](docs/) series in order.

## Evaluation

The engine ships with a small evaluation harness that compares GraphRAG against a
vector-only baseline on single-hop, multi-hop, and global questions — reproducing the
central claim that GraphRAG wins on multi-hop and global queries. See
[`docs/09-evaluation.md`](docs/09-evaluation.md).

## Project layout

```
src/graphrag_engine/   # the engine (models, providers, pipeline, storage, retrieval)
docs/                  # numbered learning series — the "why" behind every decision
tests/                 # pytest suite, runs fully offline
examples/              # sample corpus + runnable scripts
```

## License

MIT
