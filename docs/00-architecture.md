# 00 — Architecture & Design Decisions

> This is the first document in the `docs/` learning series. Each doc explains **what**
> we built, **why** we built it that way, and **what alternatives we rejected**. Read them
> in order (00, 01, 02, …) to follow the project as it grows.

## 1. What is GraphRAG, and why does it exist?

Standard **Retrieval-Augmented Generation (RAG)** works like this:

1. Split documents into chunks.
2. Embed each chunk into a vector.
3. At query time, embed the query, find the nearest chunks by cosine similarity, and
   feed them to an LLM.

This is excellent for questions whose answer lives in **one place** ("What is our refund
policy?"). It falls apart for questions whose answer must be **assembled from several
places connected by relationships**:

> "Which companies founded by Stanford alumni also received Sequoia funding?"

No single chunk contains that answer. You have to:

- find people who are Stanford alumni,
- find companies they founded,
- find which of those companies Sequoia funded,
- intersect the results.

That is a **graph traversal** problem, not a similarity problem. Vector search has no
concept of "follow the *founded* edge, then the *invested_in* edge."

**GraphRAG** (popularized by Microsoft Research, 2024) fixes this by building an explicit
**knowledge graph** — entities (nodes) connected by relationships (edges) — extracted from
the documents by an LLM. Retrieval then combines:

- **Graph traversal** — walk edges to answer multi-hop and structural questions.
- **Vector search** — the classic semantic retrieval, still great for direct lookups.
- **Community summaries** — LLM-written summaries of clusters of related entities, which
  answer *global* questions like "What are the main themes in this corpus?"

## 2. The pipeline at a glance

```
Documents
   │
   ▼
[ Chunker ]  ── splits text into overlapping windows
   │
   ▼
[ KG Builder ] ── LLM extracts (entities, relations) per chunk, then deduplicates
   │
   ├──────────────► [ Graph Store ]   nodes + edges (networkx + SQLite)
   │
   ├──────────────► [ Vector Store ]  chunk / entity / community embeddings
   │
   ▼
[ Community Detection ] ── clusters entities, LLM summarizes each cluster
   │
   ▼
[ Retriever ] ── graph traversal + vector search, fused with Reciprocal Rank Fusion
   │
   ▼
[ Answer Synthesizer ] ── LLM writes the final answer from fused context
```

Each stage is a separate module so you can read, test, and swap them independently.

## 3. Design decisions (and the alternatives we rejected)

### Decision 1 — Provider abstraction with a deterministic mock

**What:** All LLM and embedding calls go through the `LLMProvider` / `EmbeddingProvider`
interfaces in `providers/`. We ship a `MockProvider` that is fully deterministic and needs
no API key or network.

**Why:** The project must *run on the reader's machine immediately*. If day one
required an OpenAI key, a Neo4j cluster, and a GPU, most people would never see it work.
The mock lets every test and the full CLI demo run offline. Real providers
(`OpenAIProvider`, `AnthropicProvider`) plug into the same interface when you add a key.

**Rejected:** Hard-coding a single SDK. That couples the whole codebase to one vendor and
makes it untestable without spending money on every run.

### Decision 2 — `networkx` + SQLite instead of Neo4j

**What:** The graph lives in a `networkx.MultiDiGraph` in memory, persisted to SQLite.

**Why:** Neo4j is the "production" answer, but it is a heavy external service. At this
scale, `networkx` gives us BFS, neighbor lookups, and — via
`igraph`/`leidenalg` when available — the Leiden community algorithm, with zero services to
run. SQLite makes the graph durable and inspectable with any SQLite browser. The storage
layer is behind an interface (`GraphStore`), so a `Neo4jGraphStore` can be added later
without touching the pipeline.

**Rejected:** Requiring Neo4j up front (deployment burden); a raw Python dict (loses the
battle-tested graph algorithms networkx provides).

### Decision 3 — NumPy brute-force vector store

**What:** Embeddings are stored in a NumPy matrix; search is exact cosine similarity.

**Why:** For the corpus sizes this engine targets (thousands of chunks), exact search over
a NumPy matrix is fast, simple, and dependency-light. It keeps the *retrieval logic*
front-and-center instead of burying it in an ANN index's configuration.

**Rejected:** FAISS/pgvector as the default. They matter at millions of vectors; here they
would add setup friction that obscures the ideas. The `VectorStore` interface leaves room
to add them.

### Decision 4 — Everything is typed and dataclass-based

Entities, relations, chunks, and retrieval results are all dataclasses in `models.py`.
This makes the data flowing between stages self-documenting and lets your editor catch
mistakes.

## 4. Repository layout

```
src/graphrag_engine/
  models.py            # core dataclasses: Entity, Relation, Chunk, ...
  config.py            # one dataclass holding all tunable knobs
  providers/           # LLM + embedding abstractions (mock, openai, anthropic)
  pipeline/            # chunker, kg_builder, community
  graph/               # community detection algorithms
  storage/             # graph store + vector store
  retrieval/           # graph retrieval, vector retrieval, fusion, synthesis
  engine.py            # GraphRAGEngine — ties it all together
  cli.py               # command-line interface
docs/                  # this learning series
tests/                 # pytest suite (runs fully offline via MockProvider)
examples/              # sample corpus + runnable scripts
```

## 5. How to read this series

| Doc | Topic |
|-----|-------|
| 00 | Architecture & decisions (this file) |
| 01 | Data models & configuration |
| 02 | Providers: LLM & embeddings (and the mock) |
| 03 | Chunking |
| 04 | Knowledge-graph construction & entity dedup |
| 05 | Storage: graph store & vector store |
| 06 | Community detection & summarization |
| 07 | Hybrid retrieval & Reciprocal Rank Fusion |
| 08 | Answer synthesis & the engine facade |
| 09 | Evaluation: proving GraphRAG beats vector-only RAG |
| 10 | CLI & putting it all together |

Next: [`01-data-models.md`](01-data-models.md).
