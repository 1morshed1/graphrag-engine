# 10 — CLI & Wrap-Up

Source: [`cli.py`](../src/graphrag_engine/cli.py)

## The CLI

Three commands mirror the engine lifecycle. The CLI is intentionally thin — it parses
arguments and calls the engine, nothing more.

```bash
# Build a knowledge graph from a folder of .txt/.md files and persist it.
graphrag build examples/corpus --out data/demo.graphrag

# Ask a question against a built graph (try each mode).
graphrag ask data/demo.graphrag "Where is Stripe based?" --mode hybrid --show-context
graphrag ask data/demo.graphrag "..." --mode vector    # the RAG baseline
graphrag ask data/demo.graphrag "..." --mode graph     # traversal only

# Inspect what was built.
graphrag stats data/demo.graphrag
```

`--preset` selects the providers (`mock` default, or `local` / `openai` / `anthropic`).
`--show-context` prints the retrieved passages so you can see *why* an answer was given.

### Design note: stdout vs stderr

The answer goes to **stdout**; progress and context go to **stderr**. This makes the CLI
composable — `graphrag ask ... > answer.txt` captures only the answer, while you still see
progress in the terminal. A small habit that makes command-line tools pleasant to script.

## Using real providers

```bash
pip install '.[openai]'      # or .[anthropic] , .[embeddings] , .[leiden]
export OPENAI_API_KEY=...
graphrag --preset openai build examples/corpus --out data/real.graphrag
```

Nothing in the pipeline changes — only the two provider objects behind the interface. That
is the payoff of the abstraction from docs/02.

## What we built — the whole picture

```
Documents
  → Chunker (docs/03)
  → KG Builder: LLM extraction + 2-layer dedup (docs/04)
  → Graph Store + Vector Store (docs/05)
  → Community detection + summarization (docs/06)
  → Hybrid retrieval: graph + vector, RRF-fused (docs/07)
  → Answer synthesis, context attached (docs/08)
  → Evaluation: hybrid 100% vs vector 62% (docs/09)
```

## Ideas to take it further

- **Real graph DB:** implement a `Neo4jGraphStore` behind the existing interface; run Cypher
  instead of in-memory BFS.
- **ANN vector index:** swap the NumPy store for FAISS/pgvector for large corpora.
- **Hierarchical communities:** summarize communities *of* communities for multi-level
  global context (the full Microsoft GraphRAG design).
- **LLM-judge eval:** add an answer-quality metric alongside context recall for end-to-end
  scoring with real providers.
- **Incremental ingest:** update the graph as new documents arrive instead of rebuilding.
- **Query routing:** classify a query as local vs global and skip the retriever that won't
  help, saving latency.

## The takeaway

GraphRAG isn't magic — it's vanilla RAG plus an explicit, deduplicated entity graph and
community summaries, retrieved together and fused by rank. Each piece is individually simple;
the *combination* is what unlocks multi-hop and global questions that vector search alone
cannot answer. The evaluation proves it on a corpus you can read in two minutes.

← Back to [`00-architecture.md`](00-architecture.md) · [README](../README.md)
