# 08 — Answer Synthesis & the Engine Facade

Source: [`retrieval/synthesizer.py`](../src/graphrag_engine/retrieval/synthesizer.py),
[`engine.py`](../src/graphrag_engine/engine.py)

## Synthesis: grounding the answer

The fused context (docs/07) is a ranked list of passages and community summaries. The
synthesizer:

1. Formats them into a numbered block, tagging each as `PASSAGE` or `SUMMARY`.
2. Prompts the LLM to answer using **only** that context, and to say so when it's
   insufficient.
3. Returns an `Answer` that carries both the text **and the context it used**.

Two deliberate choices:

- **"Use only the context."** This is the standard RAG guardrail against hallucination: the
  model answers from retrieved evidence rather than its parametric memory, which keeps
  answers grounded and current with your corpus.
- **Attach the context to the answer.** GraphRAG's whole pitch is *traceable, multi-hop*
  reasoning. Returning `Answer.context` lets a caller show its work — which passages and
  which community summaries produced the answer. Observability is a feature, not an
  afterthought.

## The engine facade

`GraphRAGEngine` is the one class users touch. It wires every stage and exposes a tiny
surface:

```python
engine = GraphRAGEngine.default()  # offline mock providers
engine.ingest_directory("examples/corpus")  # build chunks, graph, communities, indexes
answer = engine.query("...", mode="hybrid")
engine.save("data/g.graphrag")
engine.load("data/g.graphrag")
```

### Why a facade?

The pipeline has ~10 collaborating objects (chunker, KG builder, two vector stores, graph
store, community builder, two retrievers, fuser, synthesizer). A facade hides that wiring
behind `ingest`/`query`/`save`/`load` so callers — and the CLI, and the eval harness — don't
re-assemble it each time. It's the single composition root where dependency choices (which
providers, which config) are made once.

### `ingest_documents` — the build order matters

```
1. chunk         →  chunks + chunk_texts
2. embed chunks  →  chunk vector store
3. build graph   →  entities + relations (LLM extraction + dedup)
4. communities   →  detect + summarize + community vector store
```

Each step depends on the previous: communities need the graph, the graph needs chunks. This
ordering is the whole pipeline in one method.

### `retrieve` vs `query`

`retrieve()` returns the fused context *without* calling the LLM to synthesize. The eval
harness uses it to measure the *retriever* in isolation (did the needed facts get
retrieved?), separate from the LLM's answering ability. `query()` = `retrieve()` +
synthesize.

### Persistence

`save()` pickles config + chunks + graph + communities + both vector stores into one file so
you don't re-run (potentially expensive) LLM extraction every session. `load()` restores it
and rebuilds the retrieval components against the loaded config.

Next: [`09-evaluation.md`](09-evaluation.md).
