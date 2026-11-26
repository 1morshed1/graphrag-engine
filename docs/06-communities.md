# 06 — Community Detection & Summarization

Source: [`graph/community.py`](../src/graphrag_engine/graph/community.py),
[`pipeline/community.py`](../src/graphrag_engine/pipeline/community.py)

## The problem communities solve: *global* questions

Chunk retrieval (vector or graph) is great for *local* questions — ones answered by a few
specific passages. It is bad at *global* ones:

> "What are the main themes across these documents?"

No single chunk contains "the themes." The answer is an emergent property of the whole
corpus. Microsoft's GraphRAG paper's key insight: detect **communities** (clusters of
densely-connected entities), summarize each with an LLM, and retrieve those summaries for
global questions.

## Step 1 — detection

A *community* is a set of nodes with many edges among themselves and few to the rest —
e.g. "a VC firm and its portfolio companies," or "an AI lab, its founder, and its models."

We use the **Leiden** algorithm (via `igraph`/`leidenalg`) when installed. Leiden improves
on the older Louvain method by guaranteeing communities are internally well-connected. Both
maximize *modularity* — roughly, "more edges inside clusters than you'd expect by chance."

Community detection is defined on **undirected, weighted** graphs, so we first project our
MultiDiGraph down: parallel edges between a pair collapse into a single edge whose `weight`
is the count. Direction is irrelevant to "who clusters with whom."

### Graceful fallback

Leiden's dependencies are optional. When they're absent we fall back to networkx's
`greedy_modularity_communities` — same *idea* (maximize modularity), lower quality, but
pure-Python and always available. **This is a recurring pattern in the engine:** prefer the
best tool, degrade gracefully so the default install always runs.

```python
try:
    return _leiden(undirected, resolution)  # best, needs extras
except ImportError:
    return _greedy_modularity(undirected)  # always works
```

`community_resolution` tunes granularity: higher → more, smaller communities.

## Step 2 — summarization

For each community above `min_community_size`, we build a text description (its member
entities + the relationships *among them*) and ask the LLM for a `{title, summary}`. We then
**embed the summary** and add it to the community vector store, so it can be retrieved by
semantic similarity like any chunk.

We skip tiny communities (size < `min_community_size`): two loosely-connected nodes rarely
form a coherent theme worth summarizing, and summarizing them wastes LLM calls.

## Why this matters

Community summaries are the third leg of retrieval (alongside chunks and graph traversal).
In the evaluation (docs/09), they are the *only* thing that answers global questions — which
is exactly why the honest vector-only baseline, which has no communities, scores 0% on that
category while hybrid scores 100%.

**Lesson:** hierarchical summarization converts a graph's *structure* into retrievable
*global context* — the capability plain RAG fundamentally lacks.

Next: [`07-retrieval.md`](07-retrieval.md).
