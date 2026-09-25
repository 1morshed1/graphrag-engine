# 07 — Hybrid Retrieval & Reciprocal Rank Fusion

Source: [`retrieval/`](../src/graphrag_engine/retrieval/)

This is where GraphRAG earns its name: it retrieves with **two complementary strategies** and
fuses them.

## Vector retrieval (the RAG baseline)

`vector_retriever.py`: embed the query, pull the nearest chunks (fine-grained facts) and —
in hybrid mode — the nearest community summaries (global context). This is exactly what
vanilla RAG does. It excels at direct, single-hop lookups where the answer is semantically
close to the question.

Its blind spot: **multi-hop**. "Which company that Sequoia funded builds a file-syncing
platform?" requires chaining `Sequoia --funded--> X` with `X --builds--> file-syncing`. A
chunk about Sequoia and a chunk about file-syncing may be far apart in embedding space, so
top-k vector search can miss one hop.

## Graph retrieval (the new capability)

`graph_retriever.py`:

1. Ask the LLM which entities the query names ("Sequoia", "file-syncing").
2. Resolve each to a graph node (fuzzy match via `find_node`).
3. BFS out to `graph_max_hops`, collecting each visited node's source chunks, tracking the
   **minimum hop distance** at which each chunk was reached.
4. Score chunks `1 / (1 + hop_distance)` — nodes nearer the query's entities rank higher.

Because it *follows edges*, it naturally surfaces the passages that **connect** the query's
entities — precisely the multi-hop facts vector search drops.

## Fusion: why we can't just add scores

The two retrievers return scores on **incomparable scales** — a cosine similarity (≈0–1
distribution) vs. a `1/(1+hop)` value (1, 0.5, 0.33…). Adding or concatenating them lets one
scale dominate arbitrarily.

**Reciprocal Rank Fusion (RRF)** sidesteps this by ignoring the raw scores and using only
**rank position**:

```
score(item) = Σ over lists   1 / (k + rank_in_list)      # rank is 0-based
```

- An item ranked #1 in either retriever gets a big `1/(k+0)` contribution.
- An item that appears in *both* lists accumulates from both → floats to the top.
- `k` (default 60) dampens how much low ranks matter; larger `k` = flatter weighting.

RRF is the standard hybrid-search fuser because it's simple, needs zero score calibration,
and is robust to wildly different score distributions. Our unit test asserts the intuition:
an item present in both lists beats items present in only one.

## The three modes

The engine exposes retrieval as three modes so you can compare them directly:

| Mode | Retrievers | Communities? | Use |
|------|-----------|--------------|-----|
| `vector` | vector only | **no** | the honest vanilla-RAG baseline |
| `graph`  | graph only | no | ablation — traversal in isolation |
| `hybrid` | graph + vector, RRF-fused | **yes** | the full system |

`vector` deliberately excludes community summaries because those only exist thanks to the
graph — including them would flatter the baseline unfairly (see docs/09).

## Putting it together

```
query
 ├─ graph_retriever  → [chunks connecting the query's entities]  (ranked by hops)
 └─ vector_retriever → [semantically similar chunks + community summaries]
                         │
                    RRF fusion (rank-based)
                         │
                 top-K fused context → synthesizer (docs/08)
```

Next: [`08-synthesis.md`](08-synthesis.md).
