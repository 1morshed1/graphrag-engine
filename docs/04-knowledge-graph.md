# 04 — Knowledge-Graph Construction & Entity Dedup

Source: [`pipeline/kg_builder.py`](../src/graphrag_engine/pipeline/kg_builder.py)

This is the stage that makes GraphRAG *graph* RAG. We turn free text into a graph of
entities (nodes) and relationships (edges).

## The flow

For each chunk:
1. Ask the LLM to extract `entities` and `relationships` as JSON (prompt in `prompts.py`).
2. **Absorb** each entity into a global entity table, deduplicating on the way in.
3. **Absorb** each relation, resolving its endpoints to canonical (deduped) nodes.

After all chunks, we have `GraphData(entities, relations)` — a global graph assembled from
thousands of local, per-chunk extractions.

## The central problem: deduplication

The same real-world thing appears with many surface forms across a corpus:

```
"Sequoia"   "Sequoia Capital"   "Sequoia Capital Inc."   "sequoia  capital"
```

If these become four separate nodes, the graph breaks: an edge `Stripe --funded_by-->
"Sequoia"` and another `Airbnb --funded_by--> "Sequoia Capital"` now point at *different*
nodes, so traversal can never discover that the same investor backed both. **Dedup is not a
nicety — it is what makes the graph connected enough to traverse.**

We dedup in two layers, cheapest first:

### Layer 1 — deterministic (normalized name)

`normalize_name()` lowercases, trims punctuation, and collapses whitespace. This instantly
collapses `"Sequoia Capital."` and `"sequoia  capital"`. It's free, so we always try it
first.

### Layer 2 — semantic (embedding similarity)

`"Sequoia"` and `"Sequoia Capital"` don't normalize equal, but they *mean* the same thing.
So we embed `name + description` and compare cosine similarity against existing entities. If
it clears `Config.dedup_threshold` (default 0.90) **and the types match**, we merge.

Two safeguards worth noting:
- **Same-type only.** A `Person` named "Washington" and a `Location` named "Washington" are
  different nodes; we never merge across types.
- **`merge()` keeps everything.** Merging unions `source_chunks` (never lose a mention) and
  keeps the longer description. See docs/01.

### The alias map — the subtle bit

After a semantic merge, the entity is stored under one canonical key (say `sequoia
capital`), but a *relation* in some chunk still says `source: "Sequoia"`. If we only looked
up `normalize_name("Sequoia") = "sequoia"`, we'd find no node and drop the edge.

So the builder keeps an **alias map**: every raw name we've seen → the canonical key it
resolved to. Relations resolve their endpoints through this map, so `"Sequoia"` correctly
lands on the `sequoia capital` node. Without this, semantic dedup would silently delete
edges — a nasty, hard-to-spot bug.

```python
# relation endpoint resolution
source = alias.get(normalize_name(raw["source"]))   # "sequoia" -> "sequoia capital"
target = alias.get(normalize_name(raw["target"]))
if source in entities and target in entities:        # drop dangling edges
    add_edge(source, relation, target)
```

## Why drop dangling edges?

The extractor sometimes names an entity in a relation that it never emitted as a first-class
entity. Such an edge has an endpoint with no node — it can't be traversed usefully — so we
drop it to keep the graph sound. (A more elaborate system would back-fill the missing node;
we favor a clean graph.)

## Scaling note

Semantic dedup here is **brute force**: each new entity is compared against all existing
ones. That's fine for thousands of entities and keeps the logic transparent. At millions,
you'd index entity embeddings in an ANN structure — an optimization, not a change in the
idea.

Next: [`05-storage.md`](05-storage.md).
