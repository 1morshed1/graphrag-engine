# 01 — Data Models & Configuration

Source: [`models.py`](../src/graphrag_engine/models.py), [`config.py`](../src/graphrag_engine/config.py)

Before writing any logic, we define the *shapes* of the data that move between stages.
Doing this first is a design technique: once the data structures are right, most functions
become obvious ("take these, return those").

## The five core types

| Type | Role | Key fields |
|------|------|-----------|
| `Chunk` | A window of source text | `id`, `text`, `doc_id`, `index`, `embedding` |
| `Entity` | A **node** in the graph | `name`, `type`, `description`, `source_chunks`, `community` |
| `Relation` | A directed **edge** | `source`, `relation`, `target`, `source_chunk` |
| `Community` | A cluster of entities + its summary | `entity_keys`, `title`, `summary`, `embedding` |
| `RetrievedItem` | One context unit before fusion | `id`, `text`, `kind`, `score`, `source` |

### Why `Entity.source_chunks` matters most

It is the bridge from the **graph world** back to the **text world**. Graph retrieval walks
nodes and edges, but an LLM needs *text* to answer from. `source_chunks` records which
chunks mentioned an entity, so once traversal lands on a node we can pull the real passages
behind it. Lose this link and the graph becomes a pretty diagram that can't ground answers.

### Why `EntityType` is a closed enum

If the extractor were allowed free-form type strings, the graph would fill with
near-duplicates: `Company`, `company`, `Organisation`, `org`. A closed vocabulary plus a
`coerce()` method (which maps aliases and defaults unknowns to `Concept`) keeps the type
space clean. **Lesson:** constrain what the LLM can emit whenever the downstream code cares
about consistency.

### `normalize_name()` — the cheap half of deduplication

```python
"Sequoia Capital." -> "sequoia capital"
```

Entity dedup has two halves:
1. **Deterministic** — lowercase, trim punctuation, collapse whitespace (this function).
2. **Semantic** — embedding similarity, to catch "Sequoia" vs "Sequoia Capital" (docs/04).

We do the cheap deterministic pass first because it's free and catches the majority of
surface-form duplicates.

### `Entity.merge()`

When we decide two entities are the same real-world thing, we *union* their source chunks
(never lose a mention) and keep the longer description (usually more informative). Merging
rather than overwriting is what lets one entity accumulate knowledge from across many
documents — the whole point of building a graph over a corpus.

## Configuration: one dataclass, every knob

`Config` holds every tunable value — chunk size, dedup threshold, retrieval `top_k`s, the
RRF constant, and so on — with an inline comment explaining each. Two reasons:

1. **Reproducibility.** An evaluation run can print its `Config` next to its scores, so a
   number always has the settings that produced it attached.
2. **Discoverability.** A newcomer can read one file and see the entire surface area of
   things they're allowed to tune.

Next: [`02-providers.md`](02-providers.md).
