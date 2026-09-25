# 05 — Storage: Graph Store & Vector Store

Source: [`storage/graph_store.py`](../src/graphrag_engine/storage/graph_store.py),
[`storage/vector_store.py`](../src/graphrag_engine/storage/vector_store.py)

GraphRAG needs two kinds of storage side by side. This is the "dual index" at the heart of
the system.

## Graph store — `networkx.MultiDiGraph`

### Why a MultiDiGraph?

- **Di** (directed): relationships have direction. `founded` runs founder → company, not
  the reverse.
- **Multi** (parallel edges): the same pair can have several distinct relationships — a
  person both `founded` and `leads` a company. A plain graph would clobber one with the
  other.

Each node's id is the entity's canonical key; the full `Entity` object rides along as a node
attribute, and each edge stores its `relation` type and the `source_chunk` it came from.

### Traversal: directed storage, undirected walking

We *store* direction (for display and reasoning) but *traverse* undirected. Why? If a query
mentions Stripe, the fact "Patrick **founded** Stripe" is relevant regardless of which way
the edge points. Retrieval cares about *connectivity*, so `neighbors()` returns both
out-edges and in-edges.

### `bfs_chunks()` — the graph-retrieval workhorse

Breadth-first search from a set of start nodes, out to `max_hops`, collecting each visited
node's `source_chunks`. This is the operation that turns "entities in the query" into "text
passages about them and their neighborhood." Hop distance later becomes a relevance score
(closer = better). See docs/07.

## Vector store — brute-force NumPy

### Why not FAISS/pgvector by default?

Exact cosine search over a NumPy matrix is simple, dependency-free, and plenty fast for
thousands of vectors. It keeps the retrieval logic in plain sight instead of buried in an
ANN index's parameters. The `VectorStore` interface (`add`, `search`, `to_dict`/`from_dict`)
leaves a clean seam to drop in FAISS/pgvector when a corpus actually needs approximate
search at scale.

### Three namespaces

We keep **separate** vector stores for:
- **chunks** — fine-grained passages (classic RAG),
- **community summaries** — corpus-level themes (docs/06),

so a query can search each independently and we can weight them differently. (Entity
embeddings are used during dedup but aren't a retrieval namespace here.)

### Search math

Normalize the query and every stored row, then a single matrix–vector product gives all
cosine similarities at once; `argpartition` grabs the top-k without fully sorting. This is
the whole "vector search" idea in ~10 lines — worth reading to demystify it.

## Persistence

Both stores expose `to_dict`/`from_dict`, and the engine pickles the graph + stores + config
into one `.graphrag` file. Pickle keeps a learning project simple; the interfaces are what
let a production build swap in Neo4j + pgvector without touching the pipeline.

**Lesson:** the "dual index" (graph *and* vectors) is the structural signature of GraphRAG.
Everything upstream feeds these two stores; everything downstream reads from them.

Next: [`06-communities.md`](06-communities.md).
