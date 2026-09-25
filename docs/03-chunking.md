# 03 — Chunking

Source: [`pipeline/chunker.py`](../src/graphrag_engine/pipeline/chunker.py)

## Why chunk?

Two hard constraints force us to split documents:

1. **Model limits.** Embedding models produce *one* vector per input and have a max input
   length. You cannot embed a whole book into a single meaningful vector.
2. **Retrieval granularity.** The chunk is the atomic unit we retrieve. Its size is a
   trade-off:
   - **Small chunks** → precise (a hit is tightly on-topic) but may lack context.
   - **Large chunks** → self-contained but dilute relevance (a hit drags in off-topic text).

## Why overlap?

A fact can straddle a boundary:

```
...Patrick founded Stripe. | He later raised money from Sequoia...
                           ^ chunk boundary
```

Without overlap, "He" (Patrick) and "Sequoia" land in different chunks and the *link*
between them is lost to both. Overlapping windows re-include the tail of the previous chunk
at the head of the next, so a boundary-straddling fact appears **whole** in at least one
chunk. `chunk_overlap` controls how much tail we carry.

## Our approach: sentence-aware packing

Rather than cutting at a hard character count (which can slice "Stanford Univ|ersity" or a
sentence in half — bad for the entity extractor), we:

1. Split text into sentences.
2. Pack whole sentences into a chunk until the next one would exceed `chunk_size`.
3. Start the next chunk by re-including trailing sentences worth up to `chunk_overlap`
   characters.

This keeps every chunk readable and every entity mention intact — which directly improves
extraction quality downstream.

```python
Chunker(chunk_size=800, chunk_overlap=150).chunk_document(text, doc_id="report.txt")
# -> [Chunk(id="report.txt::chunk0", ...), Chunk(id="report.txt::chunk1", ...), ...]
```

The chunk `id` encodes its origin (`doc::chunkN`), which becomes the provenance we track all
the way to the final answer.

**Lesson:** chunking looks trivial but silently determines retrieval quality. Sentence-aware
+ overlap is a cheap, robust default; tuning `chunk_size` is one of the first knobs to turn
if retrieval feels too coarse or too fragmented.

Next: [`04-knowledge-graph.md`](04-knowledge-graph.md).
