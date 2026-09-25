# 02 — Providers: LLM & Embeddings (and the Mock)

Source: [`providers/`](../src/graphrag_engine/providers/), [`prompts.py`](../src/graphrag_engine/prompts.py)

## The seam

A GraphRAG engine leans on two external capabilities:

- an **LLM** to extract entities/relations, summarize communities, and synthesize answers;
- an **embedding model** to turn text into vectors for similarity search and dedup.

We hide both behind tiny abstract classes — `LLMProvider.complete(prompt)` and
`EmbeddingProvider.embed(texts)`. **Every** model call in the codebase goes through these
two methods. That single seam is what makes the engine:

- **testable** — swap in a deterministic mock, assert on exact outputs;
- **runnable offline** — no key, no network, no GPU required to see it work;
- **vendor-neutral** — OpenAI, Anthropic, or a local model is a one-line swap.

```python
from graphrag_engine.providers import make_providers
llm, embedder = make_providers("mock")      # offline default
# llm, embedder = make_providers("openai")  # needs OPENAI_API_KEY
```

## Prompts live in one file

`prompts.py` holds every template. Each begins with a marker line like `[[TASK:extract]]`.
A real model treats it as a harmless heading; the mock reads it to decide what to do. The
templates also wrap their payload in `<<<TEXT>>> … <<<END>>>` delimiters so the mock can
slice the input back out with `payload_of()`.

**Lesson:** centralizing prompts turns prompt-engineering into a one-file, reviewable diff
and keeps the "contract" between the pipeline and the model in one readable place.

## The MockProvider — predictable, not smart

The mock is the most important provider for *learning*, because it lets the entire pipeline
run and be asserted on. Two parts:

### MockEmbeddingProvider — hashing bag-of-words

Each token is hashed to a bucket in a fixed-length vector (with a sign to spread
collisions), then the vector is L2-normalized. Consequence: **texts sharing vocabulary get
similar vectors**, which is exactly enough structure for cosine similarity search to behave
sensibly in tests. It is not semantic (no synonyms), but it is deterministic and free.

### MockLLMProvider — rule-based extraction

Dispatching on the task marker, it:

- **extract** — finds proper-noun phrases via regex, infers a type from keyword hints,
  and connects entities that surround a known relation verb (`founded`, `invested in`,
  `acquired`, …) in the same sentence.
- **summarize** — builds a title/summary from the member names.
- **answer** — returns the context lines that overlap the question's vocabulary most.
- **query_entities** — extracts proper nouns from the question.

It is intentionally simple. The goal is a **predictable** graph you can write tests against,
not human-quality extraction. For real quality, point `make_providers` at a hosted or local
model — the pipeline code does not change at all.

## Real providers (optional extras)

| Preset | LLM | Embeddings | Needs |
|--------|-----|-----------|-------|
| `mock` | rule-based | hashing | nothing (default) |
| `local` | rule-based | sentence-transformers | `pip install .[embeddings]` |
| `openai` | gpt-4o-mini | text-embedding-3-small | `OPENAI_API_KEY` |
| `anthropic` | claude-sonnet-5 | sentence-transformers | `ANTHROPIC_API_KEY` |

All real providers are **lazy-imported** inside their factory branch, so importing
`graphrag_engine` never forces the OpenAI/Anthropic/torch SDKs to be installed.

Next: [`03-chunking.md`](03-chunking.md).
