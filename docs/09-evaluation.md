# 09 — Evaluation: Proving GraphRAG Beats Vector-Only RAG

Source: [`evaluation/`](../src/graphrag_engine/evaluation/),
[`examples/run_eval.py`](../examples/run_eval.py)

A project that *claims* GraphRAG is better must **measure** it. This is the payoff
doc.

## The metric: context recall (deterministic, no judge LLM)

Grading free-text answers usually needs an LLM judge, which is non-deterministic and
un-testable. Instead we measure the **retriever** directly with *context recall*:

- **single_hop / multi_hop** — a *hit* means every expected keyword appears somewhere in the
  retrieved context. Reasoning: if a required fact was never retrieved, no downstream LLM
  could answer from it. So this cleanly measures "did retrieval surface the needed
  evidence?"
- **global** — a *hit* means at least one **community summary** was retrieved, since that is
  the only source of corpus-level thematic context.

This metric is fully deterministic, so it lives in the test suite (`test_engine.py`) and
can't silently regress.

## The question set

`dataset.py` has hand-built questions in three categories. The multi-hop ones are
deliberately constructed so the connecting facts live in **different source files** —
forcing retrieval to bridge documents:

> "Which chip company, founded by a Stanford alum, makes the GPUs OpenAI uses?"

`Nvidia` (a chip company, Jensen Huang studied at Stanford) is in `founders.txt`; the fact
that `OpenAI uses GPUs` is in `ai_labs.txt`. The bridge is the shared `GPUs` node.

## A fair fight

The comparison only means something if the baseline is honest:

- All three modes answer the **same** questions with the **same** config.
- Pure `vector` mode **excludes community summaries** — those are a GraphRAG artifact, so
  giving them to the "vanilla RAG" baseline would be cheating.
- Chunks are kept small and `vector_top_k` modest, so vector search must actually *choose*
  and can't just grab the whole corpus. (With a tiny corpus and large top-k, vector trivially
  retrieves everything and the comparison is meaningless — a real pitfall we hit and fixed.)

## The result

```
category        vector     graph    hybrid
------------------------------------------
single_hop        100%      100%      100%
multi_hop          67%      100%      100%
global              0%        0%      100%
------------------------------------------
OVERALL            62%       75%      100%
```

Reproduce it yourself:

```bash
python examples/run_eval.py
```

## Reading the table

- **single_hop:** everyone wins. Vector search is entirely adequate for direct lookups —
  GraphRAG doesn't *hurt* here.
- **multi_hop:** vector drops to 67% (misses cross-document hops); graph and hybrid hit
  100% by following edges. **This is the core thesis.**
- **global:** only hybrid scores, because only it retrieves community summaries. Pure graph
  traversal has no notion of corpus-level themes; pure vector (baseline) has no communities.
- **overall:** hybrid 100% vs vector 62% — GraphRAG strictly dominates on this corpus.

## Honest caveats

- The offline `MockProvider` does rule-based extraction; a real LLM produces a richer, noisier
  graph. The *shape* of the result (hybrid ≥ vector, winning on multi-hop/global) is what
  generalizes, not the exact percentages.
- Context recall measures retrieval, not answer fluency. Pair it with an LLM-judge metric
  when using real providers for end-to-end answer quality.

**Lesson:** design the evaluation to isolate the claim (multi-hop / global), keep the
baseline honest, and make the metric deterministic so it can guard against regressions.

Next: [`10-cli-and-wrap-up.md`](10-cli-and-wrap-up.md).
