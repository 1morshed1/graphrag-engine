"""Reciprocal Rank Fusion (RRF): merge several ranked lists into one.

The graph retriever and the vector retriever each return their own ranked list, on
*incomparable* score scales (a cosine similarity vs. a 1/(1+hop) value). We must not just
concatenate or naively add those scores. RRF sidesteps the scale problem by using only
**rank position**:

    score(item) = sum over lists of  1 / (k + rank_in_that_list)

where ``rank`` is 0-based and ``k`` (default 60) dampens the influence of low ranks. An item
that ranks highly in *either* retriever, or decently in *both*, floats to the top. This is
the same fusion GraphRAG and many hybrid-search systems use because it is simple, robust,
and needs no score calibration. See docs/07.
"""

from __future__ import annotations

from ..models import RetrievedItem


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedItem]], k: int, top_k: int
) -> list[RetrievedItem]:
    scores: dict[str, float] = {}
    best_item: dict[str, RetrievedItem] = {}

    for items in ranked_lists:
        for rank, item in enumerate(items):
            scores[item.id] = scores.get(item.id, 0.0) + 1.0 / (k + rank)
            # Keep one representative object per id; prefer graph-sourced for provenance
            # clarity, otherwise the first seen.
            if item.id not in best_item or item.source == "graph":
                best_item[item.id] = item

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    fused: list[RetrievedItem] = []
    for item_id, fused_score in ordered[:top_k]:
        item = best_item[item_id]
        fused.append(
            RetrievedItem(
                id=item.id,
                text=item.text,
                kind=item.kind,
                score=fused_score,
                source=item.source,
            )
        )
    return fused
