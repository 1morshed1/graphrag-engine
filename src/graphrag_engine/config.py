"""One place for every tunable knob in the engine.

Keeping configuration in a single dataclass (rather than scattering constants across
modules) makes experiments reproducible: you can print the config alongside an evaluation
run and know exactly what produced the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- Chunking (docs/03) ---
    chunk_size: int = 800  # target characters per chunk
    chunk_overlap: int = 150  # characters shared between neighboring chunks

    # --- Entity deduplication (docs/04) ---
    # Cosine similarity above which two entity embeddings are treated as the same entity.
    # Higher = stricter (fewer merges, more duplicate nodes). 0.90 is a sane default.
    dedup_threshold: float = 0.90

    # --- Community detection (docs/06) ---
    # Leiden resolution: higher -> more, smaller communities.
    community_resolution: float = 1.0
    # Communities smaller than this are not summarized (too little signal).
    min_community_size: int = 2

    # --- Retrieval (docs/07) ---
    vector_top_k: int = 8  # chunks pulled by pure vector search
    community_top_k: int = 3  # community summaries pulled for global context
    graph_max_hops: int = 2  # BFS depth from query-matched entities
    graph_top_k: int = 8  # cap on chunks returned by graph traversal
    # Reciprocal Rank Fusion constant. Larger = flatter weighting across ranks.
    rrf_k: int = 60
    final_top_k: int = 10  # context items handed to the answer synthesizer

    # --- Answer synthesis (docs/08) ---
    max_answer_tokens: int = 512
