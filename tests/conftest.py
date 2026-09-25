"""Shared fixtures. All tests run offline via the mock providers."""

from __future__ import annotations

import pytest

from graphrag_engine import Config, GraphRAGEngine

DOCS = {
    "a.txt": (
        "Patrick Collison co-founded Stripe. Patrick Collison studied at MIT. "
        "Sequoia Capital invested in Stripe. Stripe is based in San Francisco."
    ),
    "b.txt": (
        "Jensen Huang co-founded Nvidia. Jensen Huang studied at Stanford University. "
        "Nvidia developed GPUs. OpenAI uses GPUs to train models."
    ),
}


@pytest.fixture
def engine() -> GraphRAGEngine:
    eng = GraphRAGEngine.default(Config(chunk_size=120, chunk_overlap=30, vector_top_k=4))
    eng.ingest_documents(DOCS)
    return eng
