"""Integration tests for the full engine and the evaluation claim."""

from __future__ import annotations

import os

from graphrag_engine import Config, GraphRAGEngine
from graphrag_engine.evaluation.evaluator import evaluate


def test_ingest_builds_graph(engine):
    stats = engine.stats()
    assert stats["entities"] > 0
    assert stats["relations"] > 0
    assert stats["chunks"] > 0


def test_query_returns_grounded_answer(engine):
    answer = engine.query("Where is Stripe based?", mode="hybrid")
    assert answer.text
    assert answer.context  # answer carries the context it used


def test_hybrid_retrieves_multi_hop_connection(engine):
    # "GPUs OpenAI uses" + "Stanford alum who founded the chip company" spans two docs.
    ctx = engine.retrieve("Which company makes the GPUs OpenAI uses?", mode="hybrid")
    blob = " ".join(c.text for c in ctx).lower()
    assert "nvidia" in blob and "gpus" in blob


def test_save_and_load_roundtrip(engine, tmp_path):
    path = os.path.join(tmp_path, "g.graphrag")
    engine.save(path)
    reloaded = GraphRAGEngine.default()
    reloaded.load(path)
    assert reloaded.stats() == engine.stats()
    # Retrieval still works after reload.
    assert reloaded.retrieve("Where is Stripe based?", mode="hybrid")


def test_hybrid_beats_vector_overall():
    """The headline claim: hybrid retrieval has higher overall context recall."""
    cfg = Config(chunk_size=180, chunk_overlap=40, vector_top_k=4, community_top_k=2,
                 final_top_k=8)
    eng = GraphRAGEngine.default(cfg)
    eng.ingest_directory("examples/corpus")
    reports = evaluate(eng)
    assert reports["hybrid"].overall.accuracy > reports["vector"].overall.accuracy
    # And specifically wins (or ties high) on multi-hop.
    assert (
        reports["hybrid"].by_category["multi_hop"].accuracy
        >= reports["vector"].by_category["multi_hop"].accuracy
    )
