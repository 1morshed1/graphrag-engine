"""Unit tests for the individual building blocks."""

from __future__ import annotations

from graphrag_engine.config import Config
from graphrag_engine.graph.community import detect_communities
from graphrag_engine.models import Entity, EntityType, RetrievedItem, normalize_name
from graphrag_engine.pipeline.chunker import Chunker
from graphrag_engine.pipeline.kg_builder import KnowledgeGraphBuilder
from graphrag_engine.providers import make_providers
from graphrag_engine.retrieval.fusion import reciprocal_rank_fusion
from graphrag_engine.storage.graph_store import GraphStore
from graphrag_engine.storage.vector_store import VectorStore
from graphrag_engine.utils import parse_json_array, parse_json_object

# --- models -----------------------------------------------------------------------------

def test_normalize_name_collapses_variants():
    assert normalize_name("  Sequoia   Capital. ") == "sequoia capital"
    assert normalize_name("SEQUOIA capital") == "sequoia capital"


def test_entity_type_coerce_aliases():
    assert EntityType.coerce("company") is EntityType.ORGANIZATION
    assert EntityType.coerce("PERSON") is EntityType.PERSON
    assert EntityType.coerce("nonsense") is EntityType.CONCEPT


def test_entity_merge_unions_chunks_and_keeps_longer_description():
    a = Entity("Stripe", EntityType.ORGANIZATION, "short", {"c1"})
    b = Entity("Stripe", EntityType.ORGANIZATION, "a much longer description", {"c2"})
    a.merge(b)
    assert a.source_chunks == {"c1", "c2"}
    assert a.description == "a much longer description"


# --- chunker ----------------------------------------------------------------------------

def test_chunker_overlaps_and_splits():
    text = " ".join(f"Sentence number {i} here." for i in range(20))
    chunks = Chunker(chunk_size=100, chunk_overlap=30).chunk_document(text, "doc")
    assert len(chunks) > 1
    # Overlap: the tail of one chunk should reappear at the head of the next.
    first_tail = chunks[0].text.split()[-1]
    assert first_tail in chunks[1].text


def test_chunker_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        Chunker(chunk_size=100, chunk_overlap=100)


# --- utils ------------------------------------------------------------------------------

def test_parse_json_handles_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('here you go: {"a": 1} thanks') == {"a": 1}
    assert parse_json_object("not json") == {}
    assert parse_json_array('["x", "y"]') == ["x", "y"]


# --- kg builder dedup -------------------------------------------------------------------

def test_kg_builder_dedups_repeated_entities():
    llm, emb = make_providers("mock")
    from graphrag_engine.models import Chunk

    chunks = [
        Chunk("c0", "Sequoia Capital invested in Stripe.", "d", 0),
        Chunk("c1", "Sequoia Capital invested in Nvidia.", "d", 1),
    ]
    data = KnowledgeGraphBuilder(llm, emb, Config()).build(chunks)
    # "Sequoia Capital" must be ONE node whose provenance covers both chunks.
    seq = data.entities.get("sequoia capital")
    assert seq is not None
    assert seq.source_chunks == {"c0", "c1"}


# --- graph store BFS --------------------------------------------------------------------

def test_graph_bfs_reaches_multi_hop_chunks():
    store = GraphStore()
    for key, chunk in [("a", "ca"), ("b", "cb"), ("c", "cc")]:
        store.add_entity(Entity(key, EntityType.CONCEPT, "", {chunk}))
    from graphrag_engine.models import Relation

    store.add_relation(Relation("a", "rel", "b", "ca"))
    store.add_relation(Relation("b", "rel", "c", "cb"))
    # From 'a', 2 hops should reach c's chunk.
    assert store.bfs_chunks(["a"], max_hops=2) == {"ca", "cb", "cc"}
    # 1 hop should not reach c.
    assert store.bfs_chunks(["a"], max_hops=1) == {"ca", "cb"}


# --- vector store -----------------------------------------------------------------------

def test_vector_store_search_ranks_by_similarity():
    store = VectorStore(dim=3)
    store.add("x", [1.0, 0.0, 0.0], "x")
    store.add("y", [0.0, 1.0, 0.0], "y")
    results = store.search([0.9, 0.1, 0.0], top_k=2)
    assert results[0][0] == "x"  # closest to the query
    assert len(results) == 2


# --- community detection ----------------------------------------------------------------

def test_detect_communities_separates_disconnected_clusters():
    import networkx as nx

    g = nx.MultiDiGraph()
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    g.add_edge("c", "d")
    g.add_edge("d", "c")
    mapping = detect_communities(g)
    assert mapping["a"] == mapping["b"]
    assert mapping["c"] == mapping["d"]
    assert mapping["a"] != mapping["c"]


# --- fusion -----------------------------------------------------------------------------

def test_rrf_rewards_agreement_across_lists():
    list1 = [RetrievedItem("shared", "s", "chunk", 0.9, "graph"),
             RetrievedItem("only1", "o", "chunk", 0.8, "graph")]
    list2 = [RetrievedItem("shared", "s", "chunk", 0.7, "vector"),
             RetrievedItem("only2", "o", "chunk", 0.6, "vector")]
    fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=3)
    # The item appearing in BOTH lists should rank first.
    assert fused[0].id == "shared"
