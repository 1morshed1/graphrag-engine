"""Community detection: find clusters of densely-connected entities.

A *community* is a group of nodes with many edges among themselves and few to the outside —
e.g. "a VC firm and its portfolio companies". Summarizing each community (docs/06) gives
the retriever **global** context to answer broad questions ("What are the main themes?")
that no single chunk contains.

The gold-standard algorithm is **Leiden** (an improvement on Louvain that avoids badly-
connected communities). It needs ``igraph`` + ``leidenalg``, which are optional. When they
are absent we fall back to networkx's greedy modularity communities — same *idea* (maximize
modularity), lower quality, zero extra dependencies. This keeps the engine fully offline by
default while allowing the better algorithm when installed. See docs/06.
"""

from __future__ import annotations

import networkx as nx


def detect_communities(graph: nx.MultiDiGraph, resolution: float = 1.0) -> dict[str, int]:
    """Return a mapping ``node_key -> community_id``.

    Operates on an undirected, weighted projection of the graph (parallel edges become an
    integer weight), because community detection is defined over undirected weighted graphs.
    """
    if graph.number_of_nodes() == 0:
        return {}

    undirected = _to_weighted_undirected(graph)

    try:
        return _leiden(undirected, resolution)
    except ImportError:
        return _greedy_modularity(undirected)


def _to_weighted_undirected(graph: nx.MultiDiGraph) -> nx.Graph:
    """Collapse a MultiDiGraph into an undirected Graph with edge 'weight' = #parallel edges."""
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        if u == v:
            continue
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += 1
        else:
            simple.add_edge(u, v, weight=1)
    return simple


def _leiden(graph: nx.Graph, resolution: float) -> dict[str, int]:
    """Leiden via igraph/leidenalg. Raises ImportError if the extras aren't installed."""
    import igraph as ig
    import leidenalg

    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    edges = [(index[u], index[v]) for u, v in graph.edges()]
    weights = [graph[u][v]["weight"] for u, v in graph.edges()]

    g = ig.Graph(n=len(nodes), edges=edges)
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
    )
    return {nodes[i]: comm_id for comm_id, community in enumerate(partition) for i in community}


def _greedy_modularity(graph: nx.Graph) -> dict[str, int]:
    """Fallback: networkx greedy modularity maximization. Pure-Python, always available."""
    from networkx.algorithms.community import greedy_modularity_communities

    if graph.number_of_edges() == 0:
        # No edges -> every node is its own community.
        return {node: i for i, node in enumerate(graph.nodes())}

    communities = greedy_modularity_communities(graph, weight="weight")
    mapping: dict[str, int] = {}
    for comm_id, members in enumerate(communities):
        for node in members:
            mapping[node] = comm_id
    return mapping
