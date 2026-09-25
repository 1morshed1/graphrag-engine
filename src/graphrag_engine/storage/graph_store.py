"""The knowledge graph itself, backed by ``networkx.MultiDiGraph``.

Why networkx? It gives us battle-tested BFS, neighbor lookups, and (via igraph/leidenalg)
community detection, with zero external services. Why *MultiDiGraph*? Relationships are
**directed** (``founded`` points one way) and there can be **multiple** distinct edges
between the same pair (A both ``founded`` and ``leads`` B), so we need parallel directed
edges.

The node id is the entity's canonical key (``models.normalize_name``); the full ``Entity``
object rides along as a node attribute. Chunks a node came from are stored so traversal can
recover grounding text (see ``chunks_for``).
"""

from __future__ import annotations

import networkx as nx

from ..models import Entity, Relation


class GraphStore:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    # --- construction -----------------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        self.g.add_node(entity.key, entity=entity)

    def add_relation(self, relation: Relation) -> None:
        # Endpoints are already canonical keys resolved by the KG builder.
        if relation.source in self.g and relation.target in self.g:
            self.g.add_edge(
                relation.source,
                relation.target,
                key=relation.relation,
                relation=relation.relation,
                source_chunk=relation.source_chunk,
            )

    # --- lookups ----------------------------------------------------------------------

    def entity(self, key: str) -> Entity | None:
        node = self.g.nodes.get(key)
        return node["entity"] if node else None

    def entities(self) -> list[Entity]:
        return [data["entity"] for _, data in self.g.nodes(data=True)]

    def neighbors(self, key: str) -> list[tuple[str, str]]:
        """Return (neighbor_key, relation) for edges in BOTH directions.

        We traverse undirected for retrieval: if the query mentions Stripe, the fact
        "Patrick founded Stripe" is relevant whether the edge points in or out. Direction
        is preserved in the stored edge for display; traversal just wants connectivity.
        """
        result: list[tuple[str, str]] = []
        for _, tgt, data in self.g.out_edges(key, data=True):
            result.append((tgt, data.get("relation", "")))
        for src, _, data in self.g.in_edges(key, data=True):
            result.append((src, data.get("relation", "")))
        return result

    def find_node(self, name: str) -> str | None:
        """Resolve a (possibly noisy) name to a node key.

        Tries exact canonical match first, then a substring match either way so a query
        saying "Sequoia" finds the "sequoia capital" node.
        """
        from ..models import normalize_name

        target = normalize_name(name)
        if not target:
            return None
        if target in self.g:
            return target
        for key in self.g.nodes:
            if target in key or key in target:
                return key
        return None

    def chunks_for(self, key: str) -> set[str]:
        entity = self.entity(key)
        return set(entity.source_chunks) if entity else set()

    def bfs_chunks(self, start_keys: list[str], max_hops: int) -> set[str]:
        """Collect chunk ids reachable within ``max_hops`` of any start node.

        Classic breadth-first search over the (undirected view of the) graph. Each visited
        node contributes its source chunks. This is the graph-retrieval workhorse: it turns
        "entities mentioned in the query" into "text passages about them and their
        neighborhood". See docs/07.
        """
        visited: set[str] = set()
        chunks: set[str] = set()
        queue: list[tuple[str, int]] = [(k, 0) for k in start_keys if k in self.g]

        while queue:
            key, depth = queue.pop(0)
            if key in visited:
                continue
            visited.add(key)
            chunks |= self.chunks_for(key)
            if depth < max_hops:
                for neighbor, _ in self.neighbors(key):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
        return chunks

    def num_nodes(self) -> int:
        return self.g.number_of_nodes()

    def num_edges(self) -> int:
        return self.g.number_of_edges()
