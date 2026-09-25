"""Graph retrieval: the part that makes GraphRAG able to answer multi-hop questions.

Pipeline:
1. Ask the LLM which entities the query mentions ("Stanford", "Sequoia").
2. Resolve each to a node in the graph (fuzzy match).
3. Breadth-first traverse up to ``graph_max_hops`` from those nodes, collecting the chunks
   each visited node came from. Closer nodes (fewer hops) score higher.
4. When the query names *multiple* entities, prefer chunks reachable from the traversal —
   these are the passages that connect the query's entities, which is precisely what a
   multi-hop question needs.

Unlike vector search, this can follow relationships: from "Stanford" along ``studied_at``
edges to people, then along ``founded`` edges to companies. See docs/07.
"""

from __future__ import annotations

from ..config import Config
from ..models import RetrievedItem
from ..prompts import render_query_entities
from ..providers.base import LLMProvider
from ..storage.graph_store import GraphStore
from ..utils import parse_json_array


class GraphRetriever:
    def __init__(self, llm: LLMProvider, config: Config):
        self.llm = llm
        self.config = config

    def retrieve(
        self, query: str, graph: GraphStore, chunk_texts: dict[str, str]
    ) -> list[RetrievedItem]:
        names = parse_json_array(self.llm.complete(render_query_entities(query)))
        start_keys: list[str] = []
        for name in names:
            if not isinstance(name, str):
                continue
            node = graph.find_node(name)
            if node and node not in start_keys:
                start_keys.append(node)

        if not start_keys:
            return []

        # BFS tracking the minimum hop distance at which each chunk is reached.
        chunk_depth: dict[str, int] = {}
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(k, 0) for k in start_keys]
        while queue:
            key, depth = queue.pop(0)
            if key in visited:
                continue
            visited.add(key)
            for chunk_id in graph.chunks_for(key):
                if chunk_id not in chunk_depth or depth < chunk_depth[chunk_id]:
                    chunk_depth[chunk_id] = depth
            if depth < self.config.graph_max_hops:
                for neighbor, _ in graph.neighbors(key):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        # Closer chunks score higher: score = 1 / (1 + hop_distance).
        ranked = sorted(chunk_depth.items(), key=lambda kv: kv[1])[: self.config.graph_top_k]
        items: list[RetrievedItem] = []
        for chunk_id, depth in ranked:
            text = chunk_texts.get(chunk_id)
            if text is None:
                continue
            items.append(
                RetrievedItem(
                    id=chunk_id,
                    text=text,
                    kind="chunk",
                    score=1.0 / (1 + depth),
                    source="graph",
                )
            )
        return items
