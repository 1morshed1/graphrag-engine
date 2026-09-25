"""Turn detected communities into summarized, embeddable objects.

Detection (graph/community.py) gives us ``node -> community_id``. Here we:
1. group entities by community id,
2. drop communities below ``min_community_size`` (too little signal to summarize),
3. build a text description of each community (its entities + their internal edges),
4. ask the LLM to write a title + summary,
5. embed the summary so it can be retrieved by vector search.

The community summaries are what let GraphRAG answer *global* questions. See docs/06.
"""

from __future__ import annotations

from ..config import Config
from ..graph.community import detect_communities
from ..models import Community
from ..prompts import render_community_summary
from ..providers.base import EmbeddingProvider, LLMProvider
from ..storage.graph_store import GraphStore
from ..utils import parse_json_object


class CommunityBuilder:
    def __init__(self, llm: LLMProvider, embedder: EmbeddingProvider, config: Config):
        self.llm = llm
        self.embedder = embedder
        self.config = config

    def build(self, graph: GraphStore) -> list[Community]:
        mapping = detect_communities(graph.g, self.config.community_resolution)
        # Stamp the community id back onto each entity (useful for display/debugging).
        for key, comm_id in mapping.items():
            entity = graph.entity(key)
            if entity:
                entity.community = comm_id

        grouped: dict[int, list[str]] = {}
        for key, comm_id in mapping.items():
            grouped.setdefault(comm_id, []).append(key)

        communities: list[Community] = []
        for comm_id, keys in grouped.items():
            if len(keys) < self.config.min_community_size:
                continue
            community = Community(id=comm_id, entity_keys=keys)
            self._summarize(community, graph)
            communities.append(community)
        return communities

    def _summarize(self, community: Community, graph: GraphStore) -> None:
        description = self._describe(community, graph)
        raw = parse_json_object(self.llm.complete(render_community_summary(description)))
        community.title = (raw.get("title") or "").strip() or f"Community {community.id}"
        community.summary = (raw.get("summary") or "").strip() or description
        community.embedding = self.embedder.embed_one(
            f"{community.title}. {community.summary}"
        )

    def _describe(self, community: Community, graph: GraphStore) -> str:
        """Build the raw text fed to the summarizer: entities and their internal relations."""
        lines: list[str] = []
        member_set = set(community.entity_keys)
        for key in community.entity_keys:
            entity = graph.entity(key)
            if entity:
                lines.append(f"- {entity.name} ({entity.type.value}): {entity.description}")
        lines.append("Relationships:")
        for src, tgt, data in graph.g.edges(data=True):
            if src in member_set and tgt in member_set:
                s, t = graph.entity(src), graph.entity(tgt)
                if s and t:
                    lines.append(f"- {s.name} {data.get('relation','')} {t.name}")
        return "\n".join(lines)
