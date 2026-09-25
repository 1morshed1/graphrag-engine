"""Knowledge-graph construction: the heart of GraphRAG.

For every chunk we ask the LLM to extract entities and relationships, then we **merge**
those local extractions into one global graph. The hard part is *deduplication*: the same
real-world entity ("Sequoia", "Sequoia Capital", "Sequoia Capital Inc.") appears with
different surface forms across chunks and must collapse to a single node — otherwise
traversal breaks (edges land on different nodes that are really the same thing).

Dedup has two layers, cheap-first:
1. **Deterministic** — normalized-name equality (``models.normalize_name``).
2. **Semantic** — embedding cosine similarity above ``Config.dedup_threshold``, which
   catches variants that don't normalize equal ("Sequoia" vs "Sequoia Capital").

See docs/04.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..models import Entity, EntityType, Relation, normalize_name
from ..prompts import render_extraction
from ..providers.base import EmbeddingProvider, LLMProvider
from ..utils import cosine_similarity, parse_json_object


@dataclass
class GraphData:
    """The assembled graph: entities keyed by canonical name, plus the relation list."""

    entities: dict[str, Entity]
    relations: list[Relation]


class KnowledgeGraphBuilder:
    def __init__(self, llm: LLMProvider, embedder: EmbeddingProvider, config: Config):
        self.llm = llm
        self.embedder = embedder
        self.config = config

    def build(self, chunks) -> GraphData:
        entities: dict[str, Entity] = {}
        relations: list[Relation] = []
        seen_relations: set[tuple[str, str, str]] = set()
        # Maps every raw normalized name we've seen to the canonical entity key it resolved
        # to. This is what lets a relation naming "Sequoia" find the node stored under
        # "sequoia capital" after semantic dedup merged them.
        alias: dict[str, str] = {}

        for chunk in chunks:
            raw = parse_json_object(self.llm.complete(render_extraction(chunk.text)))
            for raw_entity in raw.get("entities", []):
                self._absorb_entity(raw_entity, chunk.id, entities, alias)
            for raw_rel in raw.get("relationships", []):
                self._absorb_relation(
                    raw_rel, chunk.id, entities, relations, seen_relations, alias
                )

        return GraphData(entities=entities, relations=relations)

    # --- entities ---------------------------------------------------------------------

    def _absorb_entity(
        self, raw: dict, chunk_id: str, entities: dict[str, Entity], alias: dict[str, str]
    ) -> None:
        name = (raw.get("name") or "").strip()
        if not name:
            return
        entity = Entity(
            name=name,
            type=EntityType.coerce(raw.get("type", "")),
            description=(raw.get("description") or "").strip(),
            source_chunks={chunk_id},
        )

        # Layer 1: exact canonical-name match.
        key = entity.key
        if key in entities:
            entities[key].merge(entity)
            alias[key] = key
            return

        # Layer 2: semantic match against existing entities.
        entity.embedding = self.embedder.embed_one(f"{entity.name}. {entity.description}")
        match = self._find_semantic_duplicate(entity, entities)
        if match is not None:
            match.merge(entity)
            alias[key] = match.key
            return

        entities[key] = entity
        alias[key] = key

    def _find_semantic_duplicate(
        self, entity: Entity, entities: dict[str, Entity]
    ) -> Entity | None:
        """Return an existing entity whose embedding is close enough to be the same thing.

        Brute-force over current entities: fine at the corpus sizes this engine targets,
        and keeps the logic transparent. For very large corpora you'd back this with an ANN
        index — but that's an optimization, not a change in the idea.
        """
        best: Entity | None = None
        best_sim = 0.0
        for other in entities.values():
            if other.embedding is None:
                continue
            # Only merge same-typed entities: a Person and an Organization with similar
            # names are not the same node.
            if other.type != entity.type:
                continue
            sim = cosine_similarity(entity.embedding, other.embedding)
            if sim > best_sim:
                best_sim, best = sim, other
        if best is not None and best_sim >= self.config.dedup_threshold:
            return best
        return None

    # --- relations --------------------------------------------------------------------

    def _absorb_relation(
        self,
        raw: dict,
        chunk_id: str,
        entities: dict[str, Entity],
        relations: list[Relation],
        seen: set[tuple[str, str, str]],
        alias: dict[str, str],
    ) -> None:
        relation = (raw.get("relation") or "").strip()
        # Resolve each endpoint through the alias map to the canonical deduped node key.
        source = alias.get(normalize_name(raw.get("source", "")))
        target = alias.get(normalize_name(raw.get("target", "")))
        if not source or not target or not relation or source == target:
            return
        # A dangling edge (endpoint that never resolved to a real node) cannot be traversed
        # usefully, so we drop it. The extractor is imperfect and this keeps the graph sound.
        if source not in entities or target not in entities:
            return
        signature = (source, relation, target)
        if signature in seen:
            return
        seen.add(signature)
        relations.append(
            Relation(source=source, relation=relation, target=target, source_chunk=chunk_id)
        )
