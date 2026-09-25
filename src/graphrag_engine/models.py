"""Core data structures that flow between pipeline stages.

Everything here is a plain ``@dataclass`` on purpose: the shapes of the data moving
through the engine are the most important thing to understand, so we make them explicit
and typed rather than passing around loose dicts. See ``docs/01-data-models.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EntityType(str, Enum):
    """The closed vocabulary of entity types the extractor is allowed to emit.

    A *closed* set (rather than free-form strings) keeps the graph clean: it prevents the
    LLM from inventing near-duplicate types like "Company" vs "Organisation" vs "org".
    """

    PERSON = "Person"
    ORGANIZATION = "Organization"
    LOCATION = "Location"
    TECHNOLOGY = "Technology"
    EVENT = "Event"
    CONCEPT = "Concept"

    @classmethod
    def coerce(cls, value: str) -> "EntityType":
        """Map a raw LLM string onto a known type, defaulting to CONCEPT.

        LLMs return messy casing/synonyms; this normalizes them instead of crashing.
        """
        normalized = (value or "").strip().lower()
        for member in cls:
            if member.value.lower() == normalized:
                return member
        aliases = {
            "company": cls.ORGANIZATION,
            "org": cls.ORGANIZATION,
            "corporation": cls.ORGANIZATION,
            "people": cls.PERSON,
            "human": cls.PERSON,
            "place": cls.LOCATION,
            "city": cls.LOCATION,
            "country": cls.LOCATION,
            "tech": cls.TECHNOLOGY,
            "tool": cls.TECHNOLOGY,
            "product": cls.TECHNOLOGY,
            "idea": cls.CONCEPT,
        }
        return aliases.get(normalized, cls.CONCEPT)


@dataclass
class Chunk:
    """A contiguous window of source text plus provenance metadata."""

    id: str
    text: str
    doc_id: str
    index: int  # position of this chunk within its document
    embedding: list[float] | None = None


@dataclass
class Entity:
    """A node in the knowledge graph.

    ``source_chunks`` records every chunk that mentioned this entity. It is the bridge
    back from a graph node to the raw text — graph retrieval walks nodes, then pulls the
    chunks those nodes point to.
    """

    name: str
    type: EntityType
    description: str
    source_chunks: set[str] = field(default_factory=set)
    embedding: list[float] | None = None
    # Filled in during community detection (see docs/06).
    community: int | None = None

    @property
    def key(self) -> str:
        """The canonical identity used for deduplication and graph node ids."""
        return normalize_name(self.name)

    def merge(self, other: "Entity") -> None:
        """Fold another entity that we decided is the *same* real-world thing into this one.

        We keep the longer description (usually more informative) and union the chunk
        provenance so we never lose a mention.
        """
        self.source_chunks |= other.source_chunks
        if len(other.description) > len(self.description):
            self.description = other.description


@dataclass
class Relation:
    """A directed edge in the knowledge graph: source --relation--> target.

    ``source`` and ``target`` are entity *names* as emitted by the LLM; the builder
    resolves them to canonical entity keys during graph assembly.
    """

    source: str
    relation: str
    target: str
    source_chunk: str
    description: str = ""


@dataclass
class Community:
    """A cluster of densely-connected entities plus its LLM-written summary.

    Communities give us *global* context: instead of only retrieving fine-grained chunks,
    we can retrieve "the fintech-VC cluster summary" to answer broad thematic questions.
    """

    id: int
    entity_keys: list[str]
    level: int = 0  # hierarchy level; 0 = finest
    title: str = ""
    summary: str = ""
    embedding: list[float] | None = None


@dataclass
class RetrievedItem:
    """One unit of context returned by a retriever, before fusion.

    ``kind`` tells the fuser where it came from ("chunk", "community", ...) and ``score``
    is the retriever-local relevance (higher = better). ``source`` names which retriever
    produced it, purely for debugging/observability.
    """

    id: str
    text: str
    kind: str
    score: float
    source: str


@dataclass
class Answer:
    """The final synthesized answer plus the context that produced it (for transparency)."""

    query: str
    text: str
    context: list[RetrievedItem] = field(default_factory=list)


def normalize_name(name: str) -> str:
    """Canonicalize an entity name so surface variants collapse to one key.

    "Sequoia Capital", "sequoia  capital", "Sequoia Capital." -> "sequoia capital".
    This is the *cheap, deterministic* half of deduplication; the *semantic* half
    (embedding similarity) lives in the KG builder. See docs/04.
    """
    lowered = (name or "").strip().lower()
    # Drop trailing punctuation and collapse internal whitespace.
    stripped = lowered.strip(".,;:!?\"'()[]{} \t\n")
    return " ".join(stripped.split())
