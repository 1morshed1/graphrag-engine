## Project I: GraphRAG — Knowledge Graph Enhanced Retrieval

### Why This Project

Your CV shows extensive RAG experience with vector search (FAISS, pgvector, ChromaDB). But vanilla vector RAG has a fundamental limitation: it retrieves individual chunks by semantic similarity. It cannot answer multi-hop questions like "Which companies that were founded by Stanford alumni have also received funding from Sequoia?" because the answer requires connecting information across multiple documents.

GraphRAG solves this by building a knowledge graph from your documents (entities and their relationships), then using graph traversal + vector search together for retrieval. It's the next evolution of RAG, and it's what Microsoft Research published about in 2024 and what companies like Notion and Glean are integrating.

You've built standard RAG systems. This project proves you're at the frontier.

### Architecture

```
  Documents                                                                    
   │                                                                           
   ▼                                                                           
┌──────────────────────────────────────────────────────────────────────────────┐
│  GraphRAG Pipeline                                                           │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Phase 1: Knowledge Graph Construction                               │   │
│  │                                                                      │   │
│  │  Document → Chunker → Entity Extractor (LLM) → Relation Extractor   │   │
│  │                                                                      │   │
│  │  For each chunk, the LLM extracts:                                   │   │
│  │  - Entities: (name, type, description)                               │   │
│  │    e.g., ("Sequoia Capital", "VC Firm", "Silicon Valley VC firm")    │   │
│  │  - Relations: (entity_a, relation, entity_b)                         │   │
│  │    e.g., ("Sequoia Capital", "invested_in", "Stripe")               │   │
│  │                                                                      │   │
│  │  Deduplication:                                                      │   │
│  │  - "Sequoia", "Sequoia Capital", "Sequoia Capital Inc" → same node  │   │
│  │  - Use embedding similarity to detect duplicates across documents    │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Phase 2: Community Detection & Summarization                        │   │
│  │                                                                      │   │
│  │  - Run Leiden algorithm on the knowledge graph to find communities   │   │
│  │    (clusters of densely connected entities)                          │   │
│  │  - Generate a summary for each community using an LLM               │   │
│  │    "This community contains VC firms and their portfolio companies   │   │
│  │     in the fintech sector..."                                        │   │
│  │  - Hierarchical communities: drill from high-level themes to         │   │
│  │    specific entity clusters                                          │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Storage                                                             │   │
│  │                                                                      │   │
│  │  Neo4j (or PostgreSQL with recursive CTEs):                          │   │
│  │  - Nodes: entities with properties + embedding                       │   │
│  │  - Edges: relationships with type + source chunk reference           │   │
│  │                                                                      │   │
│  │  pgvector:                                                           │   │
│  │  - Chunk embeddings (same as standard RAG)                           │   │
│  │  - Entity description embeddings                                     │   │
│  │  - Community summary embeddings                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Phase 3: Hybrid Retrieval (Graph + Vector)                          │   │
│  │                                                                      │   │
│  │  Query: "Which Stanford alumni founders received Sequoia funding?"   │   │
│  │                                                                      │   │
│  │  Step 1: Entity Recognition in Query                                 │   │
│  │    → Detected entities: "Stanford", "Sequoia"                        │   │
│  │                                                                      │   │
│  │  Step 2: Graph Traversal                                             │   │
│  │    → Find "Stanford" node → follow "alumni" edges → get people       │   │
│  │    → Find "Sequoia Capital" node → follow "invested_in" edges        │   │
│  │    → Intersect: people connected to both Stanford AND Sequoia        │   │
│  │    → Return subgraph + connected chunks                              │   │
│  │                                                                      │   │
│  │  Step 3: Vector Search (parallel)                                    │   │
│  │    → Standard semantic search on chunk embeddings                    │   │
│  │    → Search community summaries (global context)                     │   │
│  │                                                                      │   │
│  │  Step 4: Merge & Rank                                                │   │
│  │    → Combine graph-retrieved chunks + vector-retrieved chunks         │   │
│  │    → Reciprocal Rank Fusion                                          │   │
│  │    → Feed top-K to LLM for answer synthesis                          │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Evaluation (use Project A's EvalKit)                                │   │
│  │                                                                      │   │
│  │  Compare against standard vector RAG:                                │   │
│  │  - Single-hop questions: GraphRAG ≈ Vector RAG (both good)          │   │
│  │  - Multi-hop questions: GraphRAG >> Vector RAG (the whole point)    │   │
│  │  - Global questions ("What are the main themes?"): GraphRAG wins     │   │
│  │    because community summaries provide high-level context            │   │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Implementation: Entity Extraction with LLM

```python
ENTITY_EXTRACTION_PROMPT = """Extract all entities and relationships from the following text.

For each entity, provide:
- name: the entity's canonical name
- type: one of [Person, Organization, Location, Technology, Event, Concept]
- description: a one-sentence description based on the text

For each relationship, provide:
- source: the source entity name
- relation: the relationship type (e.g., "founded", "invested_in", "works_at", "located_in", "uses")
- target: the target entity name

Text:
{chunk_text}

Respond in JSON:
{
  "entities": [...],
  "relationships": [...]
}"""

class KnowledgeGraphBuilder:
    def process_chunk(self, chunk: str, chunk_id: str) -> tuple[list[Entity], list[Relation]]:
        response = self.llm.complete(
            ENTITY_EXTRACTION_PROMPT.format(chunk_text=chunk)
        )
        data = json.loads(response)
        
        entities = []
        for e in data["entities"]:
            entity = Entity(
                name=e["name"],
                type=e["type"],
                description=e["description"],
                source_chunks=[chunk_id],
            )
            # Check for existing entity with similar name (dedup)
            existing = self.find_similar_entity(entity)
            if existing:
                existing.merge(entity)  # merge descriptions and source chunks
            else:
                entities.append(entity)
        
        relations = []
        for r in data["relationships"]:
            relations.append(Relation(
                source=r["source"],
                relation=r["relation"],
                target=r["target"],
                source_chunk=chunk_id,
            ))
        
        return entities, relations
    
    def find_similar_entity(self, entity: Entity) -> Entity | None:
        """Deduplicate entities using embedding similarity."""
        entity_embedding = self.embedder.embed(entity.name + " " + entity.description)
        
        similar = self.entity_store.search(entity_embedding, top_k=3)
        for match in similar:
            if match.similarity > 0.9:
                return match.entity
            # Also check for exact name match after normalization
            if normalize(match.entity.name) == normalize(entity.name):
                return match.entity
        
        return None
```

### Graph Traversal for Multi-Hop Queries

```python
def graph_retrieve(self, query: str, max_hops: int = 2) -> list[str]:
    # Step 1: Extract entities from the query
    query_entities = self.extract_entities_from_query(query)
    # e.g., ["Stanford", "Sequoia Capital"]
    
    # Step 2: Find matching nodes in the graph
    start_nodes = []
    for entity_name in query_entities:
        node = self.graph.find_node(entity_name)  # fuzzy match
        if node:
            start_nodes.append(node)
    
    # Step 3: BFS traversal up to max_hops
    visited_nodes = set()
    relevant_chunks = set()
    
    queue = [(node, 0) for node in start_nodes]
    
    while queue:
        current_node, depth = queue.pop(0)
        
        if current_node.id in visited_nodes:
            continue
        visited_nodes.add(current_node.id)
        
        # Collect chunks associated with this node
        relevant_chunks.update(current_node.source_chunks)
        
        if depth < max_hops:
            # Expand to neighbors
            neighbors = self.graph.get_neighbors(current_node.id)
            for neighbor, relation in neighbors:
                queue.append((neighbor, depth + 1))
    
    # Step 4: If multiple start nodes, find the intersection subgraph
    # (nodes reachable from ALL start nodes, not just any)
    if len(start_nodes) > 1:
        relevant_chunks = self.intersect_subgraphs(start_nodes, max_hops)
    
    return list(relevant_chunks)
```

### Why This Is Your Natural Next Step

You've already built production RAG systems. GraphRAG is the research frontier — Microsoft published the foundational paper in 2024, and very few production implementations exist. Building one positions you as someone who doesn't just follow tutorials but stays at the cutting edge of the field. Combined with Project A (EvalKit), you can quantitatively prove that GraphRAG outperforms your previous vector-only RAG systems on multi-hop questions.

---
---