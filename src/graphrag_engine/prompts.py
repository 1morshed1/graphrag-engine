"""LLM prompt templates, kept in one place.

Centralizing prompts means (a) you can read the exact instructions the model receives,
(b) the deterministic ``MockProvider`` can recognize each task by a stable marker, and
(c) prompt-tuning is a one-file change. Each template starts with a ``[[TASK:...]]`` marker
line: real models simply ignore it as a heading, while the mock uses it to dispatch.
"""

from __future__ import annotations

# Delimiters the MockProvider slices on to recover the payload text from a prompt.
TEXT_OPEN = "<<<TEXT>>>"
TEXT_CLOSE = "<<<END>>>"

EXTRACTION_PROMPT = f"""[[TASK:extract]]
You extract a knowledge graph from text. Identify entities and the relationships between
them.

For each entity provide:
- name: the canonical name
- type: one of [Person, Organization, Location, Technology, Event, Concept]
- description: one sentence grounded in the text

For each relationship provide:
- source: source entity name (must match an entity name above)
- relation: a short verb phrase, e.g. "founded", "invested_in", "works_at", "located_in"
- target: target entity name (must match an entity name above)

Respond with ONLY valid JSON of the form:
{{"entities": [{{"name": ..., "type": ..., "description": ...}}],
  "relationships": [{{"source": ..., "relation": ..., "target": ...}}]}}

Text:
{TEXT_OPEN}
{{chunk_text}}
{TEXT_CLOSE}
"""

COMMUNITY_SUMMARY_PROMPT = f"""[[TASK:summarize]]
You are summarizing a community of related entities from a knowledge graph. Write a short
title and a 2-4 sentence summary describing what connects these entities and what themes
they represent.

Respond with ONLY valid JSON: {{"title": ..., "summary": ...}}

Community members and their relationships:
{TEXT_OPEN}
{{community_text}}
{TEXT_CLOSE}
"""

ANSWER_PROMPT = f"""[[TASK:answer]]
Answer the user's question using ONLY the context provided. If the context is insufficient,
say so plainly. Be concise and cite concrete entities where relevant.

Question: {{query}}

Context:
{TEXT_OPEN}
{{context}}
{TEXT_CLOSE}
"""

QUERY_ENTITIES_PROMPT = f"""[[TASK:query_entities]]
List the named entities (people, organizations, locations, technologies) mentioned in the
question below. Respond with ONLY a JSON array of strings.

Question:
{TEXT_OPEN}
{{query}}
{TEXT_CLOSE}
"""


def render_extraction(chunk_text: str) -> str:
    return EXTRACTION_PROMPT.replace("{chunk_text}", chunk_text)


def render_community_summary(community_text: str) -> str:
    return COMMUNITY_SUMMARY_PROMPT.replace("{community_text}", community_text)


def render_answer(query: str, context: str) -> str:
    return ANSWER_PROMPT.replace("{query}", query).replace("{context}", context)


def render_query_entities(query: str) -> str:
    return QUERY_ENTITIES_PROMPT.replace("{query}", query)


def payload_of(prompt: str) -> str:
    """Extract the text between the TEXT markers (used by MockProvider)."""
    try:
        start = prompt.index(TEXT_OPEN) + len(TEXT_OPEN)
        end = prompt.index(TEXT_CLOSE, start)
        return prompt[start:end].strip()
    except ValueError:
        return prompt


def task_of(prompt: str) -> str:
    """Read the [[TASK:...]] marker so the mock knows which behavior to run."""
    if prompt.startswith("[[TASK:"):
        return prompt[len("[[TASK:") : prompt.index("]]")]
    return "unknown"
