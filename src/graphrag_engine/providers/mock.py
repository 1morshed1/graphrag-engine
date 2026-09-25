"""Deterministic, offline provider used for tests and the zero-setup demo.

Nothing here calls a network or needs an API key. The embedder is a hashing bag-of-words
so that texts sharing vocabulary end up with similar vectors (enough to make retrieval
behave sensibly). The LLM is a small rule-based extractor that reads the ``[[TASK:...]]``
marker and produces the JSON the pipeline expects.

This is deliberately *not* trying to be smart — it is trying to be **predictable**, so the
whole engine can be exercised end-to-end and asserted on in unit tests. Swap in a real
provider (``providers/openai.py`` etc.) for real quality. See docs/02.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from ..prompts import payload_of, task_of
from .base import EmbeddingProvider, LLMProvider

# Verb phrases the heuristic relation extractor recognizes, mapped to a canonical relation.
_RELATION_VERBS = {
    "founded": "founded",
    "co-founded": "founded",
    "cofounded": "founded",
    "invested in": "invested_in",
    "funded": "invested_in",
    "backed": "invested_in",
    "acquired": "acquired",
    "works at": "works_at",
    "worked at": "works_at",
    "joined": "works_at",
    "studied at": "studied_at",
    "graduated from": "studied_at",
    "attended": "studied_at",
    "located in": "located_in",
    "based in": "located_in",
    "uses": "uses",
    "built": "built",
    "developed": "developed",
    "leads": "leads",
    "led": "leads",
}

# Cheap type inference keyed on words appearing in the surrounding description.
_TYPE_HINTS = {
    "Organization": ["capital", "ventures", "inc", "corp", "company", "labs", "university",
                     "institute", "bank", "fund", "startup", "firm"],
    "Location": ["valley", "city", "york", "francisco", "london", "california", "street"],
    "Technology": ["python", "gpt", "model", "database", "framework", "api", "platform",
                   "algorithm", "engine", "language"],
}

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with",
    "is", "was", "are", "were", "by", "as", "that", "this", "it", "from", "which", "who",
}

# Common sentence-initial words that get capitalized but are not entities.
_NON_ENTITY_CAPS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "He", "She", "They",
    "We", "In", "On", "At", "For", "But", "And", "Or", "After", "Before", "When",
    "While", "Their", "His", "Her", "Its",
}

_PROPER_NOUN = re.compile(
    r"\b([A-Z][a-zA-Z0-9]*(?:\s+(?:of\s+|&\s+)?[A-Z][a-zA-Z0-9]*)*)"
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Hashing bag-of-words embeddings: deterministic and dependency-free."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for tok in tokens:
            if tok in _STOPWORDS:
                continue
            # Hash the token to a bucket; sign spreads mass to reduce collisions.
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            bucket = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]


class MockLLMProvider(LLMProvider):
    """Rule-based stand-in for a real LLM. Dispatches on the prompt's [[TASK:...]] marker."""

    def complete(self, prompt: str, system: str | None = None) -> str:
        task = task_of(prompt)
        payload = payload_of(prompt)
        if task == "extract":
            return self._extract(payload)
        if task == "summarize":
            return self._summarize(payload)
        if task == "answer":
            return self._answer(prompt, payload)
        if task == "query_entities":
            return json.dumps(self._proper_nouns(payload))
        return ""

    # --- entity / relation extraction -------------------------------------------------

    def _proper_nouns(self, text: str) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        for m in _PROPER_NOUN.finditer(text):
            phrase = m.group(1).strip()
            # Drop leading sentence-starter capitals ("The Stanford lab" -> "Stanford lab"
            # is wrong; instead we reject phrases that are ONLY a non-entity cap word).
            words = phrase.split()
            if len(words) == 1 and phrase in _NON_ENTITY_CAPS:
                continue
            # Strip a leading non-entity capital if followed by more words.
            if words[0] in _NON_ENTITY_CAPS and len(words) > 1:
                phrase = " ".join(words[1:])
            key = phrase.lower()
            if key not in seen and len(phrase) > 1:
                seen.add(key)
                found.append(phrase)
        return found

    def _infer_type(self, name: str, sentence: str) -> str:
        haystack = (name + " " + sentence).lower()
        for etype, hints in _TYPE_HINTS.items():
            if any(h in haystack for h in hints):
                return etype
        # Multi-word names that are not orgs/locations/tech are usually people here.
        if len(name.split()) >= 2:
            return "Person"
        return "Concept"

    def _extract(self, text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        entities: dict[str, dict] = {}
        relations: list[dict] = []

        for sentence in sentences:
            names = self._proper_nouns(sentence)
            for name in names:
                if name.lower() not in entities:
                    entities[name.lower()] = {
                        "name": name,
                        "type": self._infer_type(name, sentence),
                        "description": sentence.strip(),
                    }
            # Relations: for each recognized verb phrase, connect the nearest proper noun
            # on its left to the nearest on its right.
            lowered = sentence.lower()
            for verb, canonical in _RELATION_VERBS.items():
                idx = lowered.find(verb)
                if idx == -1:
                    continue
                left = self._nearest_name(names, sentence, idx, side="left")
                right = self._nearest_name(names, sentence, idx + len(verb), side="right")
                if left and right and left != right:
                    relations.append({"source": left, "relation": canonical, "target": right})

        return json.dumps({"entities": list(entities.values()), "relationships": relations})

    def _nearest_name(self, names: list[str], sentence: str, pos: int, side: str) -> str | None:
        best = None
        best_dist = 10**9
        for name in names:
            i = sentence.find(name)
            if i == -1:
                continue
            if side == "left" and i < pos:
                dist = pos - i
            elif side == "right" and i >= pos:
                dist = i - pos
            else:
                continue
            if dist < best_dist:
                best_dist = dist
                best = name
        return best

    # --- community summarization ------------------------------------------------------

    def _summarize(self, text: str) -> str:
        names = self._proper_nouns(text)
        title = ", ".join(names[:3]) if names else "Community"
        summary = (
            f"This community connects {len(names)} entities including "
            f"{', '.join(names[:5]) if names else 'various concepts'}. "
            "They are linked through the relationships described in the source documents."
        )
        return json.dumps({"title": title[:60], "summary": summary})

    # --- answer synthesis -------------------------------------------------------------

    def _answer(self, prompt: str, context: str) -> str:
        m = re.search(r"Question:\s*(.+)", prompt)
        query = m.group(1).strip() if m else ""
        if not context.strip():
            return "The provided context does not contain enough information to answer."
        # Return the most relevant-looking lines: those overlapping the query vocabulary.
        q_tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in _STOPWORDS}
        lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
        scored = sorted(
            lines,
            key=lambda ln: len(q_tokens & set(re.findall(r"[a-z0-9]+", ln.lower()))),
            reverse=True,
        )
        top = scored[:3]
        return (
            "Based on the retrieved context: " + " ".join(top)
        )
