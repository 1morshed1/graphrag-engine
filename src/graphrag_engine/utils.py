"""Small shared helpers: robust JSON parsing and cosine similarity."""

from __future__ import annotations

import json
import re

import numpy as np


def parse_json_object(text: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response.

    Real models wrap JSON in prose or ```json fences. We strip fences, then fall back to
    slicing from the first ``{`` to the last ``}``. Returns ``{}`` if nothing parses, so a
    single malformed response never crashes an ingest of thousands of chunks.
    """
    if not text:
        return {}
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


def parse_json_array(text: str) -> list:
    """Like :func:`parse_json_object` but for a top-level JSON array."""
    if not text:
        return []
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return []
    return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors. Assumes non-zero norms in practice."""
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)
