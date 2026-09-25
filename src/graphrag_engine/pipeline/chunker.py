"""Splitting documents into overlapping chunks.

Why chunk at all? Two reasons:
1. LLMs and embedding models have context limits — you cannot embed a whole book as one
   vector and expect it to be useful.
2. Retrieval granularity: a chunk is the smallest unit we retrieve, so its size trades off
   precision (small chunks = focused) against context (large chunks = self-contained).

Why *overlap*? A fact can straddle a chunk boundary ("...founded Stripe. | He then raised
from Sequoia..."). Overlapping windows ensure such straddling facts appear whole in at
least one chunk. See docs/03.
"""

from __future__ import annotations

import re

from ..models import Chunk


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter. Good enough to avoid cutting mid-sentence."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


class Chunker:
    """Sentence-aware, character-budgeted chunker with overlap.

    We pack whole sentences into a chunk until adding the next would exceed ``chunk_size``,
    then start a new chunk that *re-includes* the tail sentences covering ``chunk_overlap``
    characters. Splitting on sentence boundaries (rather than hard character cuts) keeps
    each chunk readable and keeps entities intact for the extractor.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, text: str, doc_id: str) -> list[Chunk]:
        sentences = _split_sentences(text)
        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        index = 0

        for sentence in sentences:
            # +1 for the space we'll join with.
            if current and current_len + len(sentence) + 1 > self.chunk_size:
                chunks.append(self._make_chunk(current, doc_id, index))
                index += 1
                current, current_len = self._carry_overlap(current)

            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            chunks.append(self._make_chunk(current, doc_id, index))

        return chunks

    def _make_chunk(self, sentences: list[str], doc_id: str, index: int) -> Chunk:
        text = " ".join(sentences)
        return Chunk(id=f"{doc_id}::chunk{index}", text=text, doc_id=doc_id, index=index)

    def _carry_overlap(self, sentences: list[str]) -> tuple[list[str], int]:
        """Return the trailing sentences that fit within ``chunk_overlap`` characters.

        These seed the next chunk so a boundary-straddling fact survives intact.
        """
        carried: list[str] = []
        length = 0
        for sentence in reversed(sentences):
            if length + len(sentence) + 1 > self.chunk_overlap:
                break
            carried.insert(0, sentence)
            length += len(sentence) + 1
        return carried, length
