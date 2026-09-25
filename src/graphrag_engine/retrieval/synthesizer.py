"""Answer synthesis: turn fused context into a grounded answer.

We format the retrieved items into a numbered context block and instruct the LLM to answer
using *only* that context (reducing hallucination). The context items are attached to the
returned ``Answer`` so callers can inspect exactly what the answer was based on — important
for a system whose selling point is traceable, multi-hop reasoning. See docs/08.
"""

from __future__ import annotations

from ..config import Config
from ..models import Answer, RetrievedItem
from ..prompts import render_answer
from ..providers.base import LLMProvider


class AnswerSynthesizer:
    def __init__(self, llm: LLMProvider, config: Config):
        self.llm = llm
        self.config = config

    def synthesize(self, query: str, context: list[RetrievedItem]) -> Answer:
        context_block = self._format_context(context)
        text = self.llm.complete(render_answer(query, context_block)).strip()
        return Answer(query=query, text=text, context=context)

    def _format_context(self, context: list[RetrievedItem]) -> str:
        lines: list[str] = []
        for i, item in enumerate(context, 1):
            tag = "SUMMARY" if item.kind == "community" else "PASSAGE"
            lines.append(f"[{i}] ({tag}) {item.text}")
        return "\n".join(lines)
