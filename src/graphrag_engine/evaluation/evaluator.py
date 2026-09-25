"""Run the QA set through each retrieval mode and score context recall.

Metric — **context recall** (deterministic, no judge LLM required):
  - single_hop / multi_hop: a hit means *every* expected keyword appears somewhere in the
    retrieved context. If a needed fact wasn't retrieved, no LLM could answer from it, so
    this measures the retriever directly.
  - global: a hit means at least one *community summary* was retrieved — the only source of
    corpus-level thematic context.

We score three modes side by side (vector = the RAG baseline, graph, hybrid = full system)
so the comparison is apples-to-apples on identical questions. See docs/09.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..engine import GraphRAGEngine
from .dataset import DATASET, QAItem

MODES = ("vector", "graph", "hybrid")


@dataclass
class CategoryScore:
    total: int = 0
    hits: int = 0

    @property
    def accuracy(self) -> float:
        return self.hits / self.total if self.total else 0.0


@dataclass
class ModeReport:
    mode: str
    by_category: dict[str, CategoryScore] = field(default_factory=dict)

    def record(self, category: str, hit: bool) -> None:
        score = self.by_category.setdefault(category, CategoryScore())
        score.total += 1
        score.hits += int(hit)

    @property
    def overall(self) -> CategoryScore:
        agg = CategoryScore()
        for score in self.by_category.values():
            agg.total += score.total
            agg.hits += score.hits
        return agg


def _is_hit(item: QAItem, context) -> bool:
    if item.category == "global":
        return any(c.kind == "community" for c in context)
    blob = " ".join(c.text for c in context).lower()
    return all(kw.lower() in blob for kw in item.expected)


def evaluate(engine: GraphRAGEngine, dataset: list[QAItem] | None = None) -> dict[str, ModeReport]:
    data = dataset or DATASET
    reports = {mode: ModeReport(mode) for mode in MODES}
    for item in data:
        for mode in MODES:
            context = engine.retrieve(item.question, mode=mode)
            reports[mode].record(item.category, _is_hit(item, context))
    return reports


def format_report(reports: dict[str, ModeReport]) -> str:
    """Render a comparison table: rows = category, columns = mode (accuracy %)."""
    categories = ["single_hop", "multi_hop", "global"]
    header = f"{'category':<12}" + "".join(f"{m:>10}" for m in MODES)
    lines = [header, "-" * len(header)]
    for cat in categories:
        row = f"{cat:<12}"
        for mode in MODES:
            score = reports[mode].by_category.get(cat, CategoryScore())
            row += f"{score.accuracy*100:>9.0f}%"
        lines.append(row)
    lines.append("-" * len(header))
    overall = f"{'OVERALL':<12}"
    for mode in MODES:
        overall += f"{reports[mode].overall.accuracy*100:>9.0f}%"
    lines.append(overall)
    return "\n".join(lines)
