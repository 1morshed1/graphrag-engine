"""A small, hand-built QA set over ``examples/corpus`` for comparing retrieval modes.

Each item declares the keywords that MUST appear somewhere in the retrieved context for the
question to be answerable. This gives a deterministic, LLM-free metric (context recall) that
still captures the core claim: multi-hop and global questions need more than vector search.

Categories:
  single_hop  answer lives in one passage        -> vector search alone is enough
  multi_hop   answer spans passages via relations -> needs graph traversal
  global      answer is a corpus-level theme      -> needs community summaries
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QAItem:
    question: str
    category: str          # "single_hop" | "multi_hop" | "global"
    expected: list[str]    # keywords that must all appear in retrieved context


# NOTE: the multi-hop items deliberately require connecting facts that live in *different*
# source files, so pure vector search (which pulls whole similar passages) tends to miss at
# least one hop while graph traversal follows the edges between them.
DATASET: list[QAItem] = [
    # --- single hop ---
    QAItem("Where is Stripe based?", "single_hop", ["San Francisco"]),
    QAItem("What did Demis Hassabis co-found?", "single_hop", ["DeepMind"]),
    QAItem("Who invested in Robinhood?", "single_hop", ["Sequoia"]),

    # --- multi hop (cross-document) ---
    QAItem(
        "Which chip company, founded by a Stanford alum, makes the GPUs OpenAI uses?",
        "multi_hop",
        ["Nvidia", "GPUs", "OpenAI"],
    ),
    QAItem(
        "Name a Stanford-educated founder connected to a lab Microsoft invested in.",
        "multi_hop",
        ["Sam Altman", "OpenAI", "Microsoft"],
    ),
    QAItem(
        "What technology company that Sequoia backed builds a file syncing platform?",
        "multi_hop",
        ["Dropbox", "Sequoia"],
    ),

    # --- global ---
    QAItem("What are the main themes across these documents?", "global", ["community"]),
    QAItem("Summarize the groups of related organizations.", "global", ["community"]),
]
