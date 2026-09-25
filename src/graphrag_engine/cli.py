"""Command-line interface: ``graphrag <command>``.

Three commands mirror the engine lifecycle:
  build   ingest a corpus directory and persist the graph to a file
  ask     load a persisted graph and answer a question
  stats   print what was built

Kept deliberately thin — it just parses args and calls the engine. See docs/10.
"""

from __future__ import annotations

import argparse
import sys

from .engine import GraphRAGEngine


def _build(args: argparse.Namespace) -> int:
    engine = GraphRAGEngine.from_preset(args.preset)
    print(f"Ingesting {args.corpus!r} with preset {args.preset!r} ...", file=sys.stderr)
    engine.ingest_directory(args.corpus)
    engine.save(args.out)
    stats = engine.stats()
    print(f"Built graph: {stats}", file=sys.stderr)
    print(f"Saved to {args.out}", file=sys.stderr)
    return 0


def _ask(args: argparse.Namespace) -> int:
    engine = GraphRAGEngine.from_preset(args.preset)
    engine.load(args.graph)
    answer = engine.query(args.question, mode=args.mode)
    print(answer.text)
    if args.show_context:
        print("\n--- context ---", file=sys.stderr)
        for i, item in enumerate(answer.context, 1):
            preview = item.text[:120].replace("\n", " ")
            print(f"[{i}] ({item.source}/{item.kind}) {preview}", file=sys.stderr)
    return 0


def _stats(args: argparse.Namespace) -> int:
    engine = GraphRAGEngine.from_preset(args.preset)
    engine.load(args.graph)
    for key, value in engine.stats().items():
        print(f"{key:12} {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphrag", description="GraphRAG engine CLI")
    parser.add_argument(
        "--preset", default="mock",
        help="provider preset: mock (default), local, openai, anthropic",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="ingest a corpus directory into a graph file")
    p_build.add_argument("corpus", help="directory of .txt/.md documents")
    p_build.add_argument("--out", default="data/graph.graphrag", help="output graph file")
    p_build.set_defaults(func=_build)

    p_ask = sub.add_parser("ask", help="answer a question against a built graph")
    p_ask.add_argument("graph", help="path to a built .graphrag file")
    p_ask.add_argument("question", help="the question to answer")
    p_ask.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "graph"])
    p_ask.add_argument("--show-context", action="store_true", help="print retrieved context")
    p_ask.set_defaults(func=_ask)

    p_stats = sub.add_parser("stats", help="print graph statistics")
    p_stats.add_argument("graph", help="path to a built .graphrag file")
    p_stats.set_defaults(func=_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
