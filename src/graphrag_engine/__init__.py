"""GraphRAG Engine — knowledge-graph-enhanced retrieval that runs offline out of the box.

The public entry point is :class:`graphrag_engine.engine.GraphRAGEngine`. Import it lazily
to avoid pulling the whole pipeline in for callers that only need, say, the data models.
"""

__version__ = "0.1.0"


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name == "GraphRAGEngine":
        from .engine import GraphRAGEngine

        return GraphRAGEngine
    if name == "Config":
        from .config import Config

        return Config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
