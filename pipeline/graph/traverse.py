"""Question in, subgraph out.

    question → seed entities (vector) → traversal (Cypher) → triples

Graph retrieval answers the questions flat retrieval structurally cannot. "What
was Doug Field's trajectory before Ford" needs three facts from three documents
joined through a shared entity; no single chunk contains the answer, so no
amount of chunk ranking finds it. The graph has the join.

The traversal is tried twice: a generated query first when the agent is on,
then the fixed template. The generated one can express things the template
cannot, and the template always works — so the pair is strictly better than
either, as long as the generated one is cheap to reject, which it is.
"""

from __future__ import annotations

from typing import Optional

from config import config
from observability import get_logger, metrics

from pipeline.graph.schema import Triple

log = get_logger("graph.traverse")


class GraphRetriever:
    """Seeds a traversal from a question and returns the triples it reaches."""

    def __init__(self, store=None, entities=None, agent=None) -> None:
        self._store = store
        self._entities = entities
        self._agent = agent

    @property
    def store(self):
        """Opened read-only: retrieval never writes, and many readers coexist.

        Kuzu's lock is process-wide and a read-write handle excludes everyone
        else, so opening read-write here would make search fail whenever an
        ingest worker was alive.
        """
        if self._store is None:
            from pipeline.graph.store import GraphStore

            self._store = GraphStore(read_only=True)
        return self._store

    @property
    def entities(self):
        if self._entities is None:
            from pipeline.graph.entities import EntityIndex

            self._entities = EntityIndex()
        return self._entities

    @property
    def agent(self):
        if self._agent is None:
            from pipeline.graph.cypher import CypherAgent

            self._agent = CypherAgent()
        return self._agent

    def retrieve(
        self,
        question: str,
        *,
        seeds: Optional[list[str]] = None,
        hops: Optional[int] = None,
        limit: Optional[int] = None,
        local_only: bool = False,
    ) -> tuple[list[Triple], list[str]]:
        """Triples relevant to the question, and the seeds they came from."""
        depth = min(hops or config.GRAPH_MAX_HOPS, config.GRAPH_MAX_HOPS)
        cap = limit or config.GRAPH_MAX_TRIPLES

        # An empty graph is the normal state until someone asks for one, not a
        # failure worth surfacing. Checking first keeps a Kuzu internal message
        # out of the caller's warnings.
        if self._store is None:
            from pipeline.graph.store import graph_exists

            if not graph_exists():
                log.info("graph.traverse.no_graph")
                return [], []

        found = list(seeds or [])
        if not found:
            found = self.entities.seeds(question)
        if not found:
            log.info("graph.traverse.no_seeds", question=question[:60])
            return [], []

        with metrics.timer("retrieve.graph"):
            triples = self._generated(question, found, depth, cap, local_only)
            if not triples:
                triples = self.store.neighbours(found, hops=depth, limit=cap)

        log.info(
            "graph.traverse.done", seeds=len(found), triples=len(triples), hops=depth
        )
        return triples[:cap], found

    def _generated(
        self, question: str, seeds: list[str], depth: int, cap: int, local_only: bool
    ) -> list[Triple]:
        """Try the model's query. Any failure just means the template runs."""
        if not config.GRAPH_CYPHER_AGENT:
            return []
        try:
            cypher = self.agent.generate(
                question, seeds, max_hops=depth, local_only=local_only
            )
        except Exception as exc:
            log.warning("graph.traverse.agent_failed", error=repr(exc))
            return []
        if not cypher:
            return []
        return self.store.execute_read(cypher, limit=cap)


def render(triples: list[Triple]) -> str:
    """The subgraph as lines of text, for a model to read as context."""
    return "\n".join(triple.render() for triple in triples)


__all__ = ["GraphRetriever", "render"]
