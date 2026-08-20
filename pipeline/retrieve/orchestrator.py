"""Retrieval: a question in, evidence out.

    understand → (hybrid ‖ graph) → fuse → rerank → RetrievalResult

The two legs run concurrently because they are independent and both spend their
time waiting — one on a vector scan and an inverted index, the other on a graph
traversal and possibly a model call. Run serially they add up; run together they
cost the slower of the two.

**Nothing here writes an answer.** Retrieval returns chunks and triples with
their provenance, and the orchestrator that comes next decides what to do with
them. Keeping synthesis out means this layer can be evaluated on whether it
found the right evidence, which is a question with a correct answer, rather than
on whether the prose reads well, which is not.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from config import config
from observability import get_logger, metrics

from pipeline.graph.schema import Triple
from pipeline.retrieve.filters import MetadataFilter
from pipeline.retrieve.hybrid import HybridRetriever, ScoredChunk
from pipeline.retrieve.understand import QueryPlan, understand

log = get_logger("retrieve")


@dataclass
class RetrievalResult:
    """Everything found, and enough about how to explain any of it."""

    query: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)
    plan: Optional[QueryPlan] = None
    seeds: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "triples": [
                {
                    "source": t.source,
                    "relation": t.relation,
                    "target": t.target,
                    "description": t.description,
                    "valid_year": t.valid_year,
                    "source_url": t.source_url,
                    "text": t.render(),
                }
                for t in self.triples
            ],
            "plan": {
                "sub_queries": self.plan.sub_queries if self.plan else [],
                "step_back": self.plan.step_back if self.plan else "",
                "entities": self.plan.entities if self.plan else [],
                "filters": self.plan.filters.model_dump(exclude_none=True) if self.plan else {},
                "rewritten": bool(self.plan and not self.plan.trivial),
            },
            "graph_seeds": self.seeds,
            "timings_ms": self.timings_ms,
            "warnings": self.warnings,
        }

    def context(self, *, max_chunks: int = 10) -> str:
        """The evidence as text, for whatever reasons over it next."""
        parts = []
        for index, chunk in enumerate(self.chunks[:max_chunks], start=1):
            source = chunk.metadata.get("source", "unknown")
            parts.append(f"[{index}] ({source})\n{chunk.document}")
        if self.triples:
            lines = "\n".join(t.render() for t in self.triples)
            parts.append(f"[knowledge graph]\n{lines}")
        return "\n\n".join(parts)


def retrieve(
    query: str,
    *,
    filters: Optional[MetadataFilter] = None,
    limit: Optional[int] = None,
    fusion: Optional[str] = None,
    alpha: Optional[float] = None,
    use_graph: Optional[bool] = None,
    rerank_results: Optional[bool] = None,
    rewrite: Optional[bool] = None,
    local_only: bool = False,
    retriever=None,
    graph=None,
) -> RetrievalResult:
    """Find the evidence for one question."""
    started = time.perf_counter()
    result = RetrievalResult(query=query)

    if not query or not query.strip():
        result.warnings.append("empty query")
        return result

    want_graph = config.GRAPH_ENABLED if use_graph is None else use_graph
    want_rerank = config.RETRIEVE_RERANK if rerank_results is None else rerank_results

    # --- understand ----------------------------------------------------- #
    stage = time.perf_counter()
    if rewrite is False:
        # Asked not to rewrite: make no call at all. Calling and discarding the
        # answer would pay the whole latency for nothing.
        plan = QueryPlan(original=query, filters=filters or MetadataFilter(), trivial=True)
    else:
        plan = understand(
            query,
            local_only=local_only,
            filters=filters,
            force=bool(rewrite),
        )
    result.plan = plan
    result.timings_ms["understand"] = round((time.perf_counter() - stage) * 1000, 2)

    queries = plan.queries()

    # --- retrieve, both legs at once ------------------------------------ #
    stage = time.perf_counter()
    retriever = retriever or HybridRetriever()

    with ThreadPoolExecutor(max_workers=2) as pool:
        vector_leg = pool.submit(
            retriever.retrieve,
            queries,
            filters=plan.filters,
            limit=(limit or config.RETRIEVE_TOP_K) * (3 if want_rerank else 1),
            fusion=fusion,
            alpha=alpha,
        )
        graph_leg = (
            pool.submit(_graph_leg, graph, query, plan, local_only) if want_graph else None
        )

        try:
            result.chunks = vector_leg.result()
        except Exception as exc:
            log.warning("retrieve.vector_leg_failed", error=repr(exc))
            result.warnings.append(f"vector retrieval failed: {exc}")

        if graph_leg is not None:
            try:
                result.triples, result.seeds = graph_leg.result()
            except Exception as exc:
                # A graph that is empty or absent is the common case early on,
                # and must not take the chunks down with it.
                log.warning("retrieve.graph_leg_failed", error=repr(exc))
                result.warnings.append(f"graph retrieval failed: {exc}")

    result.timings_ms["retrieve"] = round((time.perf_counter() - stage) * 1000, 2)

    # --- rerank ---------------------------------------------------------- #
    if want_rerank and result.chunks:
        stage = time.perf_counter()
        from pipeline.retrieve.rerank import rerank as rerank_chunks

        result.chunks = rerank_chunks(query, result.chunks, limit=limit or config.RETRIEVE_TOP_K)
        result.timings_ms["rerank"] = round((time.perf_counter() - stage) * 1000, 2)
    else:
        result.chunks = result.chunks[: limit or config.RETRIEVE_TOP_K]

    result.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
    metrics.incr("retrieve.queries")
    log.info(
        "retrieve.done",
        query=query[:60],
        chunks=len(result.chunks),
        triples=len(result.triples),
        rewritten=not plan.trivial,
        ms=result.timings_ms["total"],
    )
    return result


def _graph_leg(graph, query: str, plan: QueryPlan, local_only: bool):
    from pipeline.graph.traverse import GraphRetriever

    retriever = graph or GraphRetriever()
    return retriever.retrieve(
        query,
        seeds=plan.entities or None,
        local_only=local_only,
    )


__all__ = ["RetrievalResult", "retrieve"]
