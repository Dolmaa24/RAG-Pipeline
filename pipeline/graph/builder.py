"""Building the graph from a document: extract, resolve, store, index.

    text → extract → resolve against the graph → remap edges → store → index

Resolution happens *before* the write, not after, because merging two nodes that
are already in the graph means rewriting every edge that touches them. Deciding
at the door is cheap; deciding afterwards is a migration.

The store is opened per document and closed when done. Kuzu's write lock is
process-wide and excludes every other opener — including read-only ones — so a
worker that held it for its lifetime would make search unavailable for exactly
as long as it was ingesting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from observability import get_logger, metrics

log = get_logger("graph.build")


@dataclass
class GraphBuildReport:
    source: str
    entities_extracted: int = 0
    entities_new: int = 0
    entities_merged: int = 0
    relationships: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "entities_extracted": self.entities_extracted,
            "entities_new": self.entities_new,
            "entities_merged": self.entities_merged,
            "relationships": self.relationships,
            "timings_ms": self.timings_ms,
            "warnings": self.warnings,
        }


def build_graph(
    text: str,
    *,
    source_url: str = "",
    content_hash: str = "",
    local_only: bool = False,
    database=None,
    extractor=None,
    resolver=None,
    store=None,
    entity_index=None,
) -> GraphBuildReport:
    """Fold one document's text into the knowledge graph.

    ``database`` is only for the extraction cache. Without it the cache is
    per-process, which still helps inside one worker's life but is lost every
    ``worker_max_tasks_per_child`` recycle — and re-crawls are exactly the case
    the cache exists for.
    """
    from pipeline.graph.entities import EntityIndex
    from pipeline.graph.extractor import GraphExtractor
    from pipeline.graph.resolution import EntityResolver, remap_relationships
    from pipeline.graph.store import GraphStore

    report = GraphBuildReport(source=source_url)
    started = time.perf_counter()

    if not text or not text.strip():
        report.warnings.append("no text to extract from")
        return report

    extractor = extractor or GraphExtractor(database=database)
    resolver = resolver or EntityResolver()
    entity_index = entity_index if entity_index is not None else EntityIndex()

    stage = time.perf_counter()
    extraction = extractor.extract(
        text, source_url=source_url, content_hash=content_hash, local_only=local_only
    )
    report.timings_ms["extract"] = round((time.perf_counter() - stage) * 1000, 2)
    report.entities_extracted = len(extraction.entities)

    if not extraction.entities and not extraction.relationships:
        report.warnings.append("the model found no entities")
        return report

    owned = store is None
    store = store or GraphStore()
    try:
        stage = time.perf_counter()
        canonical, aliases = resolver.resolve(extraction.entities, store.entities())
        relationships = remap_relationships(extraction.relationships, aliases)
        report.timings_ms["resolve"] = round((time.perf_counter() - stage) * 1000, 2)
        report.entities_new = len(canonical)
        report.entities_merged = len(aliases)

        stage = time.perf_counter()
        written = store.upsert(canonical, relationships)
        report.timings_ms["store"] = round((time.perf_counter() - stage) * 1000, 2)
        report.relationships = written["relationships"]
    finally:
        if owned:
            store.close()

    if canonical and entity_index is not None:
        stage = time.perf_counter()
        try:
            entity_index.index(canonical)
        except Exception as exc:
            # The graph is written; a stale seed index degrades traversal but
            # does not lose the facts.
            log.warning("graph.build.index_failed", error=repr(exc))
            report.warnings.append(f"entity index not updated: {exc}")
        report.timings_ms["index"] = round((time.perf_counter() - stage) * 1000, 2)

    report.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
    metrics.incr("graph.documents")
    metrics.incr("graph.entities", report.entities_new)
    log.info(
        "graph.build.done",
        source=source_url[:80],
        new=report.entities_new,
        merged=report.entities_merged,
        edges=report.relationships,
        ms=report.timings_ms["total"],
    )
    return report


__all__ = ["GraphBuildReport", "build_graph"]
