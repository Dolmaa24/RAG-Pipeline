"""Graph extraction, keyed on the content it came from.

Extracting a knowledge graph is the most expensive thing this pipeline does —
one model call per chunk, measured at ~16 seconds on a full chunk locally. It is
also perfectly deterministic given the same text, the same prompt and the same
model, which makes it the ideal thing to cache. Without one, re-crawling an
unchanged corpus pays the whole bill again for a graph it already has.

This reuses :class:`~pipeline.extract.cache.ExtractionCache` rather than adding
a second store. That gets the memory tier, the Mongo tier and the TTL for free,
and keeps one answer to "have we seen this content before". The key discriminates
on more than the content hash:

* **content hash** — different text is a different graph;
* **the prompt** — changing what we ask for changes what comes back;
* **the model** — llama3.2 and gpt-oss do not extract the same graph from the
  same paragraph, and serving one where the other was asked for would be a
  silent quality regression that looks like a cache hit.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Optional

from config import config
from observability import get_logger, metrics

from pipeline.extract.cache import ExtractionCache, cache_key
from pipeline.graph.schema import Entity, KnowledgeGraphExtraction, Relationship

log = get_logger("graph.cache")

#: Bumped when the extraction output shape changes, so old entries do not
#: deserialise into a schema that no longer matches them.
SCHEMA_VERSION = "graph:v1"


def content_key(text: str) -> str:
    """The content hash for a chunk, when the caller did not supply one."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class GraphCache:
    """Finished extractions, addressed by what produced them."""

    def __init__(self, database=None, cache: Optional[ExtractionCache] = None) -> None:
        self._cache = cache or ExtractionCache(database=database)

    @staticmethod
    def key(content_hash: str, prompt: str, model: str) -> str:
        return cache_key(
            content_hash,
            f"{SCHEMA_VERSION}|{model}",
            config.prompt_hash(prompt),
        )

    def get(self, content_hash: str, prompt: str, model: str) -> Optional[KnowledgeGraphExtraction]:
        if not config.GRAPH_CACHE_ENABLED or not content_hash:
            return None

        entry = self._cache.get(self.key(content_hash, prompt, model))
        if not entry:
            return None

        try:
            payload = entry.get("data") or {}
            extraction = KnowledgeGraphExtraction(
                entities=[Entity(**e) for e in payload.get("entities", [])],
                relationships=[Relationship(**r) for r in payload.get("relationships", [])],
            )
        except Exception as exc:
            # A stored entry that no longer parses is a miss, not a crash.
            log.warning("graph.cache.unreadable", error=repr(exc))
            return None

        metrics.incr("graph.cache_hit")
        log.info(
            "graph.cache.hit",
            entities=len(extraction.entities),
            relationships=len(extraction.relationships),
        )
        return extraction

    def put(
        self,
        content_hash: str,
        prompt: str,
        model: str,
        extraction: KnowledgeGraphExtraction,
        *,
        url: str = "",
    ) -> None:
        if not config.GRAPH_CACHE_ENABLED or not content_hash:
            return
        if not extraction.entities and not extraction.relationships:
            # Caching "the model found nothing" would make a transient failure
            # permanent for the TTL.
            return

        self._cache.put(
            self.key(content_hash, prompt, model),
            {
                "entities": [e.model_dump() for e in extraction.entities],
                "relationships": [r.model_dump() for r in extraction.relationships],
            },
            method="graph",
            tier=None,
            confidence=1.0,
            url=url,
            content_hash=content_hash,
        )
        metrics.incr("graph.cache_store")

    def stats(self) -> dict:
        return self._cache.stats()


_instance: Optional[GraphCache] = None
_lock = threading.Lock()


def get_cache(database=None) -> GraphCache:
    """The process-wide cache.

    A per-call instance is worse than useless: the whole point is that the
    *second* extraction of the same text is free, and a cache constructed inside
    the call that would have hit it is empty every time. Held per process, like
    the embedding model, and backed by Mongo when one is configured so it also
    survives ``worker_max_tasks_per_child`` recycling the worker.
    """
    global _instance
    with _lock:
        if _instance is None:
            _instance = GraphCache(database=database)
        elif database is not None and _instance._cache._db is None:
            # First caller had no database, this one does. Attach it rather than
            # discarding the entries already in memory.
            _instance._cache._db = database
        return _instance


def reset() -> None:
    """Drop the cache. Used by tests."""
    global _instance
    with _lock:
        _instance = None


__all__ = ["GraphCache", "SCHEMA_VERSION", "content_key", "get_cache", "reset"]
