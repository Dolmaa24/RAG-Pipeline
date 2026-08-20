"""Pulling entities and relationships out of a chunk with a model.

Ported from ``Dolmaa24/GraphRAG``'s ``src/extractor.py``, onto
:func:`pipeline.extract.llm.get_backend` instead of a directly-constructed Groq
client. That is not tidying: it is how this inherits Ollama/Groq selection,
``local_only`` (graph extraction sends document text to a model, which is
exactly the thing a local-only job must not do), schema validation, and the
retry-with-feedback the extraction cascade already has.

This is **one model call per chunk**, which makes it the most expensive thing in
the pipeline by a wide margin, and the reason graph building is opt-in per job
rather than part of indexing.
"""

from __future__ import annotations

from typing import Optional

from config import config
from errors import MissingDependency
from observability import get_logger, metrics

from pipeline.graph.schema import Entity, KnowledgeGraphExtraction, Relationship

log = get_logger("graph.extract")

_SCHEMA_HINT = {
    "entities": "list of objects",
    "relationships": "list of objects",
}

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relation": {"type": "string"},
                    "description": {"type": "string"},
                    "valid_year": {"type": "string"},
                },
                "required": ["source", "target", "relation", "description"],
            },
        },
    },
    "required": ["entities", "relationships"],
}

_RELATIONS_SCHEMA_HINT = {"relationships": "list of objects"}

_RELATIONS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "relationships": _JSON_SCHEMA["properties"]["relationships"],
    },
    "required": ["relationships"],
}

_RELATIONS_PROMPT = """The entities in this text have already been identified and
are listed below. Do not look for more.

Extract the directional connections the text states between them. source and
target must be names from the list, spelled exactly as given. relation is a verb
in UPPERCASE_SNAKE_CASE, e.g. WORKED_AT, ACQUIRED, LOCATED_IN. Direction follows
the sentence: in "A acquired B" the source is A and the target is B. Include
valid_year when the text gives a date.

Extract only connections the text states. Do not infer one to link things up.

Entities:
{entities}"""


_PROMPT = """Extract a knowledge graph from the text.

entities — every significant person, organisation, location, product,
technology or concept the text actually names. Give each a canonical name (the
fullest form the text uses), a type, and a one-line description drawn from the
text.

relationships — the directional connections the text states between those
entities. source and target must be names from your entities list, spelled
identically. relation is a verb in UPPERCASE_SNAKE_CASE, e.g. WORKED_AT,
ACQUIRED, LOCATED_IN. Include valid_year when the text gives a date.

Extract only what the text supports. Do not infer a relationship the text does
not state, and do not invent entities to connect things up."""


class GraphExtractor:
    """One chunk of text in, a small knowledge graph out.

    Cached on the chunk's content hash. This is the most expensive call in the
    pipeline and a perfectly deterministic one, so a re-crawl of unchanged text
    should cost a lookup rather than the sixteen seconds it cost the first time.
    """

    def __init__(self, backend=None, cache=None, database=None) -> None:
        self._backend = backend
        self._cache = cache
        self._database = database

    @property
    def cache(self):
        if self._cache is None:
            from pipeline.graph.cache import get_cache

            # Process-wide: a cache built per extractor would be empty on every
            # call it was meant to serve.
            self._cache = get_cache(self._database)
        return self._cache

    def extract(
        self,
        text: str,
        *,
        source_url: str = "",
        content_hash: str = "",
        local_only: bool = False,
    ) -> KnowledgeGraphExtraction:
        if not text or not text.strip():
            return KnowledgeGraphExtraction()

        backend = self._backend
        if backend is None:
            from pipeline.extract.llm import get_backend

            backend = get_backend(local_only=local_only)

        # Hash the text when the caller had no hash of its own — a chunk from a
        # crawl carries one, a string handed straight to build_graph does not.
        from pipeline.graph.cache import content_key

        key = content_hash or content_key(text)
        model = getattr(backend, "model", "unknown")

        # The cache key names the prompt, and the two paths ask different
        # questions — a cached whole-graph extraction must not be served to a
        # relationships-only request or the other way round.
        use_gliner = config.GRAPH_ENTITY_BACKEND == "gliner"
        prompt_key = _RELATIONS_PROMPT if use_gliner else _PROMPT

        cached = self.cache.get(key, prompt_key, model)
        if cached is not None:
            return cached

        if use_gliner:
            extraction = self._with_gliner(
                text, backend, source_url=source_url, content_hash=content_hash
            )
            if extraction is not None:
                self.cache.put(key, prompt_key, model, extraction, url=source_url)
                return extraction
            # GLiNER unavailable: fall through to the all-in-one model call.

        try:
            with metrics.timer("graph.extract"):
                response = backend.complete_json(
                    prompt=_PROMPT,
                    content=text[: config.MAX_CHUNK_SIZE],
                    schema_hint=_SCHEMA_HINT,
                    json_schema=_JSON_SCHEMA,
                )
            data = response.data or {}
        except Exception as exc:
            log.warning("graph.extract_failed", url=source_url[:80], error=repr(exc))
            metrics.incr("graph.extract.failed")
            return KnowledgeGraphExtraction()

        entities = _entities(data.get("entities"), source_url, content_hash)
        relationships = _relationships(
            data.get("relationships"), source_url, content_hash, {e.name for e in entities}
        )

        # Before the cache, so a stored extraction is already corrected and a
        # re-ingest does not repeat the work.
        relationships = self._validate(relationships, entities, text, source_url)

        extraction = KnowledgeGraphExtraction(entities=entities, relationships=relationships)
        self.cache.put(key, prompt_key, model, extraction, url=source_url)

        log.info(
            "graph.extracted",
            url=source_url[:80],
            entities=len(entities),
            relationships=len(relationships),
        )
        return extraction


    def _with_gliner(
        self,
        text: str,
        backend,
        *,
        source_url: str,
        content_hash: str,
    ) -> Optional[KnowledgeGraphExtraction]:
        """Entities from GLiNER, relationships from the model.

        Returns ``None`` when GLiNER is not installed, so the caller can fall
        back to the single all-in-one call rather than failing a job over an
        optional dependency.
        """
        try:
            from pipeline.graph.ner import extract_entities
        except Exception as exc:  # pragma: no cover - import-time only
            log.warning("graph.ner.unavailable", error=repr(exc))
            return None

        try:
            entities = extract_entities(
                text, source_url=source_url, content_hash=content_hash
            )
        except MissingDependency as exc:
            log.warning("graph.ner.missing", error=str(exc))
            return None
        except Exception as exc:
            log.warning("graph.ner.failed", error=repr(exc))
            return None

        if not entities:
            return KnowledgeGraphExtraction()

        listing = "\n".join(f"- {e.name} ({e.type})" for e in entities)
        try:
            with metrics.timer("graph.extract_relations"):
                response = backend.complete_json(
                    prompt=_RELATIONS_PROMPT.format(entities=listing),
                    content=text[: config.MAX_CHUNK_SIZE],
                    schema_hint=_RELATIONS_SCHEMA_HINT,
                    json_schema=_RELATIONS_JSON_SCHEMA,
                )
            data = response.data or {}
        except Exception as exc:
            # The entities are real and worth keeping even with no edges; a
            # later document may connect them.
            log.warning("graph.relations_failed", url=source_url[:80], error=repr(exc))
            metrics.incr("graph.extract.failed")
            return KnowledgeGraphExtraction(entities=entities)

        relationships = _relationships(
            data.get("relationships"), source_url, content_hash, {e.name for e in entities}
        )
        relationships = self._validate(relationships, entities, text, source_url)

        log.info(
            "graph.extracted",
            url=source_url[:80],
            entities=len(entities),
            relationships=len(relationships),
            entity_backend="gliner",
        )
        return KnowledgeGraphExtraction(entities=entities, relationships=relationships)

    @staticmethod
    def _validate(relationships, entities, text: str, source_url: str):
        """Flip the edges whose direction is provably backwards."""
        if not config.GRAPH_VALIDATE_DIRECTION or not relationships:
            return relationships

        from pipeline.graph.validate import validate

        corrected, corrections = validate(
            relationships, types={e.name: e.type for e in entities}, text=text
        )
        if corrections:
            log.info(
                "graph.directions_corrected",
                url=source_url[:80],
                count=len(corrections),
            )
        return corrected


def _entities(raw, source_url: str, content_hash: str) -> list[Entity]:
    out: list[Entity] = []
    seen: set[str] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(
            Entity(
                name=name,
                type=str(item.get("type") or "Unknown").strip(),
                description=str(item.get("description") or "").strip(),
                source_url=source_url,
                content_hash=content_hash,
            )
        )
    return out


def _relationships(
    raw, source_url: str, content_hash: str, known: set[str]
) -> list[Relationship]:
    """Keep only edges whose endpoints the model also listed as entities.

    A model that names ``Apple`` in a relationship but never lists it as an
    entity has produced a node with no type and no description. The store would
    create it anyway; dropping it here keeps the graph to things the extraction
    actually described.
    """
    lowered = {name.lower() for name in known}
    out: list[Relationship] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        target = str(item.get("target") or "").strip()
        if not source or not target or source == target:
            continue
        if source.lower() not in lowered or target.lower() not in lowered:
            log.debug("graph.edge_dropped_unknown_endpoint", source=source[:40], target=target[:40])
            continue
        out.append(
            Relationship(
                source=source,
                target=target,
                relation=str(item.get("relation") or "RELATED_TO").strip().upper().replace(" ", "_"),
                description=str(item.get("description") or "").strip(),
                valid_year=str(item.get("valid_year") or "UNKNOWN").strip(),
                source_url=source_url,
                content_hash=content_hash,
            )
        )
    return out


__all__ = ["GraphExtractor"]
