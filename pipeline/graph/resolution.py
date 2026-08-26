"""Deciding when two names are the same thing.

"Apple", "Apple Inc." and "Apple Corp" are one node or three, and getting it
wrong ruins the graph in both directions: three nodes fragment the evidence, one
wrong merge invents a connection that does not exist. Three screens, cheapest
first:

1. **Case-insensitive exact match** — free, and catches most of it.
2. **Cosine similarity** over the names — catches spelling and suffix variants.
3. **A model** — asked only about pairs that survived (2), because it is the
   only screen that can tell "Apple" the company from "Apple" the record label,
   and the only one that costs a network round trip.

The port fixes a cost bug. The original encoded the *entire* candidate pool
inside the loop, once per new entity: ingesting a document with 20 entities into
a graph of 500 meant 10,000 name encodings, and it got quadratically worse as
the graph grew. The pool is embedded once here and extended incrementally.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from config import config
from observability import get_logger, metrics

from pipeline.graph.schema import Entity, Relationship

log = get_logger("graph.resolve")

#: Legal and organisational suffixes that never distinguish two entities.
#: "Apple" and "Apple Inc." are the single most common alias pattern there is,
#: and resolving it deterministically is both free and more reliable than asking
#: a small model — which, asked directly, gets it wrong.
_SUFFIXES = (
    "inc", "inc.", "incorporated", "corp", "corp.", "corporation", "co", "co.",
    "company", "ltd", "ltd.", "limited", "llc", "l.l.c.", "plc", "gmbh", "ag",
    "sa", "s.a.", "nv", "n.v.", "bv", "b.v.", "ab", "as", "oy", "pty",
    "group", "holdings", "holding", "technologies", "labs", "laboratories",
)

_PUNCTUATION = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def canonical_key(name: str) -> str:
    """A name reduced to what actually identifies it.

    Case, punctuation and trailing corporate suffixes are stripped, so
    ``Apple Inc.``, ``apple``, and ``Apple Corp.`` all collapse to ``apple``.
    Stripping is repeated because ``Acme Holdings Ltd.`` carries two.
    """
    text = _SPACES.sub(" ", _PUNCTUATION.sub(" ", (name or "").lower())).strip()
    parts = text.split()
    while len(parts) > 1 and parts[-1] in _SUFFIXES:
        parts.pop()
    return " ".join(parts)


class EntityResolver:
    """Maps newly extracted entities onto the ones already in the graph."""

    def __init__(
        self,
        *,
        threshold: Optional[float] = None,
        embedder=None,
        backend=None,
        verify_with_model: bool = True,
    ) -> None:
        self.threshold = (
            config.GRAPH_RESOLUTION_THRESHOLD if threshold is None else threshold
        )
        self._embedder = embedder
        self._backend = backend
        self.verify_with_model = verify_with_model

    @property
    def embedder(self):
        if self._embedder is None:
            from pipeline.embed.dense import get_dense_embedder

            self._embedder = get_dense_embedder()
        return self._embedder

    def resolve(
        self, new_entities: list[Entity], existing: list[dict[str, Any]]
    ) -> tuple[list[Entity], dict[str, str]]:
        """Return the entities that are genuinely new, and an alias map.

        The alias map is ``{extracted name: canonical name}`` and is what
        :func:`remap_relationships` uses to point edges at the surviving node.
        """
        if not new_entities:
            return [], {}

        pool: list[dict[str, Any]] = [dict(e) for e in existing]
        pool_names = [str(e.get("name", "")) for e in pool]
        # One encode() for everything already known, then extended in place.
        pool_vectors = self._encode(pool_names) if pool_names else []

        aliases: dict[str, str] = {}
        canonical: list[Entity] = []
        lowered = {name.lower(): name for name in pool_names}
        by_key = {canonical_key(name): name for name in pool_names if canonical_key(name)}

        for entity in new_entities:
            name = entity.name

            key = canonical_key(name)
            match = lowered.get(name.lower()) or by_key.get(key)
            if match is not None:
                if match != name:
                    aliases[name] = match
                    metrics.incr("graph.resolve.exact")
                    log.info("graph.resolve.exact", alias=name[:40], canonical=match[:40])
                continue

            if pool_vectors:
                index, score = self._nearest(name, pool_vectors)
                if score >= self.threshold:
                    candidate = pool[index]
                    if not self.verify_with_model or self._same_entity(entity, candidate):
                        canonical_name = str(candidate.get("name", ""))
                        aliases[name] = canonical_name
                        metrics.incr("graph.resolve.merged")
                        log.info(
                            "graph.resolve.merged",
                            alias=name[:40],
                            canonical=canonical_name[:40],
                            similarity=round(score, 3),
                        )
                        continue

            canonical.append(entity)
            pool.append({"name": name, "type": entity.type, "description": entity.description})
            pool_names.append(name)
            pool_vectors.append(self._encode([name])[0])
            lowered[name.lower()] = name
            if key:
                by_key.setdefault(key, name)

        return canonical, aliases

    def _encode(self, names: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(names)

    def _nearest(self, name: str, pool_vectors: list[list[float]]) -> tuple[int, float]:
        """Index of and similarity to the closest pooled name.

        Vectors are already unit-normalised by the embedder, so the dot product
        is the cosine and there is nothing to divide by.
        """
        query = self._encode([name])[0]
        best_index, best_score = 0, -1.0
        for index, vector in enumerate(pool_vectors):
            score = sum(a * b for a, b in zip(query, vector))
            if score > best_score:
                best_index, best_score = index, score
        return best_index, best_score

    def _same_entity(self, new: Entity, candidate: dict[str, Any]) -> bool:
        backend = self._backend
        if backend is None:
            try:
                from pipeline.extract.llm import get_backend

                backend = get_backend()
            except Exception as exc:
                # Without a model, similarity alone decided. Refusing to merge
                # is the recoverable error: a split entity can be merged later,
                # while a wrong merge has already destroyed the distinction.
                log.warning("graph.resolve.no_backend", error=repr(exc))
                return False

        prompt = (
            "Do these two records refer to the same real-world entity?\n\n"
            "Answer TRUE when they are the same thing written differently: an "
            "abbreviation or acronym, a shortened or fuller form of a name, a "
            "former name, or a nickname.\n"
            "Answer FALSE when they are genuinely different things that happen "
            "to share a name — a company and an unrelated product, two people "
            "with the same surname, a city and a company named after it.\n\n"
            "Weigh the descriptions and types, not the spelling. Answer with "
            "is_same true or false."
        )
        content = (
            f"Entity 1 — name: {new.name}; type: {new.type}; description: {new.description}\n"
            f"Entity 2 — name: {candidate.get('name')}; type: {candidate.get('type')}; "
            f"description: {candidate.get('description')}"
        )
        try:
            response = backend.complete_json(
                prompt=prompt,
                content=content,
                schema_hint={"is_same": "boolean"},
                json_schema={
                    "type": "object",
                    "properties": {"is_same": {"type": "boolean"}},
                    "required": ["is_same"],
                },
            )
            return bool((response.data or {}).get("is_same", False))
        except Exception as exc:
            log.warning("graph.resolve.verify_failed", error=repr(exc))
            return False


def remap_relationships(
    relationships: list[Relationship], aliases: dict[str, str]
) -> list[Relationship]:
    """Point edges at canonical entity names, dropping self-loops that creates."""
    if not aliases:
        return relationships

    out: list[Relationship] = []
    for rel in relationships:
        source = aliases.get(rel.source, rel.source)
        target = aliases.get(rel.target, rel.target)
        if source == target:
            # Merging "Apple Inc." into "Apple" turns an edge between them into
            # an edge from a node to itself, which asserts nothing.
            continue
        out.append(rel.model_copy(update={"source": source, "target": target}))
    return out


__all__ = ["EntityResolver", "remap_relationships"]
