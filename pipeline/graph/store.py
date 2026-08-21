"""The Kuzu graph store.

Two things are done differently here than in the original, and both are the
same lesson from opposite ends.

**Every query is parameterised.** The version this is ported from built Cypher
by string interpolation, escaping quotes by hand — on entity names that a
language model invented from scraped web pages. That is untrusted input
constructing a query. An entity legitimately called ``O'Brien`` breaks it; one
called something crafted does worse. Kuzu takes ``parameters={...}``, so nothing
here concatenates a value into a query string.

**Writes and reads use different connections.** The read connection is opened on
a ``read_only=True`` database, which is what makes model-generated traversal
queries safe to run at all — not the prompt asking the model nicely to only
read.

Edges are ``MERGE``d, not ``CREATE``d, so re-ingesting a document updates the
graph instead of doubling it.

**Kuzu is an embedded single-writer database, and the lock is process-wide.**
Measured: many read-only processes coexist, but one read-write handle blocks
every other open — including read-only ones. So a worker that held the graph
open read-write for its lifetime would make search fail for as long as the
worker existed. Retrieval therefore opens ``read_only=True``, and the ingest
path opens read-write, writes, and closes:

    with GraphStore() as graph:
        graph.upsert(entities, relationships)
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

from config import config
from errors import MissingDependency, PersistError
from observability import get_logger

from pipeline.graph.schema import Entity, Relationship, Triple

log = get_logger("graph.store")


def graph_exists(db_path: Optional[str] = None) -> bool:
    """Whether a graph has ever been built at this path.

    Worth asking before opening read-only: Kuzu refuses to create a database in
    that mode, and the error it raises — "Cannot create an empty database under
    READ ONLY mode" — describes its own internals rather than the user's
    situation, which is simply that nothing has been ingested into a graph yet.
    """
    path = db_path or config.KUZU_DB_PATH
    return os.path.exists(path)


class GraphStore:
    """Entities and their relationships, in an embedded Kuzu database."""

    def __init__(self, db_path: Optional[str] = None, *, read_only: bool = False) -> None:
        try:
            import kuzu
        except ImportError as exc:
            raise MissingDependency("kuzu", "the knowledge graph") from exc

        self._kuzu = kuzu
        self.db_path = db_path or config.KUZU_DB_PATH
        self.read_only = read_only
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)

        self._lock = threading.Lock()
        self._db = kuzu.Database(self.db_path, read_only=read_only)
        self._conn = kuzu.Connection(self._db)
        self._read_db = None
        self._read_conn = None

        if not read_only:
            self._init_schema()

    # ------------------------------------------------------------------ #
    # Lifetime — the lock is the reason this matters
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        """Release the database, and with it the lock other processes need."""
        self._read_conn = None
        self._read_db = None
        self._conn = None
        self._db = None

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE NODE TABLE IF NOT EXISTS Entity(
                name STRING,
                type STRING,
                description STRING,
                source_url STRING,
                content_hash STRING,
                PRIMARY KEY (name)
            )
            """
        )
        # A relation key on the edge is what makes MERGE meaningful: two
        # different relationships between the same pair of entities are two
        # edges, but the same one seen twice is one.
        self._conn.execute(
            """
            CREATE REL TABLE IF NOT EXISTS CONNECTS_TO(
                FROM Entity TO Entity,
                relation STRING,
                description STRING,
                valid_year STRING,
                source_url STRING,
                content_hash STRING
            )
            """
        )

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    def upsert(self, entities: list[Entity], relationships: list[Relationship]) -> dict[str, int]:
        """Add or update nodes and edges. Idempotent for the same input."""
        if self.read_only:
            raise PersistError("this GraphStore was opened read-only")
        written = {"entities": 0, "relationships": 0}

        with self._lock:
            for entity in entities:
                try:
                    self._conn.execute(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.type = $type,
                            e.description = $description,
                            e.source_url = $source_url,
                            e.content_hash = $content_hash
                        """,
                        parameters={
                            "name": entity.name,
                            "type": entity.type or "",
                            "description": entity.description or "",
                            "source_url": entity.source_url or "",
                            "content_hash": entity.content_hash or "",
                        },
                    )
                    written["entities"] += 1
                except Exception as exc:
                    log.warning("graph.node_failed", name=entity.name[:60], error=repr(exc))

            for rel in relationships:
                try:
                    # Both endpoints must exist before the edge; a relationship
                    # can name an entity the extractor did not list separately.
                    for endpoint in (rel.source, rel.target):
                        self._conn.execute(
                            "MERGE (e:Entity {name: $name})", parameters={"name": endpoint}
                        )
                    self._conn.execute(
                        """
                        MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                        MERGE (a)-[r:CONNECTS_TO {relation: $relation}]->(b)
                        SET r.description = $description,
                            r.valid_year = $valid_year,
                            r.source_url = $source_url,
                            r.content_hash = $content_hash
                        """,
                        parameters={
                            "source": rel.source,
                            "target": rel.target,
                            "relation": rel.relation or "RELATED_TO",
                            "description": rel.description or "",
                            "valid_year": rel.valid_year or "UNKNOWN",
                            "source_url": rel.source_url or "",
                            "content_hash": rel.content_hash or "",
                        },
                    )
                    written["relationships"] += 1
                except Exception as exc:
                    log.warning(
                        "graph.edge_failed",
                        edge=f"{rel.source[:30]}->{rel.target[:30]}",
                        error=repr(exc),
                    )

        log.info("graph.upsert", **written)
        return written

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    @property
    def read_connection(self):
        """A connection that physically cannot write.

        Model-generated Cypher runs here. The rejection in
        :mod:`pipeline.graph.cypher` is the first line of defence and this is
        the one that holds when that check is wrong.
        """
        if self.read_only:
            # Already read-only: opening a second handle would deadlock against
            # the one this instance is holding.
            return self._conn

        if self._read_conn is None:
            with self._lock:
                if self._read_conn is None:
                    self._read_db = self._kuzu.Database(self.db_path, read_only=True)
                    self._read_conn = self._kuzu.Connection(self._read_db)
        return self._read_conn

    def entities(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = self._rows(
            "MATCH (e:Entity) RETURN e.name, e.type, e.description, e.source_url LIMIT $limit",
            {"limit": limit},
        )
        return [
            {"name": r[0], "type": r[1], "description": r[2], "source_url": r[3]} for r in rows
        ]

    def neighbours(self, names: list[str], *, hops: int = 1, limit: int = 50) -> list[Triple]:
        """Edges within ``hops`` of any named entity.

        Kuzu's variable-length syntax cannot take a parameter for the bound, so
        the depth is formatted in — after being coerced to an int and clamped.
        It is the one number in this module that is not a bound parameter, and
        it never touches caller input.
        """
        if not names:
            return []

        depth = max(1, min(int(hops), config.GRAPH_MAX_HOPS))
        pattern = "-[r:CONNECTS_TO]->" if depth == 1 else f"-[r:CONNECTS_TO*1..{depth}]->"

        query = f"""
            MATCH (a:Entity){pattern}(b:Entity)
            WHERE a.name IN $names OR b.name IN $names
            RETURN a.name, r, b.name
            LIMIT $limit
        """
        try:
            rows = self._rows(query, {"names": names, "limit": limit})
        except Exception as exc:
            log.warning("graph.neighbours_failed", error=repr(exc))
            return []
        return _dedupe([t for row in rows for t in _triples_from_row(row)])

    def execute_read(self, cypher: str, limit: int = 50) -> list[Triple]:
        """Run a read-only Cypher query on the read-only connection."""
        try:
            result = self.read_connection.execute(cypher)
        except Exception as exc:
            log.warning("graph.read_failed", error=repr(exc), query=cypher[:120])
            return []

        triples: list[Triple] = []
        while result.has_next() and len(triples) < limit:
            triples.extend(_triples_from_row(result.get_next()))
        return _dedupe(triples)

    def count(self) -> dict[str, int]:
        entities = self._rows("MATCH (e:Entity) RETURN count(e)", {})
        edges = self._rows("MATCH ()-[r:CONNECTS_TO]->() RETURN count(r)", {})
        return {
            "entities": int(entities[0][0]) if entities else 0,
            "relationships": int(edges[0][0]) if edges else 0,
        }

    def upsert_document(self, *args, **kwargs):  # pragma: no cover - guard
        raise AttributeError("GraphStore writes with upsert(), not upsert_document()")

    def _rows(self, query: str, parameters: dict) -> list[list[Any]]:
        result = self._conn.execute(query, parameters=parameters)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows


def _dedupe(triples: list[Triple]) -> list[Triple]:
    """One edge, one triple.

    A two-hop traversal walks the same first edge on the way to every reachable
    node, so an unfiltered result repeats it once per destination. Handing that
    to a model as context wastes the window and overweights whatever happens to
    be near a well-connected node.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[Triple] = []
    for triple in triples:
        key = (triple.source, triple.relation, triple.target)
        if key not in seen:
            seen.add(key)
            out.append(triple)
    return out


def _triples_from_row(row: list[Any]) -> list[Triple]:
    """Turn one result row into triples, whatever shape it arrived in.

    Kuzu returns three different things depending on the query, and the caller
    should not have to know which ran:

    * A **variable-length path** comes back as ``{"_nodes": [...], "_rels": [...]}``
      where ``_nodes`` holds only the *intermediate* nodes. Expanding it hop by
      hop is worth the trouble — in a two-hop answer the middle node is very
      often the thing being asked about, and collapsing the path to its
      endpoints throws it away.
    * A **single relationship** is one dict of properties.
    * A **generated query** returns plain columns.
    """
    if len(row) == 3 and isinstance(row[1], dict) and "_rels" in row[1]:
        return _path_triples(str(row[0]), row[1], str(row[2]))

    if len(row) == 3 and isinstance(row[1], dict):
        return [_triple(str(row[0]), row[1], str(row[2]))]

    values = [str(v) if v is not None else "" for v in row]
    if len(values) >= 3:
        return [
            Triple(
                source=values[0],
                relation=values[1],
                target=values[2],
                description=values[3] if len(values) > 3 else "",
                valid_year=values[4] if len(values) > 4 else "UNKNOWN",
            )
        ]
    return []


def _path_triples(start: str, path: dict, end: str) -> list[Triple]:
    """Expand a recursive path into one triple per hop."""
    rels = path.get("_rels") or []
    intermediate = [str(node.get("name", "")) for node in (path.get("_nodes") or [])]
    names = [start, *intermediate, end]

    triples: list[Triple] = []
    for index, rel in enumerate(rels):
        if not isinstance(rel, dict):
            continue
        source = names[index] if index < len(names) else start
        target = names[index + 1] if index + 1 < len(names) else end
        triples.append(_triple(source, rel, target))
    return triples


def _triple(source: str, rel: dict, target: str) -> Triple:
    return Triple(
        source=source,
        relation=str(rel.get("relation") or "RELATED_TO"),
        target=target,
        description=str(rel.get("description") or ""),
        valid_year=str(rel.get("valid_year") or "UNKNOWN"),
        source_url=str(rel.get("source_url") or ""),
    )


__all__ = ["GraphStore", "graph_exists"]
