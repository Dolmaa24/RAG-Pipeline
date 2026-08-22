"""Finding which graph entities a question is about.

A traversal needs somewhere to start, and the question rarely names nodes
exactly: "the guy who ran Apple's car project" has to become the node
``Doug Field``. So entity names and descriptions are embedded and searched
semantically, and the hits become the seeds.

Written incrementally. The version this is ported from called
``create_table(..., mode="overwrite")`` with *every* entity in the graph after
*every* document — re-embedding the whole graph to add a handful of nodes, which
is quadratic in the number of documents ingested.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from config import config
from errors import MissingDependency
from observability import get_logger, metrics

from pipeline.graph.schema import Entity
from pipeline.retrieve.filters import quote_literal

log = get_logger("graph.entities")

TABLE_NAME = "graph_entities"


def _table_names(db) -> list[str]:
    """Existing table names, across two LanceDB API generations.

    ``table_names()`` returned a list and is deprecated; ``list_tables()``
    returns a paginated response object whose names live on ``.tables``. One
    place knows that, so the churn does not spread through the module.
    """
    listing = db.list_tables()
    tables = getattr(listing, "tables", listing)
    return list(tables or [])


def _arrow_schema(dimension: int):
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dimension)),
            pa.field("name", pa.string()),
            pa.field("type", pa.string()),
            pa.field("description", pa.string()),
            pa.field("source_url", pa.string()),
        ]
    )


class EntityIndex:
    """Semantic lookup over entity names, in the same LanceDB directory."""

    def __init__(self, db_dir: Optional[str] = None, embedder=None) -> None:
        try:
            import lancedb
        except ImportError as exc:
            raise MissingDependency("lancedb", "graph entity search") from exc

        self.db = lancedb.connect(db_dir or config.LANCE_DB_DIR)
        self._embedder = embedder
        self._table = None
        self._lock = threading.Lock()

    @property
    def embedder(self):
        if self._embedder is None:
            from pipeline.embed.dense import get_dense_embedder

            self._embedder = get_dense_embedder()
        return self._embedder

    @property
    def table(self):
        if self._table is None and TABLE_NAME in _table_names(self.db):
            self._table = self.db.open_table(TABLE_NAME)
        return self._table

    def index(self, entities: list[Entity]) -> int:
        """Add or update these entities only. Not the whole graph."""
        if not entities:
            return 0

        texts = [f"{e.name}: {e.description}".strip(": ") for e in entities]
        with metrics.timer("graph.entities.embed"):
            vectors = self.embedder.embed_documents(texts)

        rows = [
            {
                "id": entity.name,
                "vector": vector,
                "name": entity.name,
                "type": entity.type or "",
                "description": entity.description or "",
                "source_url": entity.source_url or "",
            }
            for entity, vector in zip(entities, vectors)
        ]

        table = self.table
        if table is None:
            with self._lock:
                if TABLE_NAME in _table_names(self.db):
                    self._table = self.db.open_table(TABLE_NAME)
                else:
                    self._table = self.db.create_table(
                        TABLE_NAME, schema=_arrow_schema(len(vectors[0]))
                    )
                table = self._table

        (
            table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
        log.info("graph.entities.indexed", count=len(rows))
        return len(rows)

    def seeds(self, query: str, *, top_k: Optional[int] = None) -> list[str]:
        """Entity names this question is probably about.

        Tries an exact name match before reaching for the embedder. That is not
        only an optimisation: when the caller already has the entity's real
        name — which is the usual case for a graph tool, because the name came
        out of a previous traversal — an exact match is *more* precise than
        nearest-neighbour. Asked for "Acme Corporation", the vector search here
        returned ``['ACME', 'Acme Corporation', 'Beta Industries']``, seeding a
        traversal from a company that merely appears nearby.

        The saving is the whole cost. Embedding one short string loads BGE on
        first use, measured at 10.6s against 0.03s for the Kuzu query it feeds —
        so an exact match answers in milliseconds where the semantic path pays
        for a model the process may not otherwise need at all.
        """
        table = self.table
        if table is None or not query.strip():
            return []

        limit = top_k or config.GRAPH_SEED_ENTITIES

        exact = self._exact(table, query, limit)
        if exact:
            metrics.incr("graph.entities.exact_hit")
            return exact

        try:
            vector = self.embedder.embed_query(query)
            with metrics.timer("graph.entities.search"):
                rows = table.search(vector, vector_column_name="vector").limit(limit).to_list()
        except Exception as exc:
            log.warning("graph.entities.search_failed", error=repr(exc))
            return []
        metrics.incr("graph.entities.semantic_hit")
        return [str(row["name"]) for row in rows if row.get("name")]

    def _exact(self, table, query: str, limit: int) -> list[str]:
        """Names equal to ``query``, ignoring case. Empty when there are none.

        A failure here is not worth reporting: the semantic path runs next and
        answers the same question, more slowly.
        """
        wanted = query.strip()
        try:
            rows = (
                table.search()
                .where(f"LOWER(name) = {quote_literal(wanted.lower())}")
                .select(["name"])
                .limit(limit)
                .to_list()
            )
        except Exception as exc:
            log.debug("graph.entities.exact_failed", error=repr(exc))
            return []
        return [str(row["name"]) for row in rows if row.get("name")]

    def count(self) -> int:
        table = self.table
        return table.count_rows() if table is not None else 0

    def all_names(self, limit: int = 500) -> list[dict[str, Any]]:
        table = self.table
        if table is None:
            return []
        rows = table.search().select(["name", "type", "description"]).limit(limit).to_list()
        return [
            {"name": r.get("name"), "type": r.get("type"), "description": r.get("description")}
            for r in rows
        ]


__all__ = ["EntityIndex", "TABLE_NAME"]
