"""LanceDB: dense vectors, BM25 full text, and metadata filters in one table.

Chroma was here first and could not do the job. It is HNSW over float32 with no
product or scalar quantisation, and no full-text index — so "hybrid search" on
it meant hand-rolled term counts with no corpus IDF, which is not BM25 and
cannot be ranked with. LanceDB indexes the same rows three ways:

* **IVF_PQ** over the vector column — approximate nearest neighbour, with the
  vectors compressed. 384 float32 is 1536 bytes; 48 sub-vectors at 8 bits is 48.
* **BM25** over the text column, with real inverse document frequency computed
  across the corpus.
* **B-tree scalar indexes** over every filter column, which is what lets a
  filter run *before* the vector search rather than after it.

That last one is the difference between asking for ten results and getting ten,
versus asking for ten, filtering afterwards, and getting two.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from config import config
from errors import MissingDependency, PersistError
from observability import get_logger, metrics

from pipeline.chunk.models import Chunk, ChunkedDocument
from pipeline.store.schema import (
    FILTER_FIELDS,
    ID_FIELD,
    TEXT_FIELD,
    VECTOR_FIELD,
    arrow_schema,
)

log = get_logger("store.lance")


def _table_names(db) -> list[str]:
    """Existing table names, across two LanceDB API generations.

    ``table_names()`` returned a list and is deprecated; ``list_tables()``
    returns a paginated response object whose names live on ``.tables``. One
    place knows that, so the churn does not spread through the module.
    """
    listing = db.list_tables()
    tables = getattr(listing, "tables", listing)
    return list(tables or [])


def _row(chunk: Chunk) -> dict[str, Any]:
    """One chunk as a table row. Every column present, none of them None."""
    meta = chunk.metadata
    extra = meta.extra or {}
    return {
        ID_FIELD: chunk.id,
        VECTOR_FIELD: chunk.dense_embedding,
        TEXT_FIELD: chunk.document,
        "source": meta.source or "",
        "doc_type": meta.doc_type or "",
        "department": meta.department or "",
        "date": meta.date or "",
        "author": meta.author or "",
        "region": meta.region or "",
        "permission_level": meta.permission_level or "",
        "language": meta.language or "",
        "content_hash": str(extra.get("content_hash", "")),
        "page_no": int(meta.page_no),
        "section_name": meta.section_name or "",
        "chunk_strategy": meta.chunk_strategy or "",
        "embedding_model": meta.embedding_model or "",
        "extraction_tier": int(extra.get("extraction_tier", -1) or -1),
        "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class LanceStore:
    """The chunk table. One instance per process; the handle is cheap to hold."""

    def __init__(
        self,
        db_dir: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> None:
        try:
            import lancedb
        except ImportError as exc:
            raise MissingDependency("lancedb", "vector storage") from exc

        self.db_dir = db_dir or config.LANCE_DB_DIR
        self.table_name = table_name or config.LANCE_TABLE_NAME
        self.db = lancedb.connect(self.db_dir)
        self._table = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Table lifecycle
    # ------------------------------------------------------------------ #

    @property
    def table(self):
        """The table, or None when nothing has been written yet."""
        if self._table is None and self.table_name in _table_names(self.db):
            self._table = self.db.open_table(self.table_name)
        return self._table

    def _ensure_table(self, dimension: int, model: str):
        """Open the table, creating it against this embedding width if new."""
        existing = self.table
        if existing is not None:
            self._check_dimension(existing, dimension, model)
            return existing

        with self._lock:
            if self.table_name in _table_names(self.db):
                self._table = self.db.open_table(self.table_name)
                return self._table

            self._table = self.db.create_table(
                self.table_name, schema=arrow_schema(dimension)
            )
            log.info(
                "store.lance.table_created",
                table=self.table_name,
                dimension=dimension,
                model=model,
            )
            return self._table

    @staticmethod
    def _check_dimension(table, dimension: int, model: str) -> None:
        """Refuse a write in a different embedding space than the table holds.

        Cosine distance between a 384-d BGE vector and a 512-d CLIP vector is
        not a bigger or smaller number, it is a category error. LanceDB would
        raise something about Arrow list widths; this says what to do about it.
        """
        field = table.schema.field(VECTOR_FIELD)
        width = getattr(field.type, "list_size", None)
        if width is not None and width > 0 and width != dimension:
            raise PersistError(
                f"table {table.name!r} holds {width}-dimensional vectors and this "
                f"write is {dimension}-dimensional ({model}). Use a separate "
                "table per embedding model — LANCE_TABLE_NAME.",
                expected=width,
                got=dimension,
            )

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    def upsert_document(self, chunked_doc: ChunkedDocument, batch_size: int = 256) -> int:
        """Write every chunk, replacing any row with the same id."""
        chunks = chunked_doc.chunks
        if not chunks:
            return 0

        missing = [chunk.id for chunk in chunks if not chunk.dense_embedding]
        if missing:
            raise PersistError(
                f"{len(missing)} chunk(s) have no dense embedding; run "
                "DocumentEmbedder before storing",
                first=missing[0],
            )

        dimension = len(chunks[0].dense_embedding)
        table = self._ensure_table(dimension, chunks[0].metadata.embedding_model or "unknown")

        written = 0
        for start in range(0, len(chunks), batch_size):
            rows = [_row(chunk) for chunk in chunks[start : start + batch_size]]
            (
                table.merge_insert(ID_FIELD)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )
            written += len(rows)
            log.info("store.lance.upsert_batch", count=len(rows), table=self.table_name)

        self.ensure_indexes()
        return written

    # ------------------------------------------------------------------ #
    # Indexes
    # ------------------------------------------------------------------ #

    def ensure_indexes(self, *, force_ann: bool = False) -> dict[str, Any]:
        """Build whatever indexes the current row count justifies.

        The ANN index is the interesting one. IVF_PQ *trains* — it clusters the
        data to build the partitions and learns the quantisation codebook — so
        on a small table it is both slower than a flat scan and less accurate
        than one. Below ``INDEX_ANN_MIN_ROWS`` the table is left unindexed and
        every search is exact brute-force KNN, which on a few thousand vectors
        is a millisecond.
        """
        table = self.table
        if table is None:
            return {"built": []}

        from lancedb.index import FTS, BTree, IvfPq

        built: list[str] = []
        rows = table.count_rows()

        try:
            table.create_index(TEXT_FIELD, config=FTS(), replace=True)
            built.append("fts")
        except Exception as exc:  # an unindexed table still answers vector queries
            log.warning("store.lance.fts_index_failed", error=repr(exc))

        for column in FILTER_FIELDS:
            try:
                table.create_index(column, config=BTree(), replace=True)
                built.append(f"scalar:{column}")
            except Exception as exc:
                log.warning("store.lance.scalar_index_failed", column=column, error=repr(exc))

        if force_ann or rows >= config.INDEX_ANN_MIN_ROWS:
            try:
                table.create_index(
                    VECTOR_FIELD,
                    config=IvfPq(
                        distance_type="cosine",
                        num_partitions=min(
                            config.INDEX_IVF_PARTITIONS, max(1, rows // 256)
                        ),
                        num_sub_vectors=config.INDEX_PQ_SUB_VECTORS,
                    ),
                    replace=True,
                )
                built.append("ivf_pq")
                log.info("store.lance.ann_index_built", rows=rows)
            except Exception as exc:
                log.warning("store.lance.ann_index_failed", rows=rows, error=repr(exc))
        else:
            log.debug(
                "store.lance.ann_index_skipped",
                rows=rows,
                threshold=config.INDEX_ANN_MIN_ROWS,
                reason="exact search is faster and exact below the threshold",
            )

        return {"built": built, "rows": rows}

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #

    def search_dense(
        self,
        vector: list[float],
        *,
        limit: int = 50,
        where: Optional[str] = None,
    ) -> list[dict]:
        """Nearest neighbours by cosine distance, filtered before the search."""
        table = self.table
        if table is None:
            return []

        query = table.search(vector, vector_column_name=VECTOR_FIELD).limit(limit)
        if where:
            # prefilter=True is the whole point: the filter narrows the set the
            # search runs over. Post-filtering asks for k and then throws some
            # away, so a selective filter returns fewer than k — or nothing.
            query = query.where(where, prefilter=True)
        query = query.nprobes(config.RETRIEVE_NPROBES).refine_factor(
            config.RETRIEVE_REFINE_FACTOR
        )

        with metrics.timer("retrieve.dense"):
            return _normalize(query.to_list(), score_key="_distance", higher_is_better=False)

    def search_fts(
        self,
        text: str,
        *,
        limit: int = 50,
        where: Optional[str] = None,
    ) -> list[dict]:
        """BM25 over the text column — real IDF, computed across the corpus."""
        table = self.table
        if table is None or not text.strip():
            return []

        try:
            query = table.search(text, query_type="fts").limit(limit)
            if where:
                query = query.where(where, prefilter=True)
            with metrics.timer("retrieve.fts"):
                return _normalize(query.to_list(), score_key="_score", higher_is_better=True)
        except Exception as exc:
            # A table with no FTS index yet, or a query of pure stopwords.
            log.warning("store.lance.fts_failed", error=repr(exc))
            return []

    def count(self, where: Optional[str] = None) -> int:
        table = self.table
        if table is None:
            return 0
        return table.count_rows(filter=where) if where else table.count_rows()

    def distinct(self, column: str, limit: int = 200) -> list[str]:
        """The values a filter column actually holds. For building a UI."""
        table = self.table
        if table is None or column not in FILTER_FIELDS:
            return []
        rows = table.search().select([column]).limit(limit * 20).to_list()
        return sorted({str(row.get(column, "")) for row in rows} - {""})[:limit]


def _normalize(
    rows: Iterable[dict], *, score_key: str, higher_is_better: bool
) -> list[dict]:
    """Attach a comparable ``score`` to each row, best first.

    The two legs speak different languages: cosine *distance* where smaller is
    better, BM25 *relevance* where larger is. Fusion cannot mix those, so each
    leg is min-max normalised to 0..1 with 1 always meaning best. Rank-based
    fusion ignores these, but weighted fusion needs them on one scale.
    """
    rows = list(rows)
    if not rows:
        return []

    raw = [float(row.get(score_key, 0.0) or 0.0) for row in rows]
    low, high = min(raw), max(raw)
    spread = high - low

    for position, (row, value) in enumerate(zip(rows, raw)):
        if spread == 0:
            row["score"] = 1.0
        elif higher_is_better:
            row["score"] = (value - low) / spread
        else:
            row["score"] = 1.0 - (value - low) / spread
        row["rank"] = position
    return rows


__all__ = ["LanceStore"]
