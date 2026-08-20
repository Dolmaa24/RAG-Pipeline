"""The stored chunk: its columns, and which of them can be filtered on.

One module owns this list because three others depend on agreeing with it — the
store creates a scalar index per filter column, the filter compiler validates
field names against it, and the indexer populates it. When they disagree the
symptom is a query that silently returns nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cost at type-check time only
    import pyarrow as pa

#: Columns a query may filter on. Everything here gets a scalar index, which is
#: what makes ``prefilter=True`` a lookup rather than a table scan.
FILTER_FIELDS: tuple[str, ...] = (
    "doc_type",
    "department",
    "date",
    "author",
    "region",
    "permission_level",
    "language",
    "source",
)

#: Carried for traceability, not filtered on.
PROVENANCE_FIELDS: tuple[str, ...] = (
    "content_hash",
    "page_no",
    "section_name",
    "chunk_strategy",
    "embedding_model",
    "extraction_tier",
    "indexed_at",
)

#: The column full-text search indexes.
TEXT_FIELD = "document"
VECTOR_FIELD = "vector"
ID_FIELD = "id"


def arrow_schema(dimension: int) -> "pa.Schema":
    """The table schema for a given embedding width.

    Dates are ISO 8601 strings rather than a timestamp type. They sort and
    compare correctly as strings, they survive a round trip through JSON without
    a timezone argument, and ``date >= '2026-01-01'`` is a predicate anyone can
    read in a log line.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field(ID_FIELD, pa.string()),
            pa.field(VECTOR_FIELD, pa.list_(pa.float32(), dimension)),
            pa.field(TEXT_FIELD, pa.string()),
            # --- filterable ---
            pa.field("source", pa.string()),
            pa.field("doc_type", pa.string()),
            pa.field("department", pa.string()),
            pa.field("date", pa.string()),
            pa.field("author", pa.string()),
            pa.field("region", pa.string()),
            pa.field("permission_level", pa.string()),
            pa.field("language", pa.string()),
            # --- provenance ---
            pa.field("content_hash", pa.string()),
            pa.field("page_no", pa.int32()),
            pa.field("section_name", pa.string()),
            pa.field("chunk_strategy", pa.string()),
            pa.field("embedding_model", pa.string()),
            pa.field("extraction_tier", pa.int32()),
            pa.field("indexed_at", pa.string()),
        ]
    )


__all__ = [
    "FILTER_FIELDS",
    "ID_FIELD",
    "PROVENANCE_FIELDS",
    "TEXT_FIELD",
    "VECTOR_FIELD",
    "arrow_schema",
]
