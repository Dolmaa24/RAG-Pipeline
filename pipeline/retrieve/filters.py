"""Metadata filters, and the one place a filter predicate is built.

Filters arrive from two directions: a caller who knows exactly what they want,
and a language model that read the question and guessed. The second is why this
module validates rather than formats — a model asked for a filter will happily
invent a field name, and a predicate naming a column that does not exist is a
query that fails or, worse, one that quietly matches nothing.

So: field names are checked against the store's schema and unknown ones are
dropped with a warning, values are escaped, and dates are normalised through the
pipeline's existing parser before they reach a predicate.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from observability import get_logger
from pipeline.store.schema import FILTER_FIELDS

log = get_logger("retrieve.filters")

#: Fields taking a set of allowed values.
_SET_FIELDS = (
    "doc_type",
    "department",
    "author",
    "region",
    "permission_level",
    "language",
    "source",
)


def quote_literal(value: str) -> str:
    """A SQL string literal. Single quotes are doubled, which is the escape."""
    return "'" + str(value).replace("'", "''") + "'"


class MetadataFilter(BaseModel):
    """What to narrow the search to before it runs.

    Every field is optional and ``None`` means "do not constrain this". An
    *empty* value in the store means the document never recorded that field, and
    such a document will not match a filter on it — which is the safe direction
    for ``permission_level`` and the surprising one for ``department``, where a
    crawled page simply has none.
    """

    doc_type: Optional[list[str]] = None
    department: Optional[list[str]] = None
    author: Optional[list[str]] = None
    region: Optional[list[str]] = None
    permission_level: Optional[list[str]] = None
    language: Optional[list[str]] = None
    source: Optional[list[str]] = None
    #: ISO 8601, inclusive on both ends.
    date_from: Optional[str] = None
    date_to: Optional[str] = None

    def is_empty(self) -> bool:
        return not any(
            getattr(self, name) for name in (*_SET_FIELDS, "date_from", "date_to")
        )

    def compile(self) -> Optional[str]:
        """The SQL predicate, or ``None`` when nothing is constrained."""
        clauses: list[str] = []

        for field in _SET_FIELDS:
            values = getattr(self, field)
            if not values:
                continue
            if field not in FILTER_FIELDS:  # pragma: no cover - guarded by the schema
                log.warning("retrieve.filters.unknown_field", field=field)
                continue
            cleaned = [str(v).strip() for v in values if str(v).strip()]
            if not cleaned:
                continue
            if len(cleaned) == 1:
                clauses.append(f"{field} = {quote_literal(cleaned[0])}")
            else:
                clauses.append(f"{field} IN ({', '.join(quote_literal(v) for v in cleaned)})")

        start = _iso_date(self.date_from)
        if start:
            clauses.append(f"date >= {quote_literal(start)}")
        end = _iso_date(self.date_to)
        if end:
            clauses.append(f"date <= {quote_literal(end)}")

        return " AND ".join(clauses) if clauses else None

    @classmethod
    def from_model(cls, raw: dict) -> "MetadataFilter":
        """Build from a model's guess, keeping only fields we recognise.

        A hallucinated ``"sensitivity"`` or ``"team"`` is discarded here rather
        than becoming a predicate that fails at query time.
        """
        if not isinstance(raw, dict):
            return cls()

        known = set(cls.model_fields)
        kept: dict = {}
        for key, value in raw.items():
            if key not in known:
                log.info("retrieve.filters.dropped_unknown_field", field=key)
                continue
            if value in (None, "", [], {}):
                continue
            if key in ("date_from", "date_to"):
                kept[key] = str(value)
            elif isinstance(value, str):
                kept[key] = [value]
            elif isinstance(value, list):
                kept[key] = [str(v) for v in value if str(v).strip()]
        return cls(**kept)


def _iso_date(value: Optional[str]) -> Optional[str]:
    """Normalise whatever the caller or the model wrote into ``YYYY-MM-DD``."""
    if not value or not str(value).strip():
        return None

    text = str(value).strip()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return text

    from pipeline.normalize.dates import parse_date

    parsed = parse_date(text)
    if parsed is None:
        log.warning("retrieve.filters.unparsable_date", value=text[:40])
        return None
    return parsed.date().isoformat()


__all__ = ["MetadataFilter", "quote_literal"]
