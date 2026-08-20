"""Where each filter field comes from.

The filters a query can use are only as good as what populates them, and the
honest answer differs per field. Three groups:

* **Already known.** The resource kind came from magic-byte detection, the
  language from the preprocessing stage. These are free and always right.
* **Published by the document.** Author, title and date are in JSON-LD,
  OpenGraph and PDF metadata for a good fraction of real pages, and
  :func:`pipeline.extract.structured.harvest` already collected them for tier 1
  of the extraction cascade. Reading them again here costs nothing.
* **Supplied by whoever ingested it.** Department, region and permission level
  are properties of an organisation's filing system, not of a web page. Nothing
  can infer them, and nothing here tries — a permission level derived from a
  guess is worse than no permission level, because it looks like a control.

An unset field is empty, and an empty field does not match a filter on it. That
is the safe direction for permissions and the surprising one for department, so
it is written down here and in the README.
"""

from __future__ import annotations

from typing import Any, Optional

from observability import get_logger

log = get_logger("index.metadata")

#: JSON-LD and OpenGraph keys that carry a publication date, best first.
_DATE_KEYS = (
    "datePublished",
    "dateCreated",
    "article:published_time",
    "og:updated_time",
    "dateModified",
    "date",
)

_AUTHOR_KEYS = ("author", "creator", "article:author", "og:site_name", "publisher")


def derive(
    item: Any = None,
    *,
    supplied: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """The filter fields for one item, from every source that has them.

    Caller-supplied values win over anything derived: someone who states the
    department knows, and a document's own markup does not.
    """
    derived: dict[str, Any] = {}

    if item is not None:
        kind = getattr(item, "kind", None)
        if kind is not None:
            derived["doc_type"] = getattr(kind, "value", str(kind))

        structured = getattr(item, "structured", None) or {}
        metadata = getattr(item, "metadata", None) or {}

        author = _first(_AUTHOR_KEYS, structured, metadata)
        if author:
            derived["author"] = author

        raw_date = _first(_DATE_KEYS, structured, metadata)
        if raw_date:
            iso = _iso(raw_date)
            if iso:
                derived["date"] = iso

    for key, value in (supplied or {}).items():
        if value not in (None, ""):
            derived[key] = value

    return derived


def _first(keys: tuple[str, ...], *sources: dict) -> str:
    """The first non-empty value for any of these keys, at any nesting."""
    for key in keys:
        for source in sources:
            value = _search(source, key)
            if value:
                return value
    return ""


def _search(node: Any, key: str, depth: int = 0) -> str:
    """Find ``key`` in nested JSON-LD, which is nested arbitrarily deep."""
    if depth > 4:
        return ""

    if isinstance(node, dict):
        if key in node:
            return _scalar(node[key])
        for value in node.values():
            if isinstance(value, (dict, list)):
                found = _search(value, key, depth + 1)
                if found:
                    return found
    elif isinstance(node, list):
        for value in node[:20]:
            found = _search(value, key, depth + 1)
            if found:
                return found
    return ""


def _scalar(value: Any) -> str:
    """Flatten what a metadata field might hold into one string.

    ``author`` is a string in one document, ``{"@type": "Person", "name": ...}``
    in the next, and a list of either in the one after that.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("name", "@id", "url", "text"):
            if isinstance(value.get(key), str):
                return value[key].strip()
        return ""
    if isinstance(value, list) and value:
        return _scalar(value[0])
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _iso(value: str) -> str:
    """Normalise a published date to ``YYYY-MM-DD``, or drop it."""
    text = value.strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]

    from pipeline.normalize.dates import parse_date

    parsed = parse_date(text)
    if parsed is None:
        log.debug("index.metadata.unparsable_date", value=text[:40])
        return ""
    return parsed.date().isoformat()


__all__ = ["derive"]
