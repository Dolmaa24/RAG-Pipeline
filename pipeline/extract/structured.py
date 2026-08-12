"""Tier 1: read the structure the publisher already published.

Most commerce, article, recipe, event and job pages carry schema.org markup so
that search engines can understand them. That markup is the publisher's own
description of the page — the canonical price, the real author, the exact
publication timestamp — and it is sitting in the HTML, unread, while the
pipeline pays a model to infer the same facts from rendered text.

Reading it is roughly five milliseconds against thirty seconds, and the answer
is *more* accurate, because it is an assertion rather than an inference.

Sources harvested, in descending order of trust:

1. **JSON-LD** — ``<script type="application/ld+json">``. Explicit, typed, and
   by far the most common.
2. **Microdata / RDFa** — inline ``itemprop`` attributes.
3. **Framework state** — ``__NEXT_DATA__``, ``__NUXT__``,
   ``__INITIAL_STATE__``. The props the page rendered itself from, which on a
   React or Vue site is the entire record.
4. **OpenGraph / Twitter cards** — thin, but almost universal, and enough on
   its own for a title/description/image schema.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

from observability import get_logger
from urls import resolve

log = get_logger("extract.structured")

_JSONLD_RE = re.compile(
    r'<script[^>]+type\s*=\s*["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
_NEXT_DATA_RE = re.compile(
    r'<script[^>]+id\s*=\s*["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
_STATE_RE = re.compile(
    r"window\.(__NUXT__|__INITIAL_STATE__|__APOLLO_STATE__|__PRELOADED_STATE__)\s*=\s*"
    r"(\{.*?\})\s*;?\s*(?:</script>|\n)",
    re.IGNORECASE | re.DOTALL,
)
_META_RE = re.compile(
    r"<meta\s+[^>]*?(?:property|name)\s*=\s*[\"'](og:[^\"']+|twitter:[^\"']+|article:[^\"']+)[\"']"
    r"[^>]*?content\s*=\s*[\"']([^\"']*)[\"']",
    re.IGNORECASE,
)
_META_REVERSED = re.compile(
    r"<meta\s+[^>]*?content\s*=\s*[\"']([^\"']*)[\"'][^>]*?(?:property|name)\s*=\s*"
    r"[\"'](og:[^\"']+|twitter:[^\"']+|article:[^\"']+)[\"']",
    re.IGNORECASE,
)

#: JSON payloads over this are frameworks dumping their entire store. Parsing
#: them is slow and the useful part is never that deep.
_MAX_EMBEDDED_BYTES = 4 * 1024 * 1024


def harvest(html: str, base_url: str = "") -> dict[str, Any]:
    """Collect every structured payload in ``html``. Never raises."""
    found: dict[str, Any] = {}

    jsonld = _harvest_jsonld(html)
    if jsonld:
        found["jsonld"] = jsonld

    opengraph = _harvest_meta(html)
    if opengraph:
        found["opengraph"] = opengraph

    next_data = _harvest_next_data(html)
    if next_data:
        found["next_data"] = next_data

    state = _harvest_state(html)
    if state:
        found["app_state"] = state

    micro = _harvest_microdata(html, base_url)
    if micro:
        found.update(micro)

    if found:
        log.debug("structured.harvested", sources=list(found), url=base_url)
    return found


def _harvest_jsonld(html: str) -> list[dict]:
    blocks: list[dict] = []
    for raw in _JSONLD_RE.findall(html):
        text = raw.strip()
        if not text or len(text) > _MAX_EMBEDDED_BYTES:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Trailing commas and HTML comments inside JSON-LD are common
            # enough to be worth one repair attempt.
            try:
                parsed = json.loads(_repair_json(text))
            except json.JSONDecodeError:
                continue
        blocks.extend(_flatten_graph(parsed))
    return blocks


def _flatten_graph(node: Any) -> list[dict]:
    """Unwrap ``@graph`` containers and lists into a flat list of typed nodes."""
    if isinstance(node, list):
        out: list[dict] = []
        for entry in node:
            out.extend(_flatten_graph(entry))
        return out
    if isinstance(node, dict):
        if "@graph" in node:
            nested = _flatten_graph(node["@graph"])
            rest = {k: v for k, v in node.items() if k != "@graph"}
            return ([rest] if len(rest) > 1 else []) + nested
        return [node]
    return []


def _repair_json(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r",\s*([}\]])", r"\1", text)  # trailing commas
    return text.strip()


def _harvest_meta(html: str) -> dict[str, str]:
    """OpenGraph, Twitter card, and article:* meta tags.

    Two regexes because attribute order in ``<meta>`` is not fixed and half the
    web writes ``content`` before ``property``.
    """
    out: dict[str, str] = {}
    for key, value in _META_RE.findall(html):
        out.setdefault(key.lower(), _unescape(value))
    for value, key in _META_REVERSED.findall(html):
        out.setdefault(key.lower(), _unescape(value))
    return out


def _harvest_next_data(html: str) -> Optional[dict]:
    match = _NEXT_DATA_RE.search(html)
    if not match:
        return None
    raw = match.group(1).strip()
    if len(raw) > _MAX_EMBEDDED_BYTES:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    # The record lives under props.pageProps on essentially every Next.js site;
    # returning that directly saves every consumer the same two lookups.
    props = data.get("props", {}) if isinstance(data, dict) else {}
    page_props = props.get("pageProps") if isinstance(props, dict) else None
    return page_props if isinstance(page_props, dict) and page_props else data


def _harvest_state(html: str) -> Optional[dict]:
    match = _STATE_RE.search(html)
    if not match:
        return None
    raw = match.group(2).strip()
    if len(raw) > _MAX_EMBEDDED_BYTES:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _harvest_microdata(html: str, base_url: str) -> dict[str, Any]:
    """Microdata and RDFa, via extruct when it is installed."""
    if "itemscope" not in html and "typeof=" not in html and "vocab=" not in html:
        return {}
    try:
        import extruct
    except ImportError:
        return {}
    try:
        data = extruct.extract(
            html,
            base_url=base_url or None,
            syntaxes=["microdata", "rdfa"],
            uniform=True,
        )
    except Exception as exc:
        log.debug("structured.extruct_failed", error=repr(exc))
        return {}
    return {key: value for key, value in data.items() if value}


def _unescape(value: str) -> str:
    import html as html_module

    return html_module.unescape(value).strip()


# --------------------------------------------------------------------------- #
# Mapping harvested structure onto the caller's schema
# --------------------------------------------------------------------------- #

#: Field name → the keys that commonly hold that value in schema.org, OpenGraph
#: and framework state. Ordered: earlier entries are preferred.
FIELD_SYNONYMS: dict[str, tuple[str, ...]] = {
    "title": ("headline", "name", "title", "og:title", "twitter:title", "alternativeHeadline"),
    "name": ("name", "headline", "title", "og:title"),
    "heading": ("headline", "name", "title", "og:title"),
    "description": ("description", "og:description", "twitter:description", "abstract", "summary"),
    "summary": ("description", "abstract", "og:description", "summary"),
    "content": ("articleBody", "text", "description", "content"),
    "body": ("articleBody", "text", "content"),
    "text": ("text", "articleBody", "description"),
    "author": ("author", "creator", "byline", "article:author", "twitter:creator", "publisher"),
    "byline": ("author", "creator", "byline"),
    "publisher": ("publisher", "sourceOrganization", "og:site_name", "brand"),
    "brand": ("brand", "manufacturer", "publisher"),
    "date": ("datePublished", "dateCreated", "uploadDate", "article:published_time", "startDate"),
    "published": ("datePublished", "dateCreated", "article:published_time", "uploadDate"),
    "published_date": ("datePublished", "dateCreated", "article:published_time"),
    "date_published": ("datePublished", "dateCreated", "article:published_time"),
    "updated": ("dateModified", "article:modified_time"),
    "modified": ("dateModified", "article:modified_time"),
    "price": ("price", "lowPrice", "offers.price", "offers.lowPrice", "product:price:amount"),
    "currency": ("priceCurrency", "offers.priceCurrency", "product:price:currency"),
    "availability": ("availability", "offers.availability", "itemCondition"),
    "sku": ("sku", "mpn", "productID", "gtin13", "gtin"),
    "rating": ("ratingValue", "aggregateRating.ratingValue", "reviewRating.ratingValue"),
    "review_count": ("reviewCount", "aggregateRating.reviewCount", "ratingCount"),
    "image": ("image", "og:image", "twitter:image", "thumbnailUrl", "contentUrl", "logo"),
    "images": ("image", "og:image", "thumbnailUrl"),
    "url": ("url", "og:url", "mainEntityOfPage", "@id"),
    "category": ("category", "articleSection", "genre", "about"),
    "tags": ("keywords", "article:tag", "about", "genre"),
    "keywords": ("keywords", "article:tag"),
    "language": ("inLanguage", "og:locale"),
    "duration": ("duration", "timeRequired", "cookTime", "totalTime"),
    "location": ("location", "contentLocation", "address", "areaServed"),
    "address": ("address", "location"),
    "email": ("email",),
    "phone": ("telephone", "phone"),
    "quote": ("text", "articleBody", "description"),
    "ingredients": ("recipeIngredient", "ingredients"),
    "instructions": ("recipeInstructions", "instructions", "step"),
    "salary": ("baseSalary", "estimatedSalary", "salaryCurrency"),
    "company": ("hiringOrganization", "employer", "publisher", "brand"),
    "job_title": ("title", "name"),
    "event_date": ("startDate", "datePublished"),
    "venue": ("location", "place"),
}

#: schema.org @type values worth preferring when several JSON-LD nodes exist.
_PREFERRED_TYPES = (
    "Product", "Article", "NewsArticle", "BlogPosting", "Recipe", "JobPosting",
    "Event", "Course", "Book", "Movie", "Review", "LocalBusiness", "Organization",
    "Person", "VideoObject", "PodcastEpisode", "FAQPage", "QAPage", "SoftwareApplication",
)


def map_to_schema(structured: dict[str, Any], schema: dict) -> tuple[dict[str, Any], float]:
    """Fill ``schema``'s fields from harvested structure.

    Returns the mapped record and the fill rate — the share of requested fields
    that got a non-empty value. The caller uses the fill rate to decide whether
    this tier answered well enough or the cascade should fall through.
    """
    if not structured or not schema:
        return {}, 0.0

    flat = _flatten_candidates(structured)
    # The handler's own reading of the format is real structured data, but it
    # is coarser than the document's embedded markup: a page's <title> is
    # "Blue Widget | ACME Store" where its JSON-LD `name` is "Blue Widget".
    # So it is a fallback consulted only after the markup has had every chance.
    fallback = dict(structured.get("handler_metadata") or {})
    if not flat and not fallback:
        return {}, 0.0

    result: dict[str, Any] = {}
    for field, spec in schema.items():
        result[field] = _find_value(field, spec, flat, fallback)

    filled = sum(1 for v in result.values() if _is_filled(v))
    return result, filled / len(schema) if schema else 0.0


def _flatten_candidates(structured: dict[str, Any]) -> dict[str, Any]:
    """One flat ``key -> value`` map over every harvested source.

    Keys are both the bare property name and its dotted path, so a schema field
    can match ``price`` whether it sits at the top level or under ``offers``.
    Higher-trust sources are merged last only where they add keys, so JSON-LD
    never gets overwritten by an OpenGraph tag saying something vaguer.
    """
    flat: dict[str, Any] = {}

    for node in _ranked_jsonld(structured.get("jsonld") or []):
        _flatten_into(node, flat, prefix="")

    for key in ("microdata", "rdfa"):
        for node in structured.get(key) or []:
            if isinstance(node, dict):
                _flatten_into(node, flat, prefix="")

    for source in ("next_data", "app_state"):
        payload = structured.get(source)
        if isinstance(payload, dict):
            _flatten_into(payload, flat, prefix="", max_depth=4)

    for key, value in (structured.get("opengraph") or {}).items():
        flat.setdefault(key, value)
        bare = key.split(":", 1)[-1]
        flat.setdefault(bare, value)


    # Rows are structure too. A CSV whose columns are named `title` and `price`
    # is the most literal tier-1 hit there is, and a two-column HTML or PDF
    # table is a key-value list wearing a grid costume.
    _flatten_rows(structured, flat)

    payload = structured.get("data")
    if isinstance(payload, dict):
        _flatten_into(payload, flat, prefix="", max_depth=4)
    elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
        _flatten_into(payload[0], flat, prefix="", max_depth=4)
        flat.setdefault("_records", payload)

    return flat


#: Handler outputs that are lists of *content* records — a schema asking for a
#: title wants the first entry's title.
#:
#: Deliberately excludes ``archive_members`` and ``sitemap_urls``: those are
#: inventories, not content. A ZIP member called ``a.csv`` is a filename, and
#: letting it answer a request for "title" is exactly the kind of confident
#: wrong answer this whole tier exists to avoid.
_RECORD_LISTS = ("feed_entries", "records", "items")


def _flatten_rows(structured: dict[str, Any], flat: dict[str, Any]) -> None:
    """Expose parsed rows as candidates: first value scalar, column as list."""
    for key in _RECORD_LISTS:
        entries = structured.get(key)
        if not isinstance(entries, list) or not entries:
            continue
        flat.setdefault("_records", entries)
        if not isinstance(entries[0], dict):
            continue
        columns: dict[str, list[Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for name, value in entry.items():
                if _is_filled(value):
                    columns.setdefault(name, []).append(value)
        for name, values in columns.items():
            flat.setdefault(name, values[0])
            flat.setdefault(f"{name}__column", values)

    table = structured.get("table")
    if isinstance(table, dict):
        records = table.get("records") or []
        if records:
            flat.setdefault("_records", records)
            for column in table.get("headers") or []:
                values = [
                    record.get(column) for record in records if _is_filled(record.get(column))
                ]
                if not values:
                    continue
                # Scalar consumers take the first row; list consumers get the
                # whole column. `_coerce` picks whichever the schema asked for.
                flat.setdefault(column, values[0])
                flat.setdefault(f"{column}__column", values)

    for grid in structured.get("tables") or []:
        headers = grid.get("headers") or []
        rows = grid.get("rows") or []
        if len(headers) == 2 and rows:
            # `Colour | Blue` — a label/value list, not a data grid.
            for row in rows:
                if len(row) >= 2 and row[0]:
                    flat.setdefault(str(row[0]).strip(), row[1])
        elif headers and rows:
            for index, column in enumerate(headers):
                values = [row[index] for row in rows if len(row) > index and _is_filled(row[index])]
                if values:
                    flat.setdefault(column, values[0])
                    flat.setdefault(f"{column}__column", values)


def _ranked_jsonld(nodes: Iterable[Any]) -> list[dict]:
    """Put the node describing the page's subject first.

    A page routinely carries a BreadcrumbList, a WebSite, an Organization and
    the actual Product. Reading them in document order gives you the site's
    name where you asked for the product's.
    """
    typed: list[tuple[int, dict]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        types = node.get("@type") or node.get("type") or ""
        names = [types] if isinstance(types, str) else [t for t in types if isinstance(t, str)]
        rank = len(_PREFERRED_TYPES)
        for name in names:
            short = name.rsplit("/", 1)[-1]
            if short in _PREFERRED_TYPES:
                rank = min(rank, _PREFERRED_TYPES.index(short))
        typed.append((rank, node))
    typed.sort(key=lambda pair: pair[0])
    return [node for _, node in typed]


def _flatten_into(
    node: Any,
    out: dict[str, Any],
    prefix: str,
    depth: int = 0,
    max_depth: int = 6,
) -> None:
    if depth > max_depth or not isinstance(node, dict):
        return
    for key, value in node.items():
        if key.startswith("@") and key not in ("@id", "@type"):
            continue
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, (str, int, float, bool)):
            out.setdefault(path, value)
            out.setdefault(key, value)
        elif isinstance(value, list):
            scalars = [v for v in value if isinstance(v, (str, int, float, bool))]
            if scalars:
                out.setdefault(path, scalars if len(scalars) > 1 else scalars[0])
                out.setdefault(key, scalars if len(scalars) > 1 else scalars[0])
            for entry in value[:5]:
                if isinstance(entry, dict):
                    _flatten_into(entry, out, path, depth + 1, max_depth)
        elif isinstance(value, dict):
            # A nested node with a `name`/`url` is usually a reference; expose
            # its label under the parent key so `author` yields "Jane Doe"
            # rather than a dict nobody downstream can render.
            label = value.get("name") or value.get("headline") or value.get("url")
            if isinstance(label, str):
                out.setdefault(path, label)
                out.setdefault(key, label)
            _flatten_into(value, out, path, depth + 1, max_depth)


def _find_value(
    field: str,
    spec: Any,
    flat: dict[str, Any],
    fallback: Optional[dict[str, Any]] = None,
) -> Any:
    """Best candidate for one schema field, highest-trust source first."""
    wants_list = _wants_list(spec)
    normalized = re.sub(r"[^a-z0-9]+", "_", field.lower()).strip("_")

    candidates: list[str] = [field, normalized]
    candidates.extend(FIELD_SYNONYMS.get(normalized, ()))
    # camelCase spelling of the field, which is what schema.org uses.
    parts = normalized.split("_")
    if len(parts) > 1:
        candidates.append(parts[0] + "".join(p.title() for p in parts[1:]))

    for source in (flat, fallback or {}):
        if not source:
            continue
        for candidate in candidates:
            # A list-valued field prefers the whole column over its first cell.
            if wants_list and f"{candidate}__column" in source:
                value = _coerce(source[f"{candidate}__column"], True)
                if _is_filled(value):
                    return value
            for key in (candidate, candidate.lower()):
                if key in source:
                    value = _coerce(source[key], wants_list)
                    if _is_filled(value):
                        return value

        # A key that ends with the field name, e.g. "offers.price".
        suffix = "." + normalized
        for key, value in source.items():
            if key.lower().endswith(suffix):
                coerced = _coerce(value, wants_list)
                if _is_filled(coerced):
                    return coerced

    return [] if wants_list else None


def _wants_list(spec: Any) -> bool:
    if isinstance(spec, list):
        return True
    if isinstance(spec, str):
        lowered = spec.lower()
        return "list" in lowered or "array" in lowered or lowered.endswith("[]")
    if isinstance(spec, dict):
        return spec.get("type") == "array"
    return False


def _coerce(value: Any, wants_list: bool) -> Any:
    if wants_list:
        if isinstance(value, list):
            return [v for v in value if _is_filled(v)]
        if isinstance(value, str) and "," in value:
            return [part.strip() for part in value.split(",") if part.strip()]
        return [value] if _is_filled(value) else []
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def absolutize_images(record: dict, base_url: str) -> dict:
    """Turn relative image paths into absolute URLs. Cheap, and always wanted."""
    for key in ("image", "images", "thumbnail", "logo"):
        value = record.get(key)
        if isinstance(value, str) and value and not value.startswith(("http", "data:")):
            record[key] = resolve(base_url, value)
        elif isinstance(value, list):
            record[key] = [
                resolve(base_url, v) if isinstance(v, str) and not v.startswith(("http", "data:")) else v
                for v in value
            ]
    return record


__all__ = ["FIELD_SYNONYMS", "absolutize_images", "harvest", "map_to_schema"]
