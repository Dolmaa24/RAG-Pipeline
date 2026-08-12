"""Text normalisation, and the typed pass over an extracted record.

The whitespace work is the unglamorous part that stops a lot of downstream
grief. Zero-width and bidi format characters (BOM, ZWSP, ZWNJ, LRM/RLM, soft
hyphen) are Unicode category ``Cf``: they survive NFKC, they are not matched by
``\\s``, and they silently poison scraped text — two strings that render
identically compare unequal, and a dedupe key built from one of them misses.

On top of that, :func:`normalize_record` gives each field the treatment its
*name* implies: a price becomes an exact Decimal string with a currency, a date
becomes ISO 8601 alongside the original wording, a URL is made absolute. The
original value is never thrown away.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

from observability import get_logger
from urls import resolve

from .dates import normalize_date_field
from .numbers import normalize_money, parse_int, parse_number

log = get_logger("normalize.text")

_WHITESPACE = re.compile(r"[^\S\n]+")   # runs of spaces/tabs, newlines preserved
_BLANK_LINES = re.compile(r"\n{3,}")
_KEEP_CONTROLS = {"\n", "\r", "\t"}

#: Field-name patterns that imply a typed value.
_MONEY_FIELD = re.compile(r"(price|cost|salary|amount|fee|revenue|total|budget|wage)", re.I)
_DATE_FIELD = re.compile(r"(date|published|updated|modified|created|deadline|expires|_at$)", re.I)
_URL_FIELD = re.compile(r"(url|link|href|image|logo|thumbnail|photo|avatar)s?$", re.I)
_INT_FIELD = re.compile(r"(count|_count$|quantity|qty|pages?|views?|likes?|year)", re.I)
_FLOAT_FIELD = re.compile(r"(rating|score|percent|ratio|average|latitude|longitude)", re.I)


def strip_invisibles(text: str) -> str:
    """Remove format and control characters, keeping real newlines and tabs."""
    return "".join(
        ch for ch in text
        if ch in _KEEP_CONTROLS or unicodedata.category(ch) not in ("Cf", "Cc")
    )


def clean_text(value: str, *, collapse_newlines: bool = True) -> str:
    """NFKC-normalise, strip invisibles, collapse runs of whitespace."""
    cleaned = unicodedata.normalize("NFKC", value)
    cleaned = strip_invisibles(cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned)
    if collapse_newlines:
        cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()


def normalize_value(value: Any) -> Any:
    """Recursively clean strings inside dicts, lists and scalars."""
    if isinstance(value, str):
        return clean_text(value) or None
    if isinstance(value, dict):
        return {key: normalize_value(entry) for key, entry in value.items()}
    if isinstance(value, list):
        cleaned = [normalize_value(entry) for entry in value]
        return [entry for entry in cleaned if entry is not None]
    return value


def normalize_record(
    record: dict,
    *,
    base_url: str = "",
    dayfirst: bool = False,
    typed: bool = True,
) -> dict:
    """Clean every value, then add typed companions for the fields that warrant one.

    Typed values land in sibling keys (``price_amount``, ``date_iso``) rather
    than replacing the original. Two reasons: the raw string is the evidence,
    and a caller who wanted the string they saw on the page should still get it.
    """
    cleaned = {key: normalize_value(value) for key, value in (record or {}).items()}
    if not typed:
        return cleaned

    additions: dict[str, Any] = {}
    for key, value in list(cleaned.items()):
        if not isinstance(value, str) or not value:
            continue

        if _MONEY_FIELD.search(key):
            money = normalize_money(value)
            if money["amount"] is not None:
                additions[f"{key}_amount"] = money["amount"]
                if money["currency"]:
                    additions.setdefault(f"{key}_currency", money["currency"])

        elif _DATE_FIELD.search(key):
            parsed = normalize_date_field(value, dayfirst=dayfirst)
            if parsed["iso"]:
                additions[f"{key}_iso"] = parsed["iso"]
                if parsed["ambiguous"]:
                    additions[f"{key}_ambiguous"] = True

        elif _URL_FIELD.search(key) and base_url and not value.startswith(("http", "data:")):
            absolute = resolve(base_url, value)
            if absolute:
                cleaned[key] = absolute

        elif _INT_FIELD.search(key):
            number = parse_int(value)
            if number is not None:
                additions[f"{key}_value"] = number

        elif _FLOAT_FIELD.search(key):
            number = parse_number(value)
            if number is not None:
                additions[f"{key}_value"] = number

    cleaned.update(additions)
    return cleaned


def summarize(text: str, max_chars: int = 280) -> Optional[str]:
    """First sentences up to ``max_chars`` — a cheap preview, no model needed."""
    if not text:
        return None
    cleaned = clean_text(text, collapse_newlines=True).replace("\n", " ")
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars]
    boundary = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[: boundary + 1] if boundary > max_chars * 0.5 else cut.rsplit(" ", 1)[0]) + "…"


__all__ = [
    "clean_text",
    "normalize_record",
    "normalize_value",
    "strip_invisibles",
    "summarize",
]
