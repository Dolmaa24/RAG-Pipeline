"""Date parsing that keeps the original string.

Extracted dates arrive in every format a publisher has ever used: ISO 8601 from
JSON-LD, ``March 3, 2026`` from an article byline, ``03/04/2026`` from a table
where nobody recorded whether it is March or April.

Two rules follow from that:

* **Never discard the original.** The parsed value goes in a new field and the
  string the page actually said stays put. When a date turns out wrong, the
  only way to find out why is to see what was parsed.
* **Never guess the ambiguous ones silently.** ``03/04/2026`` is parsed under
  the stated ``dayfirst`` convention and flagged as ambiguous, so a downstream
  consumer can decide whether that matters.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Optional

#: Formats tried in order. ISO first because it is unambiguous and by far the
#: most common in structured markup.
_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
    "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
    "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y",
    "%a, %d %b %Y %H:%M:%S %z",  # RFC 822, as used by RSS
    "%Y-%m-%dT%H:%M",
    "%B %Y", "%b %Y", "%Y",
)

_AMBIGUOUS = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\s*$")
_ISO_LIKE = re.compile(r"\d{4}-\d{2}-\d{2}")
_RELATIVE = re.compile(
    r"^\s*(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago\s*$", re.IGNORECASE
)
_UNIT_DAYS = {"second": 1 / 86400, "minute": 1 / 1440, "hour": 1 / 24,
              "day": 1, "week": 7, "month": 30.44, "year": 365.25}


def parse_date(value: str, *, dayfirst: bool = False) -> Optional[datetime]:
    """Parse a date string into a timezone-aware UTC datetime, or None."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None

    relative = _parse_relative(text)
    if relative is not None:
        return relative

    # Trim a trailing timezone name that strptime cannot read ("... GMT").
    text = re.sub(r"\s+\([A-Z]{2,5}\)\s*$", "", text)
    normalized = text.replace("Z", "+0000") if text.endswith("Z") else text
    normalized = re.sub(r"([+-]\d{2}):(\d{2})$", r"\1\2", normalized)

    for fmt in _FORMATS:
        try:
            parsed = datetime.strptime(normalized, fmt)
        except ValueError:
            continue
        return _as_utc(_apply_ambiguity(parsed, text, dayfirst))

    try:
        return _as_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        pass

    # Last resort: dateutil, when it happens to be installed (it is, as a
    # pandas dependency, but the pipeline does not require it).
    try:
        from dateutil import parser as dateutil_parser

        return _as_utc(dateutil_parser.parse(text, dayfirst=dayfirst, fuzzy=True))
    except Exception:
        return None


def _parse_relative(text: str) -> Optional[datetime]:
    """"3 days ago" — common on forums and comment sections."""
    from datetime import timedelta

    match = _RELATIVE.match(text)
    if not match:
        return None
    amount, unit = int(match.group(1)), match.group(2).lower()
    return datetime.now(timezone.utc) - timedelta(days=amount * _UNIT_DAYS[unit])


def _apply_ambiguity(parsed: datetime, original: str, dayfirst: bool) -> datetime:
    """Re-read a d/m/y-style string under the requested convention."""
    match = _AMBIGUOUS.match(original)
    if not match:
        return parsed
    first, second, year = (int(part) for part in match.groups())
    if first > 12 or second > 12:
        return parsed  # only one reading is possible; strptime already got it
    if year < 100:
        year += 2000 if year < 70 else 1900
    day, month = (first, second) if dayfirst else (second, first)
    try:
        return datetime(year, month, day)
    except ValueError:
        return parsed


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def is_ambiguous(value: str) -> bool:
    """True when the string could be read as either d/m/y or m/d/y."""
    match = _AMBIGUOUS.match(value or "")
    if not match:
        return False
    first, second, _ = (int(part) for part in match.groups())
    return first <= 12 and second <= 12 and first != second


def normalize_date_field(value: str, *, dayfirst: bool = False) -> dict:
    """``{"raw", "iso", "date", "ambiguous"}`` — the original always survives."""
    parsed = parse_date(value, dayfirst=dayfirst)
    return {
        "raw": value,
        "iso": parsed.isoformat() if parsed else None,
        "date": parsed.date().isoformat() if parsed else None,
        "ambiguous": is_ambiguous(value),
    }


def today() -> date:
    return datetime.now(timezone.utc).date()


__all__ = ["is_ambiguous", "normalize_date_field", "parse_date", "today"]
