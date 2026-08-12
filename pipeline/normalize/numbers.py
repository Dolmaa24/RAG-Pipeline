"""Numbers and money.

Money is held as :class:`~decimal.Decimal` and serialised as a **string**, never
as a float. ``0.1 + 0.2`` is not ``0.3`` in binary floating point, and a price
that round-trips through JSON as ``19.989999999999998`` is a bug that surfaces
in an invoice months later. A string round-trips exactly.

Thousands separators are the other trap. ``1.234,56`` is one thousand two
hundred in most of Europe and one-point-two in the US convention, and the only
way to tell is the *pattern* of the separators rather than the separators
themselves. That inference is done explicitly below rather than by stripping
punctuation and hoping.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional

#: Symbol → ISO 4217. Only unambiguous symbols; "$" alone is not one, so it maps
#: to USD as the common reading and the raw string is always kept alongside.
CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD", "US$": "USD", "£": "GBP", "€": "EUR", "¥": "JPY", "₹": "INR",
    "₨": "INR", "R$": "BRL", "₩": "KRW", "₽": "RUB", "₺": "TRY", "₪": "ILS",
    "CHF": "CHF", "kr": "SEK", "zł": "PLN", "₫": "VND", "฿": "THB", "₦": "NGN",
    "C$": "CAD", "A$": "AUD", "NZ$": "NZD", "HK$": "HKD", "S$": "SGD",
}

_CURRENCY_CODE = re.compile(r"\b([A-Z]{3})\b")
_NUMBER = re.compile(r"[-+]?[\d][\d.,\s ']*\d|[-+]?\d")
_MULTIPLIER = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "mn": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "bn": 1_000_000_000, "billion": 1_000_000_000,
    "t": 1_000_000_000_000, "trillion": 1_000_000_000_000,
    "lakh": 100_000, "crore": 10_000_000,
}
_MULTIPLIER_RE = re.compile(
    r"(\d)\s*(" + "|".join(sorted(_MULTIPLIER, key=len, reverse=True)) + r")\b", re.IGNORECASE
)


def parse_decimal(value: str) -> Optional[Decimal]:
    """Parse a number out of free text, inferring the separator convention."""
    if value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    multiplier = 1
    scaled = _MULTIPLIER_RE.search(text)
    if scaled:
        multiplier = _MULTIPLIER[scaled.group(2).lower()]

    match = _NUMBER.search(text.replace(" ", " "))
    if not match:
        return None

    raw = re.sub(r"[\s']", "", match.group(0))
    cleaned = _resolve_separators(raw)
    try:
        return Decimal(cleaned) * multiplier
    except InvalidOperation:
        return None


def _resolve_separators(raw: str) -> str:
    """Work out which of ``.`` and ``,`` is the decimal point."""
    has_dot = "." in raw
    has_comma = "," in raw

    if has_dot and has_comma:
        # Whichever appears last is the decimal separator.
        return (
            raw.replace(",", "") if raw.rfind(".") > raw.rfind(",")
            else raw.replace(".", "").replace(",", ".")
        )

    if has_comma:
        parts = raw.split(",")
        # "1,234" and "1,234,567" are grouped thousands; "1,23" is a decimal.
        if len(parts) > 2 or (len(parts[-1]) == 3 and len(parts[0]) <= 3):
            return raw.replace(",", "")
        return raw.replace(",", ".")

    if has_dot:
        parts = raw.split(".")
        if len(parts) > 2 or (len(parts[-1]) == 3 and len(parts) == 2 and len(parts[0]) <= 3
                              and not parts[0].startswith("0")):
            # "1.234.567" or a lone "1.234" that is far more likely thousands.
            return raw.replace(".", "") if len(parts) > 2 else raw
        return raw

    return raw


def detect_currency(value: str) -> Optional[str]:
    """ISO 4217 code for the currency mentioned in ``value``, if any."""
    if not isinstance(value, str):
        return None
    code = _CURRENCY_CODE.search(value.upper())
    if code and code.group(1) in set(CURRENCY_SYMBOLS.values()) | {"CNY", "AED", "SAR", "ZAR", "MXN", "NPR"}:
        return code.group(1)
    # Longest symbol first, so "C$" is not read as "$".
    for symbol in sorted(CURRENCY_SYMBOLS, key=len, reverse=True):
        if symbol in value:
            return CURRENCY_SYMBOLS[symbol]
    return None


def normalize_money(value: str, *, default_currency: Optional[str] = None) -> dict:
    """``{"raw", "amount", "currency"}``, with ``amount`` as an exact string."""
    amount = parse_decimal(value)
    return {
        "raw": value,
        # str(Decimal) round-trips exactly; float(Decimal) does not.
        "amount": str(amount) if amount is not None else None,
        "currency": detect_currency(value) or default_currency,
    }


def parse_number(value: str) -> Optional[float]:
    """A plain float, for fields where exactness does not matter (ratings)."""
    parsed = parse_decimal(value)
    return float(parsed) if parsed is not None else None


def parse_int(value: str) -> Optional[int]:
    parsed = parse_decimal(value)
    if parsed is None:
        return None
    try:
        return int(parsed)
    except (ValueError, OverflowError):
        return None


__all__ = [
    "CURRENCY_SYMBOLS",
    "detect_currency",
    "normalize_money",
    "parse_decimal",
    "parse_int",
    "parse_number",
]
