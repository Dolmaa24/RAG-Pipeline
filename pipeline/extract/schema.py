"""Turning a loose schema hint into something enforceable.

The API takes schemas in the friendly form people actually write::

    {"quote": "string", "author": "string", "tags": "list of strings"}

Which is a *hint*, not a contract. The original pipeline passed it into the
prompt and checked only that the reply was valid JSON — so a model returning
``{"quotation": "...", "writer": "..."}`` produced a corrupt Mongo document that
nothing could detect.

This module compiles that hint into a real JSON Schema, which buys two things:

* **Groq** can be handed it as ``response_format: {"type": "json_schema"}``, and
  the API then guarantees conformance — removing the whole class of
  "valid JSON, wrong fields" bugs rather than retrying past them.
* **Ollama** has no such guarantee (``format: "json"`` promises valid JSON and
  nothing about its shape), so its output is validated here instead, and a
  mismatch becomes a caught :class:`~errors.SchemaViolation` rather than a
  stored lie.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

from observability import get_logger

log = get_logger("extract.schema")

_LIST_HINT = re.compile(r"\b(list|array)\b", re.IGNORECASE)
_INT_HINT = re.compile(r"\b(int|integer|count|year)\b", re.IGNORECASE)
_NUMBER_HINT = re.compile(r"\b(number|float|decimal|price|amount|rating|score)\b", re.IGNORECASE)
_BOOL_HINT = re.compile(r"\b(bool|boolean|flag|yes/no)\b", re.IGNORECASE)
_OBJECT_HINT = re.compile(r"\b(object|dict|map|mapping)\b", re.IGNORECASE)


def _scalar_for(hint: str) -> dict:
    """JSON Schema for one scalar hint string.

    Every type is unioned with ``null``. A field that is genuinely absent from
    the page must have a way to say so — without it the model is pushed into
    inventing a plausible value, which is worse than an empty cell.
    """
    if _BOOL_HINT.search(hint):
        return {"type": ["boolean", "null"]}
    if _INT_HINT.search(hint):
        return {"type": ["integer", "null"]}
    if _NUMBER_HINT.search(hint):
        return {"type": ["number", "string", "null"]}
    return {"type": ["string", "null"]}


def _item_type_for(hint: str) -> dict:
    """Element schema for a list hint, e.g. "list of numbers"."""
    tail = re.sub(r"^.*\b(list|array)\b\s*(of)?\s*", "", hint, flags=re.IGNORECASE).strip()
    if not tail:
        return {"type": ["string", "number", "boolean", "null"]}
    if _OBJECT_HINT.search(tail):
        return {"type": "object"}
    singular = tail.rstrip("s")
    return _scalar_for(singular)


def compile_schema(hint: Any, *, name: str = "extraction") -> dict:
    """Compile a schema hint into a JSON Schema object.

    Accepts the friendly form, a nested version of it, or an already-valid JSON
    Schema (detected by a top-level ``type``/``properties``), which is passed
    through untouched.
    """
    if isinstance(hint, dict) and ("properties" in hint or hint.get("type") == "object"):
        return hint  # already a JSON Schema

    if not isinstance(hint, dict):
        raise ValueError("a schema hint must be an object mapping field names to types")

    properties: dict[str, Any] = {}
    for field, spec in hint.items():
        properties[field] = _compile_field(spec)

    return {
        "type": "object",
        "title": name,
        "properties": properties,
        # Every field is required so the model must at least acknowledge each
        # one; `null` is how it says "not on this page". A merely-optional field
        # gets silently dropped, and a dropped field is indistinguishable from
        # a field the extractor never looked for.
        "required": list(properties),
        "additionalProperties": False,
    }


def _compile_field(spec: Any) -> dict:
    if isinstance(spec, str):
        if _LIST_HINT.search(spec):
            return {"type": ["array", "null"], "items": _item_type_for(spec)}
        if _OBJECT_HINT.search(spec):
            return {"type": ["object", "null"]}
        return _scalar_for(spec)

    if isinstance(spec, dict):
        if "type" in spec:  # a hand-written JSON Schema fragment
            return spec
        nested = {key: _compile_field(value) for key, value in spec.items()}
        return {
            "type": ["object", "null"],
            "properties": nested,
            "additionalProperties": False,
        }

    if isinstance(spec, list):
        element = _compile_field(spec[0]) if spec else {"type": ["string", "null"]}
        return {"type": ["array", "null"], "items": element}

    return {"type": ["string", "null"]}


def validate(payload: Any, json_schema: dict) -> list[str]:
    """Validate ``payload``. Returns a list of human-readable problems.

    Uses ``jsonschema`` when installed and a structural check otherwise, so a
    deployment without the optional dependency still catches the common case
    (missing and unexpected fields) rather than silently checking nothing.
    """
    if not isinstance(payload, dict):
        return [f"expected a JSON object, got {type(payload).__name__}"]

    try:
        import jsonschema
    except ImportError:
        return _structural_check(payload, json_schema)

    validator = jsonschema.Draft202012Validator(json_schema)
    problems = []
    for error in sorted(validator.iter_errors(payload), key=lambda e: list(e.path)):
        path = ".".join(str(part) for part in error.path) or "(root)"
        problems.append(f"{path}: {error.message}")
    return problems[:20]


def _structural_check(payload: dict, json_schema: dict) -> list[str]:
    expected = set(json_schema.get("properties", {}))
    if not expected:
        return []
    got = set(payload)
    problems = []
    for missing in sorted(expected - got):
        problems.append(f"{missing}: required field is absent")
    if json_schema.get("additionalProperties") is False:
        for extra in sorted(got - expected):
            problems.append(f"{extra}: field is not in the schema")
    return problems


def coerce_to_schema(payload: Any, json_schema: dict) -> dict:
    """Best-effort repair of a nearly-right payload.

    Handles the failures that are mechanical rather than semantic: a bare list
    where an object was asked for, a single-key wrapper (``{"data": {...}}``),
    a scalar where a list belongs, and a case-or-underscore mismatch in a field
    name. Anything beyond that is a real disagreement and is left to fail
    validation, because quietly guessing is how wrong data gets stored.
    """
    expected: dict = json_schema.get("properties", {})
    if not expected:
        return payload if isinstance(payload, dict) else {"items": payload}

    if isinstance(payload, list):
        # A list of records where one was requested: keep them all under a key
        # the caller can find rather than silently discarding all but the first.
        return {"items": payload}

    if not isinstance(payload, dict):
        return {"items": payload}

    # Unwrap a single-key envelope like {"result": {...}} or {"product": {...}}.
    if len(payload) == 1:
        only_value = next(iter(payload.values()))
        if isinstance(only_value, dict) and set(only_value) & set(expected):
            payload = only_value

    normalized: dict[str, Any] = {}
    lookup = {_normalize_key(key): key for key in payload}

    for field, field_schema in expected.items():
        source = lookup.get(_normalize_key(field))
        value = payload.get(source) if source is not None else None
        normalized[field] = _coerce_value(value, field_schema)

    # Keep anything the model volunteered that the schema did not ask for, so a
    # useful surprise is inspectable rather than lost.
    extras = {key: value for key, value in payload.items() if _normalize_key(key) not in
              {_normalize_key(f) for f in expected}}
    if extras and json_schema.get("additionalProperties") is not False:
        normalized.update(extras)

    return normalized


def _coerce_value(value: Any, field_schema: dict) -> Any:
    types = field_schema.get("type")
    types = [types] if isinstance(types, str) else list(types or [])

    if value is None or value == "":
        return [] if "array" in types else None

    if "array" in types and not isinstance(value, list):
        if isinstance(value, str) and "," in value:
            return [part.strip() for part in value.split(",") if part.strip()]
        return [value]

    if "array" not in types and isinstance(value, list):
        if not value:
            return None
        if len(value) == 1:
            return _coerce_value(value[0], field_schema)
        return ", ".join(str(entry) for entry in value)

    if "integer" in types and isinstance(value, str):
        digits = re.sub(r"[^\d-]", "", value)
        return int(digits) if digits.lstrip("-").isdigit() else value

    if "number" in types and "string" not in types and isinstance(value, str):
        cleaned = re.sub(r"[^\d.\-]", "", value)
        try:
            return float(cleaned)
        except ValueError:
            return value

    if "boolean" in types and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "y", "1"):
            return True
        if lowered in ("false", "no", "n", "0"):
            return False

    return value


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def fill_rate(payload: dict, hint: dict) -> float:
    """Share of the hint's fields that got a usable value."""
    if not hint:
        return 0.0
    filled = 0
    for field in hint:
        value = payload.get(field) if isinstance(payload, dict) else None
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        filled += 1
    return filled / len(hint)


def schema_hash(hint: Any) -> str:
    """Stable fingerprint of a schema, for cache and provenance keys.

    Sorted keys so ``{"a":..., "b":...}`` and ``{"b":..., "a":...}`` are one
    cache entry rather than two.
    """
    canonical = json.dumps(hint, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def describe_for_prompt(hint: Any) -> str:
    """The schema as the model should see it: readable, with the null rule."""
    return json.dumps(hint, indent=2, ensure_ascii=False, default=str)


__all__ = [
    "coerce_to_schema",
    "compile_schema",
    "describe_for_prompt",
    "fill_rate",
    "schema_hash",
    "validate",
]
