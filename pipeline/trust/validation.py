"""Output validation: catching the answers that are wrong rather than absent.

With a non-deterministic extractor this matters **more**, not less. A missing
field is obvious. A field containing ``"N/A"``, or ``"string"`` echoed back from
the schema, or ``"I'm sorry, I cannot find that information"`` looks like data
all the way into the database and out again into whatever reads it.

The rules below are deliberately narrow. Each one fires on something that is
*never* a legitimate extracted value, so a failure is evidence rather than a
suggestion. Rules that would need judgement — "is this price plausible?" —
belong in a domain-specific layer, not here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from observability import get_logger

log = get_logger("trust.validation")

#: Values that mean "no value" but are stored as though they were one.
PLACEHOLDERS = frozenset(
    {
        "n/a", "na", "none", "null", "nil", "-", "--", "—", "unknown", "unspecified",
        "not available", "not applicable", "not found", "no data", "tbd", "todo",
        "undefined", "nan", "empty", "blank", "?", "...", "[]", "{}",
    }
)

#: The schema's own type words, echoed back as if they were the answer. A
#: classic small-model failure: the model copies the schema instead of filling it.
SCHEMA_ECHOES = frozenset(
    {
        "string", "number", "integer", "boolean", "float", "array", "object", "list",
        "list of strings", "text", "date", "value", "field", "example", "your answer here",
    }
)

#: A model declining to answer, stored verbatim as the answer.
_REFUSAL = re.compile(
    r"^\s*(i'?m sorry|i cannot|i can'?t|i am unable|as an ai|i don'?t have|"
    r"unfortunately,? (i|the)|there is no|no information (is )?(available|provided)|"
    r"the (text|content|page) (does not|doesn'?t) )",
    re.IGNORECASE,
)

_LOREM = re.compile(r"lorem ipsum|dolor sit amet", re.IGNORECASE)
_URL_FIELD = re.compile(r"(url|link|href|image|logo|thumbnail|photo)s?$", re.IGNORECASE)
_EMAIL_FIELD = re.compile(r"e?mail", re.IGNORECASE)
_DATE_FIELD = re.compile(r"(date|published|updated|modified|created|时间)", re.IGNORECASE)
_EMAIL_VALUE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")

#: A free-text field longer than this is almost always the whole page dumped
#: into one cell, which is a failure to extract rather than a long answer.
MAX_FIELD_CHARS = 20_000


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    check: Callable[[str, Any, dict], Optional[str]]
    #: "error" fails the record; "warning" annotates it.
    severity: str = "error"


def _rule_placeholder(field: str, value: Any, context: dict) -> Optional[str]:
    if isinstance(value, str) and value.strip().lower() in PLACEHOLDERS:
        return f"{field!r} is the placeholder {value.strip()!r}; it should be null"
    return None


def _rule_schema_echo(field: str, value: Any, context: dict) -> Optional[str]:
    if isinstance(value, str) and value.strip().lower() in SCHEMA_ECHOES:
        return f"{field!r} contains the schema's type word {value.strip()!r}, not a value"
    return None


def _rule_refusal(field: str, value: Any, context: dict) -> Optional[str]:
    if isinstance(value, str) and _REFUSAL.match(value):
        return f"{field!r} contains a model refusal rather than data: {value[:80]!r}"
    return None


def _rule_lorem(field: str, value: Any, context: dict) -> Optional[str]:
    if isinstance(value, str) and _LOREM.search(value):
        return f"{field!r} contains placeholder lorem ipsum text"
    return None


def _rule_too_long(field: str, value: Any, context: dict) -> Optional[str]:
    if isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
        return f"{field!r} is {len(value)} characters; the page was probably not parsed into fields"
    return None


def _rule_url_shape(field: str, value: Any, context: dict) -> Optional[str]:
    if not _URL_FIELD.search(field) or not isinstance(value, str) or not value.strip():
        return None
    if value.startswith(("http://", "https://", "data:", "//")):
        return None
    return f"{field!r} should be a URL but is {value[:60]!r}"


def _rule_email_shape(field: str, value: Any, context: dict) -> Optional[str]:
    if not _EMAIL_FIELD.search(field) or not isinstance(value, str) or not value.strip():
        return None
    return None if _EMAIL_VALUE.match(value.strip()) else f"{field!r} is not a valid email address"


def _rule_future_date(field: str, value: Any, context: dict) -> Optional[str]:
    """A publication date in the future is a parse error, not a scoop."""
    if not _DATE_FIELD.search(field) or not isinstance(value, str):
        return None
    from pipeline.normalize.dates import parse_date

    parsed = parse_date(value)
    if parsed is None:
        return None
    if parsed > datetime.now(timezone.utc).replace(year=datetime.now().year + 1):
        return f"{field!r} is dated {parsed.date()}, which is implausibly far in the future"
    return None


def _rule_prompt_echo(field: str, value: Any, context: dict) -> Optional[str]:
    """The model repeating the instruction back instead of answering it."""
    prompt = (context.get("prompt") or "").strip().lower()
    if not prompt or not isinstance(value, str) or len(value) < 25:
        return None
    return f"{field!r} repeats the instruction instead of answering it" if value.strip().lower() in prompt else None


DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("placeholder", _rule_placeholder),
    Rule("schema_echo", _rule_schema_echo),
    Rule("refusal", _rule_refusal),
    Rule("lorem_ipsum", _rule_lorem),
    Rule("oversized_field", _rule_too_long),
    Rule("url_shape", _rule_url_shape, severity="warning"),
    Rule("email_shape", _rule_email_shape, severity="warning"),
    Rule("future_date", _rule_future_date, severity="warning"),
    Rule("prompt_echo", _rule_prompt_echo),
)


@dataclass(slots=True)
class ValidationResult:
    errors: list[str]
    warnings: list[str]
    checked_fields: int

    @property
    def ok(self) -> bool:
        return not self.errors

    def all_messages(self) -> list[str]:
        return self.errors + [f"(warning) {message}" for message in self.warnings]


def validate_record(
    record: dict,
    *,
    schema_hint: Optional[dict] = None,
    prompt: str = "",
    rules: tuple[Rule, ...] = DEFAULT_RULES,
    required: Optional[list[str]] = None,
) -> ValidationResult:
    """Run every rule over every field. Lists are checked element by element."""
    errors: list[str] = []
    warnings: list[str] = []
    context = {"prompt": prompt, "schema": schema_hint or {}}
    checked = 0

    for field, value in (record or {}).items():
        values = value if isinstance(value, list) else [value]
        for entry in values:
            if entry is None:
                continue
            checked += 1
            for rule in rules:
                try:
                    problem = rule.check(field, entry, context)
                except Exception as exc:  # a broken rule must not fail the record
                    log.debug("validation.rule_error", rule=rule.name, error=repr(exc))
                    continue
                if problem:
                    (errors if rule.severity == "error" else warnings).append(problem)

    for field in required or []:
        value = (record or {}).get(field)
        if value is None or (isinstance(value, (str, list, dict)) and not value):
            errors.append(f"{field!r} is required but empty")

    if schema_hint and record:
        empty = [name for name in schema_hint if _is_empty((record or {}).get(name))]
        if len(empty) == len(schema_hint):
            errors.append("every requested field is empty; nothing was extracted")

    return ValidationResult(errors=errors, warnings=warnings, checked_fields=checked)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return not value
    return False


__all__ = [
    "DEFAULT_RULES",
    "PLACEHOLDERS",
    "Rule",
    "SCHEMA_ECHOES",
    "ValidationResult",
    "validate_record",
]
