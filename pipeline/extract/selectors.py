"""Tier 2: learn a site's selectors once, then replay them for free.

This is the tier that changes what the project *is*.

The insight is that an LLM is being asked the wrong question. Asking "what is
the price on this page?" costs a model call **per page**, forever. Asking "which
CSS selector holds the price on this site?" costs one model call **per domain**,
and the answer is reusable, inspectable, versionable, and runs in about ten
milliseconds.

So on first visit to a domain the model is shown a pruned DOM skeleton and asked
to emit selectors rather than data. The spec is verified against the page it was
learned from, stored in Mongo, and replayed on every subsequent page from that
domain. A drift check watches the fill rate: when a site changes its markup the
rate collapses, the spec is retired, and the next page relearns it.

The trade-off is honest — selector specs need maintenance, which is precisely
the cost the LLM path avoids. The drift check is what makes that maintenance
automatic instead of a person noticing months later that a column went empty.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from config import config
from errors import ExtractError
from observability import get_logger
from urls import registrable_host

log = get_logger("extract.selectors")

#: Skeleton size handed to the model. Big enough to include the interesting
#: nodes on a normal page, small enough to stay well inside an 8k context.
SKELETON_MAX_CHARS = 9000
SKELETON_MAX_NODES = 220

_CLASS_NOISE = re.compile(r"^(?:css-|sc-|jsx-|styles?__|_[a-z0-9]{4,}$)", re.IGNORECASE)
#: Framework-generated class names change on every build, so a selector built
#: from them is stale before it is stored.
_HASHED_CLASS = re.compile(r"^[a-z0-9_-]*[0-9a-f]{6,}[a-z0-9_-]*$", re.IGNORECASE)


@dataclass(slots=True)
class SelectorRule:
    """Where one field lives in the DOM."""

    selector: str
    #: None means "the element's text"; otherwise an attribute name.
    attribute: Optional[str] = None
    multiple: bool = False
    #: Optional regex applied to the extracted string, first group wins.
    regex: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class SelectorSpec:
    """A learned, replayable extraction spec for one domain and schema."""

    domain: str
    schema_hash: str
    rules: dict[str, SelectorRule] = field(default_factory=dict)
    #: Path prefix this spec was learned on. A site's /product/ pages and its
    #: /blog/ pages have nothing structurally in common, so they get separate
    #: specs rather than one that half-works on both.
    path_prefix: str = "/"
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uses: int = 0
    #: Exponential moving average of the fill rate. An EMA rather than a plain
    #: mean so a spec that broke last week is not propped up by a good month.
    avg_fill_rate: float = 0.0
    learned_from: Optional[str] = None
    retired: bool = False

    @property
    def key(self) -> str:
        return f"{self.domain}|{self.path_prefix}|{self.schema_hash}"

    @property
    def is_stale(self) -> bool:
        if self.retired:
            return True
        if self.uses >= 5 and self.avg_fill_rate < config.SPEC_DRIFT_FILL_RATE:
            return True
        age = datetime.now(timezone.utc) - self.updated_at
        return age > timedelta(days=config.SPEC_MAX_AGE_DAYS)

    def record_use(self, fill_rate: float) -> None:
        self.uses += 1
        alpha = 0.25
        self.avg_fill_rate = (
            fill_rate if self.uses == 1 else (1 - alpha) * self.avg_fill_rate + alpha * fill_rate
        )
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "_id": self.key,
            "domain": self.domain,
            "schema_hash": self.schema_hash,
            "path_prefix": self.path_prefix,
            "rules": {name: rule.to_dict() for name, rule in self.rules.items()},
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "uses": self.uses,
            "avg_fill_rate": round(self.avg_fill_rate, 4),
            "learned_from": self.learned_from,
            "retired": self.retired,
        }

    @classmethod
    def from_dict(cls, doc: dict) -> "SelectorSpec":
        return cls(
            domain=doc["domain"],
            schema_hash=doc["schema_hash"],
            rules={
                name: SelectorRule(**rule) for name, rule in (doc.get("rules") or {}).items()
            },
            path_prefix=doc.get("path_prefix", "/"),
            version=doc.get("version", 1),
            created_at=_as_utc(doc.get("created_at")),
            updated_at=_as_utc(doc.get("updated_at")),
            uses=doc.get("uses", 0),
            avg_fill_rate=doc.get("avg_fill_rate", 0.0),
            learned_from=doc.get("learned_from"),
            retired=doc.get("retired", False),
        )


def _as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def spec_key(url: str, schema_hash: str) -> tuple[str, str]:
    """The (domain, path_prefix) a URL's spec is filed under."""
    from urllib.parse import urlsplit

    domain = registrable_host(url)
    segments = [part for part in urlsplit(url).path.split("/") if part]
    # One level of path is the right granularity: /product/ vs /blog/ differ
    # structurally, /product/blue-widget vs /product/red-widget do not.
    prefix = f"/{segments[0]}/" if segments else "/"
    return domain, prefix


def apply_spec(spec: SelectorSpec, html: str, schema_hint: dict) -> tuple[dict, float]:
    """Run a spec against a page. Returns the record and its fill rate."""
    from selectolax.lexbor import LexborHTMLParser

    from .schema import fill_rate as compute_fill_rate

    tree = LexborHTMLParser(html)
    record: dict[str, Any] = {}

    for name in schema_hint:
        rule = spec.rules.get(name)
        if rule is None:
            record[name] = None
            continue
        try:
            record[name] = _apply_rule(tree, rule)
        except Exception as exc:
            log.debug("selectors.rule_failed", field=name, selector=rule.selector, error=repr(exc))
            record[name] = None

    return record, compute_fill_rate(record, schema_hint)


def _apply_rule(tree, rule: SelectorRule) -> Any:
    nodes = tree.css(rule.selector)
    if not nodes:
        return [] if rule.multiple else None

    values = []
    for node in nodes if rule.multiple else nodes[:1]:
        if rule.attribute:
            raw = node.attributes.get(rule.attribute)
        else:
            raw = node.text(separator=" ", strip=True)
        if raw is None:
            continue
        value = re.sub(r"\s+", " ", str(raw)).strip()
        if rule.regex:
            match = re.search(rule.regex, value)
            if match is None:
                continue
            value = match.group(1) if match.groups() else match.group(0)
        if value:
            values.append(value)

    if rule.multiple:
        return values
    return values[0] if values else None


_LEARN_PROMPT = """You are writing a reusable web scraper for one website.

Below is a pruned outline of one page from that site: each line is a CSS path,
then the text or attribute value found there.

Return a JSON object mapping each requested FIELD to the CSS selector that
holds it, in this exact form:

{{"field_name": {{"selector": "div.price > span", "attribute": null,
  "multiple": false, "regex": null}}}}

Rules:
- "selector" must be a standard CSS selector that works on every page of this
  site, not only this one. Prefer stable hooks: semantic tags, itemprop, data-*
  and aria-* attributes, and human-written class names.
- Standard CSS only. jQuery extensions such as :contains(), :eq(), :first,
  :last and :visible are not CSS and will match nothing.
- Never use selectors built on generated class names (hashes such as
  "css-1x2y3z" or "sc-fJbEBl"), :nth-child positions, or ids that look
  generated. Those change on the next deploy.
- "attribute" is null to take the element's text, or an attribute name such as
  "href", "src", "content" or "datetime".
- "multiple" is true when the field is a list.
- "regex" is optional; use it only to pull a value out of surrounding text,
  with the value in capture group 1.
- If a field genuinely has no home on this page, map it to null.
- Return only the JSON object.

FIELDS: {fields}

PAGE OUTLINE:
{skeleton}
"""


def selector_map_schema(schema_hint: dict) -> dict:
    """JSON Schema for the *selector map* the model is asked to write.

    Worth constraining properly rather than asking for free-form JSON. A backend
    that enforces schemas (Groq) then cannot return a malformed rule at all, and
    a backend that constrains decoding from the schema (Ollama) knows exactly
    where the object ends — which is the difference between a two-second answer
    and generating until the token cap.
    """
    rule = {
        "type": ["object", "null"],
        "properties": {
            "selector": {"type": "string"},
            "attribute": {"type": ["string", "null"]},
            "multiple": {"type": "boolean"},
            "regex": {"type": ["string", "null"]},
        },
        "required": ["selector", "attribute", "multiple", "regex"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "title": "selector_map",
        "properties": {name: rule for name in schema_hint},
        "required": list(schema_hint),
        "additionalProperties": False,
    }


def learn_spec(
    url: str,
    html: str,
    schema_hint: dict,
    schema_hash: str,
    backend,
) -> Optional[SelectorSpec]:
    """Ask the model for selectors, then verify them before trusting them.

    Verification is what makes this safe to store: the spec is applied to the
    very page it was learned from, and kept only if it reproduces at least
    ``MIN_FILL_RATE`` of the fields. A hallucinated selector fills nothing and
    is discarded on the spot rather than silently returning nulls forever.
    """
    skeleton = dom_skeleton(html)
    if not skeleton.strip():
        return None

    domain, prefix = spec_key(url, schema_hash)
    prompt = _LEARN_PROMPT.format(fields=", ".join(schema_hint), skeleton=skeleton)

    started = time.perf_counter()
    try:
        response = backend.complete_json(
            prompt="Return the selector map described above.",
            content=prompt,
            schema_hint={name: "object" for name in schema_hint},
            json_schema=selector_map_schema(schema_hint),
        )
    except ExtractError as exc:
        log.warning("selectors.learn_failed", domain=domain, error=str(exc)[:200])
        return None

    rules = _parse_rules(response.data, schema_hint)
    if not rules:
        log.info("selectors.no_rules_returned", domain=domain)
        return None

    # Verify each rule on the page it was written for, and keep only the ones
    # that actually returned something. A selector that matches nothing here
    # will match nothing on every later page too — storing it means that field
    # is null forever while the spec looks perfectly reasonable.
    rules, dropped = _prune_dead_rules(rules, html)
    if dropped:
        log.info("selectors.dropped_dead_rules", domain=domain, fields=dropped)
    if not rules:
        log.info("selectors.no_working_rules", domain=domain)
        return None

    spec = SelectorSpec(
        domain=domain,
        schema_hash=schema_hash,
        rules=rules,
        path_prefix=prefix,
        learned_from=url,
    )

    _, verified_fill = apply_spec(spec, html, schema_hint)
    elapsed = time.perf_counter() - started

    if verified_fill < config.MIN_FILL_RATE:
        log.info(
            "selectors.rejected",
            domain=domain,
            fill_rate=round(verified_fill, 2),
            required=config.MIN_FILL_RATE,
            seconds=round(elapsed, 1),
        )
        return None

    spec.record_use(verified_fill)
    log.info(
        "selectors.learned",
        domain=domain,
        prefix=prefix,
        fields=len(rules),
        fill_rate=round(verified_fill, 2),
        seconds=round(elapsed, 1),
    )
    return spec


def _prune_dead_rules(
    rules: dict[str, SelectorRule], html: str
) -> tuple[dict[str, SelectorRule], list[str]]:
    """Drop rules that extract nothing from the page they were written for."""
    from selectolax.lexbor import LexborHTMLParser

    tree = LexborHTMLParser(html)
    kept: dict[str, SelectorRule] = {}
    dropped: list[str] = []

    for name, rule in rules.items():
        try:
            value = _apply_rule(tree, rule)
        except Exception:
            # An invalid selector raises rather than returning nothing. Same
            # verdict either way: the rule is not usable.
            value = None
        if value in (None, "", []):
            dropped.append(name)
        else:
            kept[name] = rule
    return kept, dropped


def _parse_rules(payload: Any, schema_hint: dict) -> dict[str, SelectorRule]:
    """Accept the shapes models actually return, reject the unusable ones."""
    if not isinstance(payload, dict):
        return {}
    if len(payload) == 1:
        only = next(iter(payload.values()))
        if isinstance(only, dict) and set(only) & set(schema_hint):
            payload = only

    rules: dict[str, SelectorRule] = {}
    for name in schema_hint:
        raw = payload.get(name)
        if raw is None:
            continue
        if isinstance(raw, str):
            raw = {"selector": raw}
        if not isinstance(raw, dict):
            continue

        selector = str(raw.get("selector") or "").strip()
        if not selector or not _selector_is_sane(selector):
            continue

        rules[name] = SelectorRule(
            selector=selector,
            attribute=(str(raw["attribute"]).strip() or None) if raw.get("attribute") else None,
            multiple=bool(raw.get("multiple")),
            regex=(str(raw["regex"]) or None) if raw.get("regex") else None,
        )
    return rules


#: jQuery extensions that look like CSS and are not. A CSS engine either raises
#: on these or matches nothing, so a rule built from one yields null forever
#: while looking perfectly reasonable in the stored spec.
_JQUERY_PSEUDO = (":contains(", ":eq(", ":first", ":last", ":visible", ":hidden",
                  ":parent", ":header", ":input", ":gt(", ":lt(")


def _selector_is_sane(selector: str) -> bool:
    """Reject selectors that are not CSS, or cannot survive the next deploy."""
    if len(selector) > 300 or "\n" in selector:
        return False
    lowered = selector.lower()
    if any(marker in lowered for marker in _JQUERY_PSEUDO):
        return False
    # Positional selectors break the moment the site adds a row.
    if ":nth-child" in lowered or ":nth-of-type" in lowered:
        return False
    for token in re.findall(r"\.([A-Za-z0-9_-]+)", selector):
        if _HASHED_CLASS.match(token) or _CLASS_NOISE.match(token):
            return False
    return True


def dom_skeleton(html: str, max_chars: int = SKELETON_MAX_CHARS) -> str:
    """A pruned outline of the DOM: CSS path, then what is there.

    Sending raw HTML would blow the context window on any real page, and most
    of those tokens are markup the model does not need. What it needs is the
    *addresses* — which tag, with which stable attributes, holds which value.
    """
    from selectolax.lexbor import LexborHTMLParser

    tree = LexborHTMLParser(html)
    for node in tree.css("script, style, noscript, svg, path, iframe"):
        node.decompose()

    lines: list[str] = []
    total = 0
    body = tree.body or tree.root
    if body is None:
        return ""

    for node in body.traverse(include_text=False):
        if len(lines) >= SKELETON_MAX_NODES or total >= max_chars:
            break
        tag = node.tag
        if tag in ("body", "html", "head", "br", "hr", "meta", "link"):
            continue

        descriptor = _describe(node)
        value = _node_value(node)
        if value is None:
            continue

        line = f"{descriptor} -> {value}"
        lines.append(line)
        total += len(line) + 1

    return "\n".join(lines)


def _describe(node) -> str:
    """A short, stable CSS-ish address for one node."""
    parts = [node.tag]
    attributes = node.attributes

    node_id = attributes.get("id")
    if node_id and not _HASHED_CLASS.match(node_id):
        parts.append(f"#{node_id}")

    classes = (attributes.get("class") or "").split()
    stable = [
        cls for cls in classes if not _HASHED_CLASS.match(cls) and not _CLASS_NOISE.match(cls)
    ][:3]
    parts.extend(f".{cls}" for cls in stable)

    # Attributes worth naming because they are contracts rather than styling.
    for name in ("itemprop", "data-testid", "data-test", "data-qa", "role", "aria-label", "name"):
        value = attributes.get(name)
        if value:
            parts.append(f"[{name}={value[:40]}]")
            break

    return "".join(parts)


def _node_value(node) -> Optional[str]:
    """What this node contributes: its own text, or a meaningful attribute."""
    tag = node.tag
    attributes = node.attributes

    if tag == "a" and attributes.get("href"):
        text = node.text(strip=True)[:80]
        return f'@href="{attributes["href"][:120]}" text="{text}"'
    if tag == "img":
        return f'@src="{(attributes.get("src") or "")[:120]}" @alt="{(attributes.get("alt") or "")[:60]}"'
    if tag == "time" and attributes.get("datetime"):
        return f'@datetime="{attributes["datetime"][:40]}"'
    if tag in ("input", "meta") and attributes.get("content"):
        return f'@content="{attributes["content"][:100]}"'

    # Only leaf-ish nodes contribute text; otherwise every ancestor repeats the
    # whole page and the skeleton is nothing but duplication.
    children = [child for child in node.iter(include_text=False)]
    if children:
        return None

    text = node.text(separator=" ", strip=True)
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)[:120]
    return f'"{text}"'


#: How many times to try authoring a spec for one (domain, prefix, schema)
#: before giving up on it. Learning costs a model call; on a site where it
#: cannot succeed, retrying on every page would double the cost of the whole
#: crawl forever. Two attempts is enough to survive one bad response.
MAX_LEARN_ATTEMPTS = 2


class SpecStore:
    """Persist specs in Mongo, with an in-process cache in front.

    The cache matters: a 16-thread io worker processing 200 pages from one
    domain should do one spec read, not two hundred.
    """

    def __init__(self, database=None) -> None:
        self._db = database
        self._memory: dict[str, SelectorSpec] = {}
        self._learn_failures: dict[str, int] = {}

    def get(self, url: str, schema_hash: str) -> Optional[SelectorSpec]:
        domain, prefix = spec_key(url, schema_hash)
        key = f"{domain}|{prefix}|{schema_hash}"

        cached = self._memory.get(key)
        if cached is not None:
            return None if cached.is_stale else cached

        doc = None
        if self._db is not None and self._db.is_configured:
            try:
                doc = self._db.specs_collection().find_one({"_id": key})
            except Exception as exc:
                log.debug("specs.read_failed", error=repr(exc))

        if not doc:
            return None

        spec = SelectorSpec.from_dict(doc)
        self._memory[key] = spec
        if spec.is_stale:
            log.info("specs.stale", domain=domain, fill_rate=round(spec.avg_fill_rate, 2))
            return None
        return spec

    def put(self, spec: SelectorSpec) -> None:
        self._memory[spec.key] = spec
        if self._db is None or not self._db.is_configured:
            return
        try:
            self._db.specs_collection().replace_one(
                {"_id": spec.key}, spec.to_dict(), upsert=True
            )
            log.info("specs.saved", domain=spec.domain, prefix=spec.path_prefix)
        except Exception as exc:
            log.warning("specs.write_failed", domain=spec.domain, error=repr(exc))

    def retire(self, spec: SelectorSpec, reason: str) -> None:
        spec.retired = True
        self._memory[spec.key] = spec
        log.warning("specs.retired", domain=spec.domain, reason=reason,
                    fill_rate=round(spec.avg_fill_rate, 2))
        if self._db is None or not self._db.is_configured:
            return
        try:
            self._db.specs_collection().update_one(
                {"_id": spec.key},
                {"$set": {"retired": True, "retired_reason": reason,
                          "retired_at": datetime.now(timezone.utc)}},
            )
        except Exception as exc:
            log.debug("specs.retire_failed", error=repr(exc))

    def should_learn(self, url: str, schema_hash: str) -> bool:
        """Whether authoring a spec for this key is still worth a model call.

        Learning is only free when it succeeds — the call it costs is the one
        tier 3 would have made anyway. When it *fails*, that page paid twice,
        so a site where learning cannot work must stop being asked.
        """
        domain, prefix = spec_key(url, schema_hash)
        key = f"{domain}|{prefix}|{schema_hash}"

        if self._learn_failures.get(key, 0) >= MAX_LEARN_ATTEMPTS:
            return False

        if self._db is not None and self._db.is_configured and key not in self._learn_failures:
            try:
                doc = self._db.specs_collection().find_one({"_id": key}, {"learn_attempts": 1})
            except Exception:
                doc = None
            if doc and doc.get("learn_attempts", 0) >= MAX_LEARN_ATTEMPTS:
                self._learn_failures[key] = doc["learn_attempts"]
                return False
        return True

    def record_learn_failure(self, url: str, schema_hash: str) -> None:
        domain, prefix = spec_key(url, schema_hash)
        key = f"{domain}|{prefix}|{schema_hash}"
        attempts = self._learn_failures.get(key, 0) + 1
        self._learn_failures[key] = attempts

        if attempts >= MAX_LEARN_ATTEMPTS:
            log.info("specs.learning_abandoned", domain=domain, prefix=prefix, attempts=attempts)
        if self._db is None or not self._db.is_configured:
            return
        try:
            self._db.specs_collection().update_one(
                {"_id": key},
                {
                    "$set": {
                        "domain": domain,
                        "schema_hash": schema_hash,
                        "path_prefix": prefix,
                        "learn_attempts": attempts,
                        "last_attempt_at": datetime.now(timezone.utc),
                    }
                },
                upsert=True,
            )
        except Exception as exc:
            log.debug("specs.failure_record_failed", error=repr(exc))

    def all(self, limit: int = 200) -> list[dict]:
        if self._db is None or not self._db.is_configured:
            return [spec.to_dict() for spec in list(self._memory.values())[:limit]]
        try:
            return list(self._db.specs_collection().find().limit(limit))
        except Exception:
            return []

    def clear_memory(self) -> None:
        self._memory.clear()


def rules_to_json(spec: SelectorSpec) -> str:
    """Human-readable dump of a spec, for the dashboard and for debugging."""
    return json.dumps(
        {name: rule.to_dict() for name, rule in spec.rules.items()}, indent=2, ensure_ascii=False
    )


__all__ = [
    "SelectorRule",
    "SelectorSpec",
    "SpecStore",
    "apply_spec",
    "dom_skeleton",
    "learn_spec",
    "rules_to_json",
    "spec_key",
]
