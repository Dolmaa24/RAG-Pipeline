"""Query understanding: one model call, three techniques.

Self-query, sub-query decomposition and step-back prompting are usually written
as three separate calls. They do not need to be. Each one asks the model to read
the same question and report something about it, so they are three fields of one
structured answer — and one round trip instead of three is the largest latency
saving available anywhere in the retrieval path, because everything else here is
measured in milliseconds while a model call is measured in hundreds.

What each contributes:

* **Self-query** turns "Q3 finance reports by Chen" into a filter. Without it the
  metadata columns are decoration — something has to decide that "finance" is a
  department and not a search term.
* **Decomposition** splits a question that is really several questions. "How did
  revenue and headcount change after the merger" retrieves badly as one query
  and well as two.
* **Step-back** generalises. When the specific question has no good match, the
  broader form of it often does, and it costs nothing extra here.

The call itself is skipped for questions simple enough not to need it, which is
most of them.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict
from typing import Optional

from pydantic import BaseModel, Field

from config import config
from observability import get_logger, metrics

from pipeline.retrieve.filters import MetadataFilter

log = get_logger("retrieve.understand")


class QueryPlan(BaseModel):
    """Everything the retriever needs to know about one question."""

    original: str
    #: The original plus any decomposition. Always contains the original, so a
    #: caller never has to special-case an empty plan.
    sub_queries: list[str] = Field(default_factory=list)
    step_back: str = ""
    filters: MetadataFilter = Field(default_factory=MetadataFilter)
    #: Named things, used to seed graph traversal.
    entities: list[str] = Field(default_factory=list)
    #: True when no model was called.
    trivial: bool = False

    def queries(self) -> list[str]:
        """Every query string to retrieve for, deduplicated, original first."""
        ordered = [self.original, *self.sub_queries]
        if self.step_back:
            ordered.append(self.step_back)
        seen: set[str] = set()
        out: list[str] = []
        for query in ordered:
            key = query.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(query.strip())
        return out


#: Signals that a question has more than one part, a comparison, or a time
#: constraint — the three things rewriting actually helps with.
_COMPLEX = re.compile(
    r"\b(and|or|versus|vs|compared?\s+to|between|both|after|before|since|"
    r"during|whereas|while|then|also|besides|as\s+well\s+as)\b"
    r"|\b(19|20)\d{2}\b"
    r"|\b(q[1-4]|quarter|month|year|latest|recent|last|this)\b"
    r"|\?.*\?",
    re.IGNORECASE,
)


def is_trivial(query: str) -> bool:
    """Whether this question can skip the model entirely.

    Short, single-clause, no time expression, no comparison. A lookup like
    "what is the refund policy" gains nothing from being decomposed, and paying
    a model call to discover that is the most avoidable latency there is.
    """
    text = query.strip()
    if not text:
        return True
    if len(text.split()) > config.RETRIEVE_SIMPLE_QUERY_WORDS:
        return False
    return not _COMPLEX.search(text)


_SCHEMA_HINT = {
    "sub_queries": "list of strings",
    "step_back": "string",
    "entities": "list of strings",
    "filters": "object",
}

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_queries": {"type": "array", "items": {"type": "string"}},
        "step_back": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "filters": {
            "type": "object",
            "properties": {
                "doc_type": {"type": "array", "items": {"type": "string"}},
                "department": {"type": "array", "items": {"type": "string"}},
                "author": {"type": "array", "items": {"type": "string"}},
                "region": {"type": "array", "items": {"type": "string"}},
                "permission_level": {"type": "array", "items": {"type": "string"}},
                "language": {"type": "array", "items": {"type": "string"}},
                "date_from": {"type": "string"},
                "date_to": {"type": "string"},
            },
        },
    },
    "required": ["sub_queries", "step_back", "entities", "filters"],
}

_PROMPT = """You are preparing a search question for a retrieval system.

Do three things and return them together:

1. sub_queries — if the question asks about more than one thing, split it into
   at most {max_subqueries} self-contained questions. If it asks about one
   thing, return an empty list. Do not pad.
2. step_back — one broader, more general version of the question, useful when
   the specific wording finds nothing. Keep it a question.
3. entities — the specific named people, organisations, products or places
   mentioned. Names only, no descriptions. Empty list if there are none.
4. filters — constraints stated in the question that belong to metadata rather
   than to the search text. Use ONLY these keys: doc_type, department, author,
   region, permission_level, language, date_from, date_to. Dates are
   YYYY-MM-DD. Omit any key the question does not state. Do not guess a
   department or a permission level that is not written down.

Return only what the question supports. An empty answer is better than an
invented one."""


_cache: "OrderedDict[str, QueryPlan]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(query: str, local_only: bool) -> str:
    normalized = " ".join(query.lower().split())
    return hashlib.sha256(f"{normalized}|{local_only}".encode("utf-8")).hexdigest()[:32]


def _cached(key: str) -> Optional[QueryPlan]:
    with _cache_lock:
        plan = _cache.get(key)
        if plan is not None:
            _cache.move_to_end(key)
        return plan


def _remember(key: str, plan: QueryPlan) -> None:
    with _cache_lock:
        _cache[key] = plan
        _cache.move_to_end(key)
        while len(_cache) > config.RETRIEVE_PLAN_CACHE_SIZE:
            _cache.popitem(last=False)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def understand(
    query: str,
    *,
    local_only: bool = False,
    filters: Optional[MetadataFilter] = None,
    force: bool = False,
) -> QueryPlan:
    """Build a :class:`QueryPlan` for one question.

    ``filters`` supplied by the caller are authoritative and are merged over
    whatever the model inferred — a caller who says which department they are
    searching knows better than a model reading the sentence.
    """
    query = query.strip()
    plan = QueryPlan(original=query, filters=filters or MetadataFilter())

    if not query:
        plan.trivial = True
        return plan

    if not force and (not config.RETRIEVE_REWRITE_ENABLED or is_trivial(query)):
        metrics.incr("retrieve.understand.skipped")
        plan.trivial = True
        log.debug("retrieve.understand.skipped", query=query[:60])
        return plan

    key = _cache_key(query, local_only)
    hit = _cached(key)
    if hit is not None:
        metrics.incr("retrieve.understand.cache_hit")
        merged = hit.model_copy(deep=True)
        if filters is not None and not filters.is_empty():
            merged.filters = _merge(merged.filters, filters)
        return merged

    try:
        from pipeline.extract.llm import INTERACTIVE, get_backend

        # Interactive: someone is waiting on this, the prompt is small, and the
        # volume is low — the three conditions under which a hosted model's
        # throughput is worth reaching for.
        backend = get_backend(local_only=local_only, role=INTERACTIVE)
        with metrics.timer("retrieve.understand"):
            response = backend.complete_json(
                prompt=_PROMPT.format(max_subqueries=config.RETRIEVE_MAX_SUBQUERIES),
                content=query,
                schema_hint=_SCHEMA_HINT,
                json_schema=_JSON_SCHEMA,
            )
        data = response.data or {}
    except Exception as exc:
        # Retrieval without rewriting is worse, not broken. A model that is down
        # must not take the search with it.
        log.warning("retrieve.understand.failed", error=repr(exc))
        metrics.incr("retrieve.understand.failed")
        plan.trivial = True
        return plan

    sub_queries = [
        str(item).strip()
        for item in (data.get("sub_queries") or [])
        if str(item).strip() and str(item).strip().lower() != query.lower()
    ][: config.RETRIEVE_MAX_SUBQUERIES]

    inferred = MetadataFilter.from_model(data.get("filters") or {})

    plan = QueryPlan(
        original=query,
        sub_queries=sub_queries,
        step_back=str(data.get("step_back") or "").strip(),
        entities=[str(e).strip() for e in (data.get("entities") or []) if str(e).strip()],
        # Cleaned here rather than at the merge below, which only runs when the
        # caller supplied filters of their own — the uncommon case. Cleaning
        # there left the ordinary path unchecked *and* cached the bad value, so
        # one wrong guess kept emptying the search until the process restarted.
        filters=drop_unknown_values(inferred),
    )
    _remember(key, plan.model_copy(deep=True))

    if filters is not None and not filters.is_empty():
        plan.filters = _merge(plan.filters, filters)

    log.info(
        "retrieve.understand.done",
        sub_queries=len(plan.sub_queries),
        entities=len(plan.entities),
        filtered=not plan.filters.is_empty(),
    )
    return plan


def _merge(inferred: MetadataFilter, explicit: MetadataFilter) -> MetadataFilter:
    """Caller-supplied fields win over inferred ones, field by field.

    ``inferred`` has already been through :func:`drop_unknown_values` by the
    time it reaches here; explicit values are never checked, because a caller
    who filters by hand and gets nothing has learned something true.
    """
    merged = inferred.model_dump()
    for field, value in explicit.model_dump().items():
        if value:
            merged[field] = value
    return MetadataFilter(**merged)


def drop_unknown_values(inferred: MetadataFilter) -> MetadataFilter:
    """Remove inferred values the corpus does not actually hold.

    Filtering happens *before* retrieval, deliberately, so that a top-k is a
    real top-k rather than whatever survived a filter afterwards. The cost is
    that one wrong value returns nothing at all — and nothing at all is exactly
    what an unanswerable question returns, so the two are indistinguishable to
    whoever asked.

    Asked "what are the rules for using articles in English", the model read
    "in English" as a language filter and produced ``language: ["English"]``.
    The store holds the ISO code ``en``. Five matching passages became zero, and
    the answer said the corpus did not cover it — about a document that was
    entirely about the subject.

    Only *inferred* values are dropped. A caller who filters by hand and gets
    nothing has learned something true about their corpus; a model that guessed
    a value out of a sentence has not, and should not be able to silence a
    search by guessing wrongly.
    """
    values = inferred.model_dump()
    if not any(values.values()):
        return inferred

    try:
        from pipeline.store.lance import LanceStore

        store = LanceStore()
        if store.table is None:
            return inferred
    except Exception as exc:  # a filter check must never fail a search
        log.debug("retrieve.filter_check_skipped", error=repr(exc))
        return inferred

    cleaned = dict(values)
    for field, wanted in values.items():
        if not isinstance(wanted, list) or not wanted:
            continue
        try:
            known = {str(v).strip().lower() for v in store.distinct(field, limit=200)}
        except Exception:
            continue
        if not known:
            continue

        kept = [v for v in wanted if str(v).strip().lower() in known]
        if kept != wanted:
            log.info(
                "retrieve.filter_dropped",
                field=field,
                dropped=[v for v in wanted if v not in kept],
                known=sorted(known)[:8],
            )
            metrics.incr("retrieve.filter_dropped")
        cleaned[field] = kept

    return MetadataFilter(**cleaned)


__all__ = ["QueryPlan", "clear_cache", "drop_unknown_values", "is_trivial", "understand"]
