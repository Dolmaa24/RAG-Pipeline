"""Query understanding: the gate, the fused call, and the cache."""

from __future__ import annotations

import pytest

from config import config
from pipeline.retrieve.filters import MetadataFilter
from pipeline.retrieve.understand import (
    QueryPlan,
    _merge,
    drop_unknown_values,
    is_trivial,
    understand,
)


# --------------------------------------------------------------------------- #
# The gate — the call that is not made
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "query",
    [
        "what is the refund policy",
        "who is the CEO",
        "printer setup",
    ],
)
def test_simple_lookups_skip_the_model(query):
    assert is_trivial(query)


@pytest.mark.parametrize(
    "query",
    [
        "how did revenue and headcount change after the merger",
        "compare the 2025 and 2026 filings",
        "what happened to the project since Q3",
        "a considerably longer question about several different topics at once here",
    ],
)
def test_complex_questions_do_not_skip(query):
    assert not is_trivial(query)


def test_a_trivial_query_makes_no_call(fake_backend, monkeypatch):
    backend = fake_backend({})
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    plan = understand("what is the refund policy")
    assert plan.trivial is True
    assert backend.calls == 0
    assert plan.queries() == ["what is the refund policy"]


def test_rewriting_can_be_forced(fake_backend, monkeypatch):
    backend = fake_backend(
        {"sub_queries": [], "step_back": "", "entities": [], "filters": {}}
    )
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    understand("what is the refund policy", force=True)
    assert backend.calls == 1


def test_rewriting_can_be_disabled_globally(fake_backend, monkeypatch):
    backend = fake_backend({})
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)
    monkeypatch.setattr(config, "RETRIEVE_REWRITE_ENABLED", False)

    plan = understand("how did revenue and headcount change after the merger")
    assert plan.trivial is True
    assert backend.calls == 0


# --------------------------------------------------------------------------- #
# The fused call
# --------------------------------------------------------------------------- #


def _plan_backend(fake_backend):
    return fake_backend(
        {
            "sub_queries": [
                "how did revenue change after the merger",
                "how did headcount change after the merger",
            ],
            "step_back": "what changed after the merger",
            "entities": ["ACME Corporation"],
            "filters": {"department": ["finance"], "date_from": "2026-01-01"},
        }
    )


def test_one_call_returns_all_three_techniques(fake_backend, monkeypatch):
    """Self-query, decomposition and step-back in a single round trip."""
    backend = _plan_backend(fake_backend)
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    plan = understand("how did revenue and headcount change after the merger")

    assert backend.calls == 1
    assert len(plan.sub_queries) == 2
    assert plan.step_back == "what changed after the merger"
    assert plan.entities == ["ACME Corporation"]
    assert plan.filters.department == ["finance"]
    assert plan.filters.compile() == "department = 'finance' AND date >= '2026-01-01'"


def test_queries_include_the_original_first_and_deduplicate():
    plan = QueryPlan(
        original="What changed?",
        sub_queries=["what changed?", "and why"],
        step_back="And Why",
    )
    assert plan.queries() == ["What changed?", "and why"]


def test_subqueries_are_capped(fake_backend, monkeypatch):
    backend = fake_backend(
        {
            "sub_queries": [f"question {i}" for i in range(20)],
            "step_back": "",
            "entities": [],
            "filters": {},
        }
    )
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    plan = understand("a long question with and or comparisons in it", force=True)
    assert len(plan.sub_queries) == config.RETRIEVE_MAX_SUBQUERIES


def test_a_subquery_echoing_the_original_is_dropped(fake_backend, monkeypatch):
    query = "how did revenue and headcount change"
    backend = fake_backend(
        {"sub_queries": [query.upper(), "real sub"], "step_back": "", "entities": [], "filters": {}}
    )
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    assert understand(query).sub_queries == ["real sub"]


def test_a_dead_model_degrades_to_no_rewriting(fake_backend, monkeypatch):
    """Retrieval without rewriting is worse. It is not broken."""
    backend = fake_backend(raises=RuntimeError("model unreachable"))
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    plan = understand("how did revenue and headcount change after the merger")
    assert plan.trivial is True
    assert plan.queries() == ["how did revenue and headcount change after the merger"]


# --------------------------------------------------------------------------- #
# The cache and caller-supplied filters
# --------------------------------------------------------------------------- #


def test_the_same_question_is_not_planned_twice(fake_backend, monkeypatch):
    backend = _plan_backend(fake_backend)
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    query = "how did revenue and headcount change after the merger"
    first = understand(query)
    second = understand(query)

    assert backend.calls == 1
    assert second.sub_queries == first.sub_queries


def test_the_cache_ignores_case_and_spacing(fake_backend, monkeypatch):
    backend = _plan_backend(fake_backend)
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    understand("how did revenue and headcount change after the merger")
    understand("How  did REVENUE and headcount   change after the merger")
    assert backend.calls == 1


def test_caller_filters_override_what_the_model_inferred(fake_backend, monkeypatch):
    backend = _plan_backend(fake_backend)
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    plan = understand(
        "how did revenue and headcount change after the merger",
        filters=MetadataFilter(department=["legal"]),
    )
    assert plan.filters.department == ["legal"]
    # A field the caller did not set keeps the inferred value.
    assert plan.filters.date_from == "2026-01-01"


def test_caller_filters_survive_a_cache_hit(fake_backend, monkeypatch):
    backend = _plan_backend(fake_backend)
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **_: backend)

    query = "how did revenue and headcount change after the merger"
    understand(query)
    plan = understand(query, filters=MetadataFilter(department=["legal"]))

    assert backend.calls == 1
    assert plan.filters.department == ["legal"]


def test_an_empty_query_is_handled():
    plan = understand("   ")
    assert plan.trivial is True
    assert plan.queries() == []


# --------------------------------------------------------------------------- #
# Inferred filters that the corpus cannot satisfy
# --------------------------------------------------------------------------- #


def test_an_inferred_value_the_corpus_lacks_is_dropped(monkeypatch):
    """The failure this exists for, in one line.

    Asked "what are the rules for using articles in English", the model read
    "in English" as a language filter and produced ``language: ["English"]``.
    The store holds the ISO code ``en``. Filtering happens before retrieval, so
    five matching passages became zero and the answer reported that the corpus
    did not cover it — about a document entirely about the subject.
    """
    from pipeline.retrieve.filters import MetadataFilter

    class FakeStore:
        table = object()

        def distinct(self, field, limit=200):
            return {"language": ["en"], "doc_type": ["html"]}.get(field, [])

    monkeypatch.setattr(
        "pipeline.store.lance.LanceStore", lambda *a, **k: FakeStore()
    )

    cleaned = drop_unknown_values(MetadataFilter(language=["English"]))
    assert cleaned.language == []


def test_a_value_the_corpus_holds_survives(monkeypatch):
    from pipeline.retrieve.filters import MetadataFilter

    class FakeStore:
        table = object()

        def distinct(self, field, limit=200):
            return {"language": ["en"], "doc_type": ["html", "document"]}.get(field, [])

    monkeypatch.setattr("pipeline.store.lance.LanceStore", lambda *a, **k: FakeStore())

    cleaned = drop_unknown_values(MetadataFilter(doc_type=["html"]))
    assert cleaned.doc_type == ["html"]


def test_matching_ignores_case(monkeypatch):
    from pipeline.retrieve.filters import MetadataFilter

    class FakeStore:
        table = object()

        def distinct(self, field, limit=200):
            return ["Finance"] if field == "department" else []

    monkeypatch.setattr("pipeline.store.lance.LanceStore", lambda *a, **k: FakeStore())

    assert drop_unknown_values(
        MetadataFilter(department=["finance"])
    ).department == ["finance"]


def test_a_caller_s_own_filter_is_never_dropped(monkeypatch):
    """A caller who filters by hand and gets nothing has learned something true.

    A model that guessed a value out of a sentence has not, and should not be
    able to silence a search by guessing wrongly. Cleaning therefore happens to
    the inferred half only, before the merge — so this checks that the explicit
    half survives a merge untouched even when the corpus does not hold it.
    """
    from pipeline.retrieve.filters import MetadataFilter

    class FakeStore:
        table = object()

        def distinct(self, field, limit=200):
            return ["en"]

    monkeypatch.setattr("pipeline.store.lance.LanceStore", lambda *a, **k: FakeStore())

    inferred = drop_unknown_values(MetadataFilter(language=["English"]))
    assert inferred.language == []

    merged = _merge(inferred, MetadataFilter(department=["nonexistent"]))
    assert merged.language == []
    assert merged.department == ["nonexistent"]


def test_an_empty_filter_needs_no_store(monkeypatch):
    from pipeline.retrieve.filters import MetadataFilter

    def explode(*args, **kwargs):
        raise AssertionError("the store was consulted for an empty filter")

    monkeypatch.setattr("pipeline.store.lance.LanceStore", explode)
    assert drop_unknown_values(MetadataFilter()).model_dump() == MetadataFilter().model_dump()


def test_a_store_that_cannot_be_read_leaves_filters_alone(monkeypatch):
    # A filter check must never fail a search.
    from pipeline.retrieve.filters import MetadataFilter

    def explode(*args, **kwargs):
        raise RuntimeError("lance is down")

    monkeypatch.setattr("pipeline.store.lance.LanceStore", explode)
    assert drop_unknown_values(MetadataFilter(language=["English"])).language == ["English"]


def test_an_inferred_filter_is_cleaned_on_the_ordinary_path(monkeypatch):
    """The path with no caller-supplied filters, which is most of them.

    The first version of this check lived in ``_merge``, which only runs when
    the caller passed filters of their own. The ordinary path went unchecked
    *and* cached the bad value, so one wrong guess kept emptying the search
    until the process restarted.
    """
    from pipeline.retrieve.filters import MetadataFilter
    from pipeline.retrieve.understand import clear_cache

    class FakeStore:
        table = object()

        def distinct(self, field, limit=200):
            return ["en"] if field == "language" else []

    class FakeBackend:
        name, model = "fake", "m"

        def complete_json(self, **kwargs):
            from pipeline.extract.llm.base import LLMResponse

            return LLMResponse(
                data={"sub_queries": [], "step_back": "", "entities": [],
                      "filters": {"language": ["English"]}},
                backend="fake", model="m",
            )

    monkeypatch.setattr("pipeline.store.lance.LanceStore", lambda *a, **k: FakeStore())
    monkeypatch.setattr(
        "pipeline.extract.llm.get_backend", lambda **kwargs: FakeBackend()
    )
    clear_cache()

    plan = understand("what are the rules for articles in English?", force=True)
    assert plan.filters.language == [], plan.filters.model_dump()


def test_the_cache_does_not_hold_an_uncleaned_filter(monkeypatch):
    # _remember stores the plan, so a value cleaned after caching would come
    # straight back on the next identical question.
    from pipeline.retrieve.understand import _cache_key, _cached, clear_cache

    class FakeStore:
        table = object()

        def distinct(self, field, limit=200):
            return ["en"] if field == "language" else []

    class FakeBackend:
        name, model = "fake", "m"

        def complete_json(self, **kwargs):
            from pipeline.extract.llm.base import LLMResponse

            return LLMResponse(
                data={"sub_queries": [], "step_back": "", "entities": [],
                      "filters": {"language": ["English"]}},
                backend="fake", model="m",
            )

    monkeypatch.setattr("pipeline.store.lance.LanceStore", lambda *a, **k: FakeStore())
    monkeypatch.setattr("pipeline.extract.llm.get_backend", lambda **kwargs: FakeBackend())
    clear_cache()

    question = "another question about articles in English?"
    understand(question, force=True)
    cached = _cached(_cache_key(question, False))

    assert cached is not None
    assert cached.filters.language == []
