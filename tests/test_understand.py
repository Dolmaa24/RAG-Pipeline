"""Query understanding: the gate, the fused call, and the cache."""

from __future__ import annotations

import pytest

from config import config
from pipeline.retrieve.filters import MetadataFilter
from pipeline.retrieve.understand import QueryPlan, is_trivial, understand


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
