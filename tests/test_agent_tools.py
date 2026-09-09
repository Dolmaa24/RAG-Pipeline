"""The tool catalog: schemas, the effect gate, and every handler.

The foundation the agent loop was built on, and deliberately built first. No
model runs in any of this — the point of declaring tools before building a loop
is that this layer stays correct whether or not anything ever calls it
agentically.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from pipeline.agents.tools import (
    READ_ONLY,
    Effect,
    ToolError,
    catalog,
    describe_all,
    get,
    invoke,
)
from pipeline.agents.tools import registry as reg
from pipeline.agents.tools.models import FilterArgs, SearchResult, Passage


class NestedArgs(BaseModel):
    """Deliberately nested, to prove registration rejects it."""

    filters: FilterArgs = FilterArgs()


class EmptyArgs(BaseModel):
    pass

ALL_EFFECTS = list(Effect)

EXPECTED = {
    "corpus_profile": Effect.READ,
    "search_corpus": Effect.READ,
    "answer_from_corpus": Effect.READ,
    "graph_neighbors": Effect.READ,
    "graph_path": Effect.READ,
    "graph_relations": Effect.READ,
    "fetch_chunk": Effect.READ,
    "poll_task": Effect.READ,
    "detect_url": Effect.NETWORK,
    "discover_sitemap": Effect.NETWORK,
    "extract_url": Effect.WRITE,
    "crawl_site": Effect.WRITE,
    "index_document": Effect.WRITE,
    # The build tools. Registered always and reachable only from a build,
    # because effects — not registration — are what gates them.
    "write_source": Effect.CODE,
    "read_source": Effect.CODE,
    "list_workspace": Effect.CODE,
    "run_tests": Effect.EXECUTE,
}


def test_every_planned_tool_is_registered_with_its_planned_effect():
    registered = {spec.name: spec.effect for spec in catalog(ALL_EFFECTS)}
    assert registered == EXPECTED


def test_catalog_hides_tools_the_run_may_not_use():
    # Not "advertises and refuses": a model shown a tool it cannot call will
    # call it, spend a turn learning that, and often try again.
    names = {spec.name for spec in catalog(READ_ONLY)}
    assert "extract_url" not in names
    assert "detect_url" not in names
    assert "search_corpus" in names


def test_read_only_is_the_default():
    assert {s.name for s in catalog()} == {s.name for s in catalog(READ_ONLY)}


@pytest.mark.parametrize("spec", catalog(ALL_EFFECTS), ids=lambda s: s.name)
def test_every_tool_publishes_a_flat_schema(spec):
    schema = spec.json_schema()
    assert "$defs" not in schema, "nested models hurt small-model tool selection"
    assert schema["type"] == "object"


@pytest.mark.parametrize("spec", catalog(ALL_EFFECTS), ids=lambda s: s.name)
def test_every_description_is_written_for_a_model(spec):
    # Long enough to say when to use it, not just what it is. A 3B model picks
    # almost entirely from this text.
    assert len(spec.description) > 80
    assert spec.description == spec.description.strip()


def test_describe_all_is_the_shape_a_tool_api_wants():
    described = describe_all(ALL_EFFECTS)
    assert len(described) == len(EXPECTED)
    for entry in described:
        assert set(entry) == {"name", "description", "input_schema"}


def test_get_names_the_alternatives_when_a_tool_does_not_exist():
    with pytest.raises(ToolError, match="search_corpus"):
        get("serch_corpus")


@pytest.fixture
def scratch_registry(monkeypatch):
    monkeypatch.setattr(reg, "_REGISTRY", {})
    return reg


def test_a_nested_argument_model_is_rejected_at_registration(scratch_registry):
    with pytest.raises(ValueError, match="flat"):

        @scratch_registry.tool(
            name="nested", description="x" * 90, effect=Effect.READ, cost_ms=1
        )
        def handler(args: NestedArgs) -> SearchResult: ...


def test_a_handler_without_a_model_argument_is_rejected(scratch_registry):
    with pytest.raises(ValueError, match="BaseModel"):

        @scratch_registry.tool(
            name="loose", description="x" * 90, effect=Effect.READ, cost_ms=1
        )
        def handler(query: str) -> SearchResult: ...


def test_a_model_the_annotation_cannot_reach_says_why(scratch_registry):
    class Local(BaseModel):  # not resolvable from module globals
        pass

    with pytest.raises(ValueError, match="module level"):

        @scratch_registry.tool(
            name="local", description="x" * 90, effect=Effect.READ, cost_ms=1
        )
        def handler(args: Local) -> SearchResult: ...


def test_registering_the_same_name_twice_is_refused(scratch_registry):
    def make():
        @scratch_registry.tool(
            name="dupe", description="x" * 90, effect=Effect.READ, cost_ms=1
        )
        def handler(args: EmptyArgs) -> SearchResult:
            return SearchResult(query="")

    make()
    with pytest.raises(ValueError, match="already registered"):
        make()


@pytest.mark.parametrize(
    "name,args",
    [
        ("detect_url", {"url": "https://example.com"}),
        ("discover_sitemap", {"url": "https://example.com"}),
        ("extract_url", {"url": "https://example.com"}),
        ("index_document", {"text": "some text", "source": "notes.txt"}),
        ("crawl_site", {"start_url": "https://example.com"}),
    ],
)
def test_a_tool_beyond_the_allowance_is_refused_before_it_runs(name, args):
    call = invoke(name, args)  # read-only by default
    assert not call.ok
    assert "did not allow" in call.error


def test_network_permission_does_not_grant_write():
    call = invoke(
        "extract_url",
        {"url": "https://example.com"},
        allowed=READ_ONLY | {Effect.NETWORK},
    )
    assert not call.ok
    assert "write" in call.error


def test_an_unknown_tool_is_an_observation_not_an_exception():
    # The loop's job is to let the model read the problem and try again; an
    # exception would end the run instead of teaching it anything.
    call = invoke("hallucinated_tool", {})
    assert not call.ok
    assert "no tool named" in call.observation


def test_bad_arguments_are_explained_in_terms_the_model_can_act_on():
    call = invoke("search_corpus", {"query": "x", "limit": 999})
    assert not call.ok
    assert "limit" in call.observation


def test_a_missing_required_argument_is_reported_not_raised():
    call = invoke("search_corpus", {})
    assert not call.ok
    assert "query" in call.observation


def test_a_handler_that_raises_becomes_a_failed_call(monkeypatch):
    def explode(args):
        raise RuntimeError("kuzu is on fire")

    import dataclasses

    broken = dataclasses.replace(get("graph_neighbors"), handler=explode)
    monkeypatch.setitem(reg._REGISTRY, "graph_neighbors", broken)

    call = invoke("graph_neighbors", {"entity": "Acme"})
    assert not call.ok
    assert "kuzu is on fire" in call.observation


def test_a_successful_call_records_what_a_trace_needs():
    call = invoke("corpus_profile", {})
    assert call.ok
    row = call.to_dict()
    assert row["tool"] == "corpus_profile"
    assert row["effect"] == "read"
    assert row["duration_ms"] >= 0
    assert isinstance(row["observation"], str)


def test_a_long_result_is_clipped_where_the_budget_is_spent():
    # An 8000 TPM ceiling is spent by whatever enters the message history, so
    # trimming has to happen before that, not in the prompt builder.
    result = SearchResult(
        query="q",
        passages=[Passage(text="x" * 5000, source="s://1")],
    )
    rendered = result.render(max_chars=500)
    assert len(rendered) <= 500
    assert "truncated" in rendered


def test_an_empty_search_says_what_to_try_next():
    rendered = SearchResult(query="nothing").render()
    assert "No results" in rendered
    assert "without it" in rendered  # names the filter as the likely cause


def test_scalars_become_single_valued_lists():
    compiled = FilterArgs(department="finance", language="en").to_filter()
    assert compiled.department == ["finance"]
    assert compiled.language == ["en"]


def test_an_unset_field_stays_none_rather_than_an_empty_list():
    # "any department" and "no department" are different searches: a document
    # that never recorded the field does not match a filter on it.
    compiled = FilterArgs(department="finance").to_filter()
    assert compiled.region is None
    assert compiled.author is None


def test_an_empty_filter_constrains_nothing():
    assert FilterArgs().to_filter().is_empty()


def test_dates_pass_through_as_scalars():
    compiled = FilterArgs(date_from="2026-01-01", date_to="2026-03-31").to_filter()
    assert compiled.date_from == "2026-01-01"
    assert compiled.date_to == "2026-03-31"


def test_indexing_queues_rather_than_blocking(monkeypatch):
    # Chunking, embedding and storing a document takes tens of seconds. A tool
    # call that waited for it would only add a way to time out.
    captured = {}

    class _Queued:
        id = "task-123"

    def fake_delay(text, **kwargs):
        captured["text"] = text
        captured.update(kwargs)
        return _Queued()

    import tasks

    monkeypatch.setattr(tasks.index_document, "delay", fake_delay)

    call = invoke(
        "index_document",
        {"text": "Acme acquired Beta.", "source": "memo.txt", "department": "legal"},
        allowed=ALL_EFFECTS,
    )

    assert call.ok
    assert "task-123" in call.observation
    assert captured["source"] == "memo.txt"
    assert captured["metadata"] == {"department": "legal"}
    assert captured["build_graph"] is False


def test_blank_provenance_is_left_out_rather_than_stored_empty(monkeypatch):
    # An empty department is not a department. Storing "" would make the value
    # show up in corpus_profile's filter list as a real thing to filter on.
    captured = {}

    class _Queued:
        id = "task-456"

    monkeypatch.setattr(
        __import__("tasks").index_document,
        "delay",
        lambda text, **kw: (captured.update(kw), _Queued())[1],
    )

    invoke(
        "index_document",
        {"text": "x", "source": "s", "department": "   ", "region": ""},
        allowed=ALL_EFFECTS,
    )
    assert captured["metadata"] is None


def test_indexing_needs_write_permission():
    call = invoke("index_document", {"text": "x", "source": "s"})
    assert not call.ok
    assert "write" in call.error.lower()


def _tools_with_a_limit():
    return [
        spec
        for spec in catalog(ALL_EFFECTS)
        if "limit" in spec.args_model.model_fields
    ]


@pytest.mark.parametrize("spec", _tools_with_a_limit(), ids=lambda s: s.name)
@pytest.mark.parametrize("word", ["all", "every", "unlimited"])
def test_every_limit_accepts_the_word_a_model_actually_sends(spec, word):
    """Across five phrasings of "list the acquisitions", every one sent
    ``limit: "all"``. The first fix mapped words to 100, which worked for the
    tools bounded at 100 and left the two bounded at 50 — search_corpus and
    answer_from_corpus, the most used of the lot — failing exactly as before.
    Walking the catalog rather than naming tools is what makes this catch the
    next field that disagrees about its ceiling.
    """
    required = {
        name: "Acme Corporation"
        for name, field in spec.args_model.model_fields.items()
        if field.is_required()
    }
    model = spec.args_model.model_validate({**required, "limit": word})

    maximum = next(
        meta.le for meta in spec.args_model.model_fields["limit"].metadata
        if hasattr(meta, "le")
    )
    assert model.limit == maximum


@pytest.mark.parametrize("spec", _tools_with_a_limit(), ids=lambda s: s.name)
def test_a_limit_that_is_not_a_number_or_a_word_is_still_refused(spec):
    # Tolerance for "all" must not become tolerance for anything.
    required = {
        name: "Acme Corporation"
        for name, field in spec.args_model.model_fields.items()
        if field.is_required()
    }
    with pytest.raises(Exception):
        spec.args_model.model_validate({**required, "limit": "quite a few"})
