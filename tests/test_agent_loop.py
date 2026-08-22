"""The loop, driven by a scripted backend.

Everything here runs without a model, because the behaviours that matter are
the ones that stop a run, and those must be provable rather than observed
occasionally. The tool-calling gate measured `qwen2.5:3b` stopping 0 times out
of 6 when no tool was needed; for that model every run ends by exhaustion, so
the exhaustion path is the *normal* path and not an edge case.
"""

from __future__ import annotations

import threading
import time

import pytest

from pipeline.agents.budget import Budget, Spend, would_exceed_effect_budget
from pipeline.agents.loop import AgentLoop, LoopResult
from pipeline.agents.tools import Effect
from pipeline.extract.llm.base import ToolRequest, ToolTurn


class Scripted:
    """A backend that returns prepared turns, then answers."""

    name = "fake"
    model = "fake-1"

    def __init__(self, *turns, final: str = "done") -> None:
        self.turns = list(turns)
        self.final = final
        self.calls: list[dict] = []

    def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
        self.calls.append({"messages": len(messages), "tools": [t["name"] for t in tools]})
        # Offered no tools, a real model answers. A fake that kept returning
        # queued tool calls here made the finishing turn look broken when it
        # was the fake that was.
        if tools and self.turns:
            return self.turns.pop(0)
        return ToolTurn(text=self.final, backend=self.name, model=self.model)


def _call(name: str, **arguments) -> ToolTurn:
    return ToolTurn(
        calls=[ToolRequest(name, arguments)], backend="fake", model="fake-1"
    )


def _answer(text: str) -> ToolTurn:
    return ToolTurn(text=text, backend="fake", model="fake-1")


@pytest.fixture
def varying_tool():
    """Make graph_neighbors return something new each call.

    Budget limits cannot be tested with a tool that repeats itself, because the
    no-progress stop is correct and fires first. That is the loop working, not
    a fixture detail: two identical observations is already enough evidence.
    """
    import itertools

    import pipeline.agents.tools.registry as registry
    from pipeline.agents.tools.models import GraphResult

    spec = registry.get("graph_neighbors")
    original = spec.handler
    counter = itertools.count()

    def handler(args):
        nth = next(counter)
        return GraphResult(seeds=[f"seed-{nth}"], edges=[f"A RELATES_TO B-{nth}"])

    object.__setattr__(spec, "handler", handler)
    try:
        yield
    finally:
        object.__setattr__(spec, "handler", original)


def _varying_calls(count: int) -> list[ToolTurn]:
    return [_call("graph_neighbors", entity=f"Entity{n}") for n in range(count)]


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_a_run_calls_a_tool_then_answers():
    backend = Scripted(_call("corpus_profile"), _answer("42 chunks."))
    result = AgentLoop(backend=backend).run("what is in the corpus?")

    assert result.answer == "42 chunks."
    assert result.stopped == "answered"
    assert result.tools_used == ["corpus_profile"]


def test_an_empty_question_costs_nothing():
    backend = Scripted()
    result = AgentLoop(backend=backend).run("   ")

    assert result.stopped == "empty question"
    assert backend.calls == []


def test_the_tool_result_is_fed_back_to_the_model():
    # The whole point of the phase. Without this the loop is a slower way of
    # asking one question.
    backend = Scripted(_call("corpus_profile"), _answer("done"))
    AgentLoop(backend=backend).run("q")

    # system + user, then system + user + assistant + tool.
    assert backend.calls[0]["messages"] == 2
    assert backend.calls[1]["messages"] == 4


# --------------------------------------------------------------------------- #
# What stops a run
# --------------------------------------------------------------------------- #


def test_a_model_that_never_stops_is_stopped_by_the_turn_limit(varying_tool):
    """The measured behaviour of qwen2.5:3b, not a hypothetical."""
    backend = Scripted(*_varying_calls(50))
    result = AgentLoop(backend=backend, budget=Budget(max_iterations=4)).run("q")

    assert "4-turn limit" in result.stopped
    assert result.spend.iterations <= 5  # the finishing turn is the fifth


def test_budget_exhaustion_still_produces_an_answer(varying_tool):
    # A run that spends a minute and returns nothing is worse than useless.
    backend = Scripted(*_varying_calls(10), final="Here is what I found.")
    result = AgentLoop(backend=backend, budget=Budget(max_iterations=3)).run("q")

    assert result.answer == "Here is what I found."
    assert result.trace[-1]["kind"] == "finish"


def test_the_finishing_turn_offers_no_tools(varying_tool):
    # Offering tools to a model that has just been cut off for calling too many
    # invites it to call one more.
    backend = Scripted(*_varying_calls(10))
    AgentLoop(backend=backend, budget=Budget(max_iterations=2)).run("q")

    assert backend.calls[-1]["tools"] == []


def test_the_tool_call_limit_stops_a_run(varying_tool):
    backend = Scripted(*_varying_calls(10))
    result = AgentLoop(
        backend=backend, budget=Budget(max_iterations=99, max_tool_calls=3)
    ).run("q")

    assert "3-tool-call limit" in result.stopped


def test_the_time_limit_stops_a_run(varying_tool):
    class Slow(Scripted):  # noqa: D401
        def complete_with_tools(self, **kwargs):
            time.sleep(0.05)
            return super().complete_with_tools(**kwargs)

    backend = Slow(*_varying_calls(50))
    result = AgentLoop(
        backend=backend, budget=Budget(max_iterations=99, max_seconds=0.1)
    ).run("q")

    assert "time limit" in result.stopped


def test_the_same_result_twice_ends_the_run():
    # Small models loop on the same failing search far more often than they
    # invent a tool. Two identical observations is enough evidence.
    backend = Scripted(*[_call("corpus_profile") for _ in range(10)])
    result = AgentLoop(backend=backend, budget=Budget(max_iterations=99)).run("q")

    assert result.stopped == "the same result came back twice"
    assert result.spend.tool_calls == 2


def test_different_results_do_not_trip_the_no_progress_stop():
    backend = Scripted(
        _call("graph_neighbors", entity="Acme"),
        _call("graph_neighbors", entity="Beta"),
        _answer("both checked"),
    )
    result = AgentLoop(backend=backend, budget=Budget(max_iterations=99)).run("q")

    assert result.stopped == "answered"


def test_a_failed_model_call_is_reported_as_a_failure_not_an_answer():
    class Broken:
        name, model = "broken", "m"

        def complete_with_tools(self, **kwargs):
            raise RuntimeError("connection refused")

    result = AgentLoop(backend=Broken()).run("q")

    assert result.stopped == "the model call failed"
    assert result.answer == ""
    assert any("connection refused" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# Effects
# --------------------------------------------------------------------------- #


def test_a_read_only_run_is_not_shown_the_tools_it_cannot_use():
    offered = {tool["name"] for tool in AgentLoop(backend=Scripted()).tools()}

    assert "search_corpus" in offered
    assert "extract_url" not in offered
    assert "index_document" not in offered


def test_budgeting_a_network_call_is_what_permits_it():
    # The counter is the grant. "May it?" and "how much?" cannot drift apart
    # when they are the same number.
    loop = AgentLoop(backend=Scripted(), budget=Budget(network_calls=2))
    offered = {tool["name"] for tool in loop.tools()}

    assert "detect_url" in offered
    assert "extract_url" not in offered  # network is not write


def test_a_call_beyond_its_effect_budget_is_refused_with_the_real_reason():
    spend = Spend(network_calls=1)
    reason = would_exceed_effect_budget(spend, Budget(network_calls=1), Effect.NETWORK)

    assert "no network calls left" in reason


# --------------------------------------------------------------------------- #
# Parallel dispatch
# --------------------------------------------------------------------------- #


def test_several_read_only_calls_in_one_turn_run_at_once():
    seen: list[str] = []
    barrier = threading.Barrier(2, timeout=5)

    def slow_tool(args):
        barrier.wait()  # deadlocks unless both run concurrently
        seen.append("ran")
        from pipeline.agents.tools.models import GraphResult

        return GraphResult(triples=[])

    import pipeline.agents.tools.registry as registry

    spec = registry.get("graph_neighbors")
    original = spec.handler
    object.__setattr__(spec, "handler", slow_tool)
    try:
        turn = ToolTurn(
            calls=[
                ToolRequest("graph_neighbors", {"entity": "Acme"}),
                ToolRequest("graph_neighbors", {"entity": "Beta"}),
            ],
            backend="fake",
            model="fake-1",
        )
        AgentLoop(backend=Scripted(turn)).run("q")
    finally:
        object.__setattr__(spec, "handler", original)

    assert len(seen) == 2


# --------------------------------------------------------------------------- #
# The trace
# --------------------------------------------------------------------------- #


def test_the_trace_records_every_step_with_what_it_cost():
    backend = Scripted(_call("corpus_profile"), _answer("done"))
    result = AgentLoop(backend=backend).run("q")

    kinds = [step["kind"] for step in result.trace]
    assert kinds == ["turn", "tool", "turn"]

    for step in result.trace:
        assert "step" in step
    assert result.trace[0]["budget_left"]["turns"] >= 0
    assert result.trace[1]["tool"] == "corpus_profile"


def test_the_trace_says_whether_a_call_was_native():
    # A trace that cannot tell a real tool call from one the shim parsed cannot
    # explain why one model behaves worse than another.
    shimmed = ToolTurn(
        calls=[ToolRequest("corpus_profile", {})], backend="fake+shim",
        model="m", native=False,
    )
    result = AgentLoop(backend=Scripted(shimmed)).run("q")

    assert result.trace[0]["native"] is False


def test_the_result_serialises_for_an_api():
    backend = Scripted(_call("corpus_profile"), _answer("done"))
    payload = AgentLoop(backend=backend).run("q").to_dict()

    assert payload["answer"] == "done"
    assert payload["cost"]["turns"] == 2
    assert payload["cost"]["tool_calls"] == 1
    assert isinstance(payload["trace"], list)


def test_a_result_with_no_run_still_serialises():
    assert LoopResult(question="q").to_dict()["cost"]["turns"] == 0
