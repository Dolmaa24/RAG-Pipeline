"""The star: roles, the sufficiency loop, and the verifier.

Phase 03 ended with a specific, reproducible defect — asked what Acme acquired
*and* what the revenue was, the loop found the acquisition and stopped, leaving
half the question answered while the other half sat in an indexed passage one
search away. Nothing ever asked whether what it had was enough.

These tests are about that question being asked. They run without a model:
the answerer is injected, so "insufficient, then sufficient" is a fixture rather
than something to hope for.
"""

from __future__ import annotations

import pytest

from pipeline.agents.budget import Budget
from pipeline.agents.roles import ACQUISITION, CORPUS, available
from pipeline.agents.supervisor import Supervisor, _leads_from, _merge
from pipeline.agents.tools import Effect
from pipeline.agents.verify import Verdict, annotate, split_sentences, verify
from pipeline.extract.llm.base import ToolRequest, ToolTurn


class Quiet:
    """A specialist backend that answers immediately, calling nothing."""

    name, model = "fake", "fake-1"

    def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
        return ToolTurn(text="nothing to add", backend=self.name, model=self.model)


class Reply:
    """A stand-in for answer_question, with sufficiency scripted per round."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.asked: list[str] = []

    def __call__(self, question):
        self.asked.append(question)
        return self.replies.pop(0) if self.replies else self.replies_default()

    @staticmethod
    def replies_default():
        return _answer("nothing further", sufficient=True)


class _Source:
    def __init__(self, text):
        self._text = text

    def to_dict(self):
        return {"number": 1, "text": self._text, "origin": "s://1"}


class _Answer:
    def __init__(self, answer, sufficient, sources):
        self.answer = answer
        self.sufficient = sufficient
        self.sources = sources
        self.cited = [1]


def _answer(text, *, sufficient, passages=("evidence",)):
    return _Answer(text, sufficient, [_Source(p) for p in passages])


def _supervisor(answerer, **kwargs):
    kwargs.setdefault("verify_answer", False)
    return Supervisor(backend=Quiet(), answerer=answerer, **kwargs)


# --------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------- #


def test_a_read_only_run_has_only_the_corpus_specialist():
    assert [role.name for role in available(Budget())] == ["corpus"]


def test_acquisition_appears_only_when_writing_is_budgeted():
    assert ACQUISITION in available(Budget(network_calls=1, write_calls=1))
    # Network alone is not enough: fetching without indexing changes nothing.
    assert ACQUISITION not in available(Budget(network_calls=1))


def test_a_role_holding_only_read_tools_is_not_that_role():
    # Acquisition keeps corpus_profile and poll_task, both READ. "Has some
    # permitted tool" would call it available on a read-only budget, where it
    # could poll tasks it can never start.
    assert ACQUISITION.catalog([Effect.READ])
    assert ACQUISITION not in available(Budget())


def test_a_specialist_sees_only_its_own_tools():
    names = {tool["name"] for tool in CORPUS.catalog([Effect.READ])}
    assert "search_corpus" in names
    assert "crawl_site" not in names
    assert "extract_url" not in names


# --------------------------------------------------------------------------- #
# The sufficiency loop — the point of the phase
# --------------------------------------------------------------------------- #


def test_a_sufficient_first_round_stops_there():
    answerer = Reply(_answer("Acme acquired Beta.", sufficient=True))
    result = _supervisor(answerer).investigate("what did Acme acquire?")

    assert result.sufficient
    assert result.rounds == 1
    assert result.stopped == "answered"


def test_an_insufficient_round_sends_the_specialist_out_again():
    """The defect phase 03 ended on, as a test."""
    answerer = Reply(
        _answer("Acme acquired Beta.", sufficient=False),
        _answer("Acme acquired Beta. Revenue was 42.5m.", sufficient=True),
    )
    result = _supervisor(answerer).investigate("what did Acme acquire, and revenue?")

    assert result.rounds == 2
    assert result.sufficient
    assert "42.5m" in result.answer


def test_rounds_are_capped_even_when_never_sufficient():
    answerer = Reply(*[_answer("partial", sufficient=False) for _ in range(10)])
    result = _supervisor(answerer, max_rounds=3).investigate("q")

    assert result.rounds == 3
    assert "3-round limit" in result.stopped
    assert not result.sufficient


def test_an_insufficient_answer_is_still_returned():
    # Half an answer beats none, as long as it is not labelled as whole.
    answerer = Reply(*[_answer("half of it", sufficient=False) for _ in range(5)])
    result = _supervisor(answerer, max_rounds=1).investigate("q")

    assert result.answer == "half of it"
    assert result.sufficient is False


def test_an_empty_question_costs_nothing():
    answerer = Reply()
    result = _supervisor(answerer).investigate("  ")

    assert result.stopped == "empty question"
    assert answerer.asked == []


# --------------------------------------------------------------------------- #
# Carrying leads between rounds
# --------------------------------------------------------------------------- #


def test_names_the_graph_surfaced_are_read_out_of_a_trace():
    leads = _leads_from(
        [{"kind": "tool", "ok": True,
          "observation": "(Acme Corporation)-[ACQUIRED]->(Beta Industries)"}]
    )
    assert leads == ["Acme Corporation", "Beta Industries"]


def test_a_failed_tool_contributes_no_leads():
    assert _leads_from([{"kind": "tool", "ok": False, "observation": "(Nope)"}]) == []


def test_leads_are_merged_without_duplicates():
    assert _merge(["Acme"], ["acme", "Beta"]) == ["Acme", "Beta"]


def test_the_second_round_is_told_what_the_first_found():
    """This is the mechanism that makes a second hop worth taking."""
    class Finder:
        name, model = "fake", "fake-1"

        def __init__(self):
            self.prompts = []
            self.first = True

        def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
            self.prompts.append(messages[-1].content)
            if self.first:
                self.first = False
                return ToolTurn(
                    calls=[ToolRequest("graph_neighbors", {"entity": "Acme"})],
                    backend="fake", model="fake-1",
                )
            return ToolTurn(text="done", backend="fake", model="fake-1")

    backend = Finder()
    answerer = Reply(
        _answer("partial", sufficient=False), _answer("complete", sufficient=True)
    )
    Supervisor(
        backend=backend, answerer=answerer, verify_answer=False
    ).investigate("what did Acme acquire?")

    # The user-role prompt opening the second specialist run carries the leads.
    assert any("worth searching for" in prompt for prompt in backend.prompts)


# --------------------------------------------------------------------------- #
# Acquisition is gated, not chosen
# --------------------------------------------------------------------------- #


def test_acquisition_never_runs_on_a_read_only_budget():
    answerer = Reply(*[_answer("no", sufficient=False) for _ in range(5)])
    result = _supervisor(answerer, max_rounds=3).investigate(
        "summarise https://example.com/report"
    )

    roles = [step["role"] for step in result.trace if step["kind"] == "specialist"]
    assert set(roles) == {"corpus"}


def test_acquisition_runs_when_budgeted_and_a_url_is_given():
    answerer = Reply(*[_answer("no", sufficient=False) for _ in range(5)])
    result = _supervisor(
        answerer,
        max_rounds=3,
        budget=Budget(network_calls=2, write_calls=2),
    ).investigate("summarise https://example.com/report")

    roles = [step["role"] for step in result.trace if step["kind"] == "specialist"]
    assert "acquisition" in roles


def test_the_same_url_is_not_fetched_twice():
    answerer = Reply(*[_answer("no", sufficient=False) for _ in range(9)])
    result = _supervisor(
        answerer, max_rounds=6, budget=Budget(network_calls=9, write_calls=9)
    ).investigate("summarise https://example.com/report")

    roles = [step["role"] for step in result.trace if step["kind"] == "specialist"]
    assert roles.count("acquisition") == 1


def test_acquisition_is_not_attempted_without_a_url():
    answerer = Reply(*[_answer("no", sufficient=False) for _ in range(5)])
    result = _supervisor(
        answerer, max_rounds=3, budget=Budget(network_calls=2, write_calls=2)
    ).investigate("what is our revenue?")

    roles = [step["role"] for step in result.trace if step["kind"] == "specialist"]
    assert "acquisition" not in roles


# --------------------------------------------------------------------------- #
# The verifier
# --------------------------------------------------------------------------- #


def test_decimals_and_abbreviations_do_not_split_a_sentence():
    # "42.5 million" split apart produces a fragment the verifier then flags,
    # which is a false alarm caused by the splitter.
    assert len(split_sentences("Revenue was 42.5 million dollars. It grew.")) == 2
    assert len(split_sentences("Acme Inc. acquired Beta.")) == 1


def test_an_unsupported_sentence_is_marked_in_place_not_removed():
    text = "Acme acquired Beta. Acme is in Berlin."
    marked = annotate(text, ["Acme is in Berlin."])

    assert "Acme acquired Beta." in marked
    assert "[unsupported]" in marked


def test_nothing_to_check_is_not_the_same_as_everything_checked_out():
    assert verify("", ["evidence"]).skipped
    assert verify("A claim.", []).skipped
    assert not verify("A claim.", []).clean


def test_a_verifier_that_cannot_run_says_so_rather_than_passing_the_answer():
    class Broken:
        def complete_json(self, **kwargs):
            raise RuntimeError("model down")

    verdict = verify("A claim.", ["evidence"], backend=Broken())

    assert verdict.skipped
    assert not verdict.clean
    assert "model down" in verdict.reason
    assert verdict.answer == "A claim."


def test_a_sentence_the_model_does_not_vouch_for_is_flagged():
    """Fail closed. The check asks which sentences ARE supported, and anything
    left off that list is flagged — including anything unreadable."""
    class Says:
        def __init__(self, data):
            self.data = data

        def complete_json(self, **kwargs):
            from pipeline.extract.llm.base import LLMResponse

            return LLMResponse(data=self.data, backend="f", model="m")

    verdict = verify("One. Two.", ["e"], backend=Says({"supported": [1]}))
    assert verdict.unsupported == ["Two."]

    everything = verify("One. Two.", ["e"], backend=Says({"supported": "nonsense"}))
    assert len(everything.unsupported) == 2


def test_out_of_range_vouches_are_ignored():
    class Says:
        def complete_json(self, **kwargs):
            from pipeline.extract.llm.base import LLMResponse

            return LLMResponse(data={"supported": [0, 7, "x", 1]}, backend="f", model="m")

    verdict = verify("One. Two.", ["e"], backend=Says())
    assert verdict.unsupported == ["Two."]


def test_the_verdict_serialises_for_an_api():
    payload = Verdict(answer="a", unsupported=["x"], checked=2).to_dict()
    assert payload["clean"] is False
    assert payload["checked"] == 2


# --------------------------------------------------------------------------- #
# The whole result
# --------------------------------------------------------------------------- #


def test_the_investigation_serialises_with_its_reasoning():
    answerer = Reply(_answer("Acme acquired Beta.", sufficient=True))
    payload = _supervisor(answerer).investigate("q").to_dict()

    assert payload["answer"] == "Acme acquired Beta."
    assert payload["sufficient"] is True
    assert payload["rounds"] == 1
    assert isinstance(payload["trace"], list)
    assert payload["sources"]
