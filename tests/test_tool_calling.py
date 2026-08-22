"""The missing primitive: a model that can call a tool and read the result.

``complete_json`` answers one question and forgets it. An agent needs a
conversation — call this, here is what it returned, now what — and until this
existed there was no way to express that. Everything in the agent plan past the
MCP server depends on it.

The wire-format tests matter more than they look. Ollama and Groq disagree about
how a tool result is addressed: Ollama matches on the tool's *name*, Groq on a
call *id* it issued. A history built for one and sent to the other is silently
wrong — the model sees results attached to nothing — so the neutral Message
carries both and each backend renders its own.
"""

from __future__ import annotations

import json

import pytest

from pipeline.extract.llm import base
from pipeline.extract.llm.base import Message, ToolRequest, ToolTurn
from pipeline.extract.llm.groq import _as_groq_tool, _parse_arguments
from pipeline.extract.llm.groq import _wire as groq_wire
from pipeline.extract.llm.ollama import _arguments_of, _as_ollama_tool
from pipeline.extract.llm.ollama import _wire as ollama_wire
from pipeline.extract.llm.toolshim import (
    ShimmedBackend,
    _turn_from,
    render_conversation,
    render_tools,
    shim_if_needed,
)

TOOLS = [
    {
        "name": "search_corpus",
        "description": "Find raw passages matching a query.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
            "required": ["query"],
        },
    },
    {"name": "corpus_profile", "description": "What the corpus holds.", "input_schema": {}},
]


# --------------------------------------------------------------------------- #
# The turn
# --------------------------------------------------------------------------- #


def test_a_turn_is_calls_or_text_never_both():
    calling = ToolTurn(calls=[ToolRequest("search_corpus", {"query": "x"})])
    answering = ToolTurn(text="here it is")

    assert calling.wants_tools and not calling.text
    assert not answering.wants_tools and answering.text


def test_a_turn_can_be_appended_to_the_history_it_came_from():
    turn = ToolTurn(calls=[ToolRequest("search_corpus", {"query": "x"})], text="")
    message = turn.as_message()
    assert message.role == "assistant"
    assert [c.name for c in message.tool_calls] == ["search_corpus"]


def test_an_observation_is_addressed_to_the_call_that_asked_for_it():
    request = ToolRequest("search_corpus", {"query": "x"}, call_id="call_7")
    message = Message.observation(request, "one passage")
    assert message.role == "tool"
    assert message.tool_call_id == "call_7"
    assert message.tool_name == "search_corpus"


# --------------------------------------------------------------------------- #
# Wire formats
# --------------------------------------------------------------------------- #


def test_ollama_addresses_a_tool_result_by_name():
    request = ToolRequest("search_corpus", {"query": "x"}, call_id="call_7")
    wire = ollama_wire(Message.observation(request, "passages"))

    assert wire["role"] == "tool"
    assert wire["tool_name"] == "search_corpus"
    # Ollama issues no call id and ignores one; sending it would be noise.
    assert "tool_call_id" not in wire


def test_groq_addresses_a_tool_result_by_id():
    request = ToolRequest("search_corpus", {"query": "x"}, call_id="call_7")
    wire = groq_wire(Message.observation(request, "passages"))

    assert wire["tool_call_id"] == "call_7"
    assert "tool_name" not in wire


def test_groq_serialises_tool_call_arguments_as_a_string():
    # OpenAI-compatible APIs take arguments as JSON text; Ollama takes an
    # object. Sending the wrong one is accepted and then misparsed.
    wire = groq_wire(
        Message(role="assistant", tool_calls=[ToolRequest("search_corpus", {"query": "x"})])
    )
    arguments = wire["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"query": "x"}


def test_ollama_sends_tool_call_arguments_as_an_object():
    wire = ollama_wire(
        Message(role="assistant", tool_calls=[ToolRequest("search_corpus", {"query": "x"})])
    )
    assert wire["tool_calls"][0]["function"]["arguments"] == {"query": "x"}


def test_groq_invents_a_call_id_when_the_history_has_none():
    # A history that began on Ollama has no ids. Groq rejects a tool_call
    # without one, so failing over must not produce an unsendable message.
    wire = groq_wire(
        Message(role="assistant", tool_calls=[ToolRequest("search_corpus", {})])
    )
    assert wire["tool_calls"][0]["id"]


@pytest.mark.parametrize("as_tool", [_as_ollama_tool, _as_groq_tool])
def test_both_backends_publish_the_registry_schema_unchanged(as_tool):
    wire = as_tool(TOOLS[0])
    assert wire["function"]["name"] == "search_corpus"
    assert wire["function"]["parameters"] == TOOLS[0]["input_schema"]


@pytest.mark.parametrize("as_tool", [_as_ollama_tool, _as_groq_tool])
def test_a_tool_without_arguments_still_publishes_an_object_schema(as_tool):
    # corpus_profile takes nothing. An empty schema is not the same as a
    # missing one, and some APIs reject the latter.
    assert as_tool(TOOLS[1])["function"]["parameters"]["type"] == "object"


# --------------------------------------------------------------------------- #
# Arguments arrive in more than one shape
# --------------------------------------------------------------------------- #


def _ollama_args(raw):
    return _arguments_of({"function": {"arguments": raw}})


@pytest.mark.parametrize("parse", [_ollama_args, _parse_arguments])
def test_arguments_are_a_dict_however_they_arrived(parse):
    assert parse({"query": "x"}) == {"query": "x"}
    assert parse('{"query": "x"}') == {"query": "x"}


@pytest.mark.parametrize("parse", [_ollama_args, _parse_arguments])
def test_unparseable_arguments_are_handed_on_not_raised(parse):
    # The model's mistake to see and correct. Raising here would end the run
    # instead of letting validation explain the problem in terms it can act on.
    result = parse("{not json")
    assert "__unparseable__" in result


# --------------------------------------------------------------------------- #
# The shim
# --------------------------------------------------------------------------- #


def test_the_catalog_leads_with_what_a_tool_is_for():
    """Ordering, not formatting.

    The first rendering put the signature first — nine optional filters ahead of
    the one required argument — and against it qwen2.5:3b said the tools did not
    include anything for answering questions about revenue, while looking
    straight at search_corpus.
    """
    rendered = render_tools(TOOLS)
    first = rendered.splitlines()[0]
    assert first.startswith("- search_corpus: Find raw passages")
    assert "required: query (string)" in rendered
    assert "optional: limit" in rendered


def test_an_optional_field_reports_its_real_type_not_anyof():
    assert "limit" not in render_tools(TOOLS).split("required:")[1].split("\n")[0]


def test_the_conversation_labels_results_by_the_tool_that_ran():
    request = ToolRequest("search_corpus", {"query": "x"})
    rendered = render_conversation(
        [
            Message.user("what about revenue?"),
            Message(role="assistant", tool_calls=[request]),
            Message.observation(request, "42.5 million"),
        ]
    )
    assert "[you called] search_corpus" in rendered
    assert "[result of search_corpus]" in rendered
    assert "42.5 million" in rendered


def test_the_shim_reads_a_chosen_tool():
    turn = _turn_from({"tool": "search_corpus", "arguments": {"query": "x"}}, TOOLS)
    assert [c.name for c in turn.calls] == ["search_corpus"]


def test_the_shim_reads_an_answer():
    turn = _turn_from({"tool": None, "arguments": None, "answer": "here it is"}, TOOLS)
    assert not turn.wants_tools
    assert turn.text == "here it is"


def test_the_shim_does_not_turn_an_invented_name_into_a_call():
    # Unlike a native tool call, this name came out of free-form generation,
    # where invention is likelier. The registry would refuse it anyway; spending
    # a turn on a refusal for something detectable here is waste.
    turn = _turn_from({"tool": "delete_everything", "arguments": {}}, TOOLS)
    assert not turn.wants_tools
    assert "delete_everything" in turn.text
    assert "search_corpus" in turn.text
    assert turn.finish_reason == "unknown_tool"


def test_the_shim_survives_arguments_that_are_not_an_object():
    turn = _turn_from({"tool": "search_corpus", "arguments": "query=x"}, TOOLS)
    assert turn.calls[0].arguments == {}


def test_a_shimmed_turn_is_marked_not_native(monkeypatch):
    # A trace that cannot tell a real tool call from a parsed one cannot
    # explain why one model behaves worse than another.
    class FakeInner:
        name = "fake"
        model = "fake-1"

        def complete_json(self, **kwargs):
            return base.LLMResponse(
                data={"tool": "search_corpus", "arguments": {"query": "x"}},
                backend="fake",
                model="fake-1",
            )

    turn = ShimmedBackend(FakeInner()).complete_with_tools(
        messages=[Message.user("q")], tools=TOOLS
    )
    assert turn.native is False
    assert turn.backend == "fake+shim"
    assert [c.name for c in turn.calls] == ["search_corpus"]


# --------------------------------------------------------------------------- #
# Choosing whether to shim
# --------------------------------------------------------------------------- #


class _Native:
    name = "native"
    model = "m"

    def supports_tools(self):
        return True


class _NotNative:
    name = "old"
    model = "m"

    def supports_tools(self):
        return False


class _Raises:
    name = "broken"
    model = "m"

    def supports_tools(self):
        raise RuntimeError("probe exploded")


def test_a_native_backend_is_left_alone():
    backend = _Native()
    assert shim_if_needed(backend) is backend


def test_a_backend_without_tools_is_wrapped_rather_than_rejected():
    assert isinstance(shim_if_needed(_NotNative()), ShimmedBackend)


def test_a_failed_capability_probe_shims_rather_than_failing_the_run():
    assert isinstance(shim_if_needed(_Raises()), ShimmedBackend)


def test_a_backend_missing_the_method_entirely_is_wrapped():
    class Ancient:
        name = "ancient"
        model = "m"

    assert isinstance(shim_if_needed(Ancient()), ShimmedBackend)


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def test_the_agent_role_can_point_somewhere_other_than_extraction(monkeypatch):
    from pipeline.extract import llm

    monkeypatch.setattr(llm.config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(llm.config, "LLM_AGENT_BACKEND", "groq")
    assert llm._configured(llm.AGENT) == "groq"
    assert llm._configured(llm.BULK) == "ollama"


def test_the_agent_role_follows_the_main_backend_when_unset(monkeypatch):
    from pipeline.extract import llm

    monkeypatch.setattr(llm.config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(llm.config, "LLM_AGENT_BACKEND", None)
    assert llm._configured(llm.AGENT) == "ollama"


def test_a_missing_agent_model_falls_back_rather_than_failing(monkeypatch):
    """The same principle as an unavailable interactive backend.

    A model that is not pulled should make the loop worse, not broken — and the
    warning names the pull command, because that is the fix.
    """
    from pipeline.extract import llm

    class FakeOllama:
        name = "ollama"
        model = "qwen2.5:3b"

        def __init__(self, model=None):
            self.model = model or "qwen2.5:3b"

        def available(self):
            return self.model == "qwen2.5:3b"

        def supports_tools(self):
            return True

    monkeypatch.setattr(llm, "OllamaBackend", FakeOllama)
    monkeypatch.setattr(llm, "get_backend", lambda **kwargs: FakeOllama())
    monkeypatch.setattr(llm.config, "AGENT_MODEL_NAME", "not-pulled:70b")

    assert llm.get_agent_backend().model == "qwen2.5:3b"


# --------------------------------------------------------------------------- #
# Against a real model
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_a_real_model_calls_a_tool_and_reads_the_result():
    from pipeline.extract.llm.ollama import OllamaBackend

    backend = OllamaBackend(model="llama3.2:3b")
    if not backend.available():
        pytest.skip("llama3.2:3b is not pulled")

    first = backend.complete_with_tools(
        messages=[
            Message.system("You answer questions using the tools provided."),
            Message.user("What do the documents say about quarterly revenue?"),
        ],
        tools=TOOLS,
    )
    assert first.wants_tools
    assert first.native

    second = backend.complete_with_tools(
        messages=[
            Message.system("You answer questions using the tools provided."),
            Message.user("What do the documents say about quarterly revenue?"),
            first.as_message(),
            Message.observation(first.calls[0], "[1] Revenue was 42.5 million dollars."),
        ],
        tools=TOOLS,
    )
    # It may answer or search again; what must hold is that the history with a
    # tool result in it was accepted rather than rejected as malformed.
    assert second.calls or second.text
