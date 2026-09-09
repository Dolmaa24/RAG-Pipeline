"""Fitting a conversation into the window the model has.

The limit these are written against is not the one people reach for.
OLLAMA_NUM_PREDICT and GROQ_MAX_OUTPUT_TOKENS both cap the *reply*; replaying
fifty turns never touches them. What it exhausts is OLLAMA_NUM_CTX, which is
8192 for input and output together — so with the reply's 4096 reserved,
everything sent has to fit in about 4096, of which the system prompt and seven
tool schemas already take roughly 1300.

Hydration calls no model, which is why it can be tested like this at all:
deciding what fits and paying to summarise what does not are separate steps.
"""

from __future__ import annotations

import pytest

from config import config
from playground import context, store


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PLAYGROUND_DB_PATH", str(tmp_path / "pg.db"))
    store.reset_cache()
    yield
    store.reset_cache()


@pytest.fixture
def thread(db):
    return store.create_thread("test")


def exchange(thread_id: str, n: int, size: int = 20) -> None:
    store.append_message(thread_id, "user", f"Q{n} " + "word " * size)
    store.append_message(thread_id, "assistant", f"A{n} " + "word " * size)


# --- what fits --------------------------------------------------------------


def test_a_short_thread_replays_whole(thread):
    exchange(thread.id, 1)
    exchange(thread.id, 2)
    assert len(context.hydrate(thread.id).messages) == 4


def test_an_empty_thread_replays_nothing(thread):
    hydrated = context.hydrate(thread.id)
    assert hydrated.messages == []
    assert hydrated.as_messages() == []


def test_a_long_thread_is_cut_to_the_budget(thread):
    for i in range(30):
        exchange(thread.id, i)
    hydrated = context.hydrate(thread.id, budget=300)
    assert hydrated.tokens <= 300
    assert len(hydrated.messages) < 60


def test_the_newest_turn_is_always_present(thread):
    """A turn that answered without seeing the message before it is worse than
    one that answered with no history at all — at least that one knows."""
    for i in range(20):
        exchange(thread.id, i)
    hydrated = context.hydrate(thread.id, budget=10)
    assert hydrated.messages
    assert hydrated.messages[-1].content.startswith("A19")


def test_history_comes_back_oldest_first(thread):
    exchange(thread.id, 1)
    exchange(thread.id, 2)
    contents = [m.content[:2] for m in context.hydrate(thread.id).messages]
    assert contents == ["Q1", "A1", "Q2", "A2"]


def test_roles_survive(thread):
    exchange(thread.id, 1)
    assert [m.role for m in context.hydrate(thread.id).messages] == ["user", "assistant"]


def test_tool_messages_are_never_replayed(thread):
    """The bulkiest thing in a thread, already acted on, and replaying it
    invites the model to read yesterday's search results as current."""
    store.append_message(thread.id, "user", "find something")
    store.append_message(thread.id, "tool", "search_corpus -> " + "x" * 5000)
    store.append_message(thread.id, "assistant", "found it")

    hydrated = context.hydrate(thread.id)
    assert [m.role for m in hydrated.messages] == ["user", "assistant"]


def test_stored_system_messages_are_not_replayed(thread):
    """The role supplies its own system prompt; a stored one is a note about
    the thread, and two system messages disagreeing is worse than one."""
    store.append_message(thread.id, "system", "a note")
    store.append_message(thread.id, "user", "hello")
    assert [m.role for m in context.hydrate(thread.id).messages] == ["user"]


# --- the token estimate -----------------------------------------------------


def test_the_estimate_is_roughly_four_characters_a_token():
    assert context.estimate_tokens("x" * 400) == 100


def test_the_estimate_rounds_up():
    """Erring toward sending less history costs a little recall. Erring the
    other way overflows the window and costs the whole turn."""
    assert context.estimate_tokens("x") == 1
    assert context.estimate_tokens("") == 0


# --- eviction and the summary ----------------------------------------------


def test_what_does_not_fit_is_reported_as_evicted(thread):
    for i in range(20):
        exchange(thread.id, i)
    hydrated = context.hydrate(thread.id, budget=200)
    assert hydrated.evicted
    assert hydrated.needs_summary


def test_a_thread_that_fits_evicts_nothing(thread):
    exchange(thread.id, 1)
    hydrated = context.hydrate(thread.id)
    assert hydrated.evicted == []
    assert not hydrated.needs_summary


class Summariser:
    """A backend that returns a fixed summary and counts how often it is asked."""

    name, model = "stub", "stub-1"

    def __init__(self, text: str = "They discussed the corpus."):
        self.text = text
        self.calls = 0
        self.last_content = ""

    def complete_json(self, *, prompt, content, schema_hint, json_schema):
        from types import SimpleNamespace

        self.calls += 1
        self.last_content = content
        return SimpleNamespace(data={"summary": self.text}, backend="stub", model="stub")


def test_eviction_produces_a_summary(thread):
    for i in range(20):
        exchange(thread.id, i)
    hydrated = context.hydrate(thread.id, budget=200)
    backend = Summariser()

    assert context.summarise(thread.id, hydrated, backend=backend) == "They discussed the corpus."
    assert backend.calls == 1
    assert store.get_thread(thread.id).summary == "They discussed the corpus."


def test_the_marker_advances_so_the_same_turns_are_not_paid_for_twice(thread):
    for i in range(20):
        exchange(thread.id, i)
    hydrated = context.hydrate(thread.id, budget=200)
    context.summarise(thread.id, hydrated, backend=Summariser())

    assert store.get_thread(thread.id).summarised_through == hydrated.evicted[-1].id


def test_a_second_turn_does_not_re_summarise(thread):
    """The whole reason summarising happens on eviction rather than per turn:
    on an 8000 tokens-a-minute budget a summary call on every message competes
    with the answer it is meant to support."""
    for i in range(20):
        exchange(thread.id, i)
    first = context.hydrate(thread.id, budget=400)
    context.summarise(thread.id, first, backend=Summariser())

    again = context.hydrate(thread.id, budget=400)
    backend = Summariser()
    context.summarise(thread.id, again, backend=backend)
    assert backend.calls == 0


def test_a_thread_that_never_overflows_never_summarises(thread):
    exchange(thread.id, 1)
    backend = Summariser()
    context.summarise(thread.id, context.hydrate(thread.id), backend=backend)
    assert backend.calls == 0


def test_an_existing_summary_is_folded_into_the_next_one(thread):
    """Otherwise the second eviction forgets the first."""
    for i in range(30):
        exchange(thread.id, i)
    first = context.hydrate(thread.id, budget=200)
    context.summarise(thread.id, first, backend=Summariser("The first part."))

    for i in range(30, 45):
        exchange(thread.id, i)
    second = context.hydrate(thread.id, budget=200)
    backend = Summariser("Everything so far.")
    context.summarise(thread.id, second, backend=backend)

    assert "The first part." in backend.last_content


def test_the_summary_is_replayed_as_a_system_message(thread):
    exchange(thread.id, 1)
    store.set_summary(thread.id, "They discussed the corpus.", 0)
    replayed = context.hydrate(thread.id).as_messages()

    assert replayed[0].role == "system"
    assert "They discussed the corpus." in replayed[0].content
    assert replayed[1].role == "user"


def test_no_summary_means_no_extra_message(thread):
    exchange(thread.id, 1)
    assert all(m.role != "system" for m in context.hydrate(thread.id).as_messages())


def test_a_failed_summary_keeps_the_old_one_and_does_not_fail_the_turn(thread):
    """Losing the beginning of a long thread is a worse conversation. Raising
    is a worse product."""
    class Broken:
        name = model = "broken"

        def complete_json(self, **kwargs):
            raise RuntimeError("the model is down")

    for i in range(20):
        exchange(thread.id, i)
    store.set_summary(thread.id, "The old summary.", 1)
    hydrated = context.hydrate(thread.id, budget=200)

    assert context.summarise(thread.id, hydrated, backend=Broken()) == "The old summary."


def test_turning_summarising_off_still_advances_the_marker(thread, monkeypatch):
    """Otherwise every turn re-evicts the same messages and the thread is
    permanently 'needs a summary'."""
    monkeypatch.setattr(config, "PLAYGROUND_SUMMARISE", False)
    for i in range(20):
        exchange(thread.id, i)
    hydrated = context.hydrate(thread.id, budget=200)
    backend = Summariser()

    context.summarise(thread.id, hydrated, backend=backend)
    assert backend.calls == 0
    assert store.get_thread(thread.id).summarised_through == hydrated.evicted[-1].id


def test_summarised_turns_are_not_replayed_again(thread):
    """They are in the summary. Sending both is paying twice for one thing."""
    for i in range(20):
        exchange(thread.id, i)
    first = context.hydrate(thread.id, budget=300)
    context.summarise(thread.id, first, backend=Summariser())

    after = context.hydrate(thread.id, budget=300)
    covered = {row.id for row in first.evicted}
    assert not covered & {id(m) for m in after.messages}
    assert len(after.messages) <= len(first.messages)


# --- the fast path ----------------------------------------------------------


def test_history_renders_as_text_for_the_answer_path(thread):
    """answer_question takes a question string and no messages, so this is the
    only shape it accepts."""
    exchange(thread.id, 1)
    text = context.hydrate(thread.id).as_text()
    assert "User:" in text and "You:" in text


def test_the_text_form_includes_the_summary(thread):
    exchange(thread.id, 1)
    store.set_summary(thread.id, "Earlier they discussed pricing.", 0)
    assert "Earlier they discussed pricing." in context.hydrate(thread.id).as_text()


def test_the_text_form_is_only_the_tail(thread):
    for i in range(10):
        exchange(thread.id, i)
    text = context.hydrate(thread.id).as_text(limit=2)
    assert "Q9" in text and "Q0" not in text
