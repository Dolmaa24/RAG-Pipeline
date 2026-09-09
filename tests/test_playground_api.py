"""The thread endpoints, and the seam that carries history into the agent.

The endpoints are a transport, so these check what a transport loses: a message
that was not stored before the slow part began, a permission that changed shape
on the way through, an unknown id that became a 500 instead of the 404 it was.

Nothing here runs a model or a broker.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app as api
from config import config
from playground import store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PLAYGROUND_DB_PATH", str(tmp_path / "pg.db"))
    store.reset_cache()
    yield TestClient(api.app)
    store.reset_cache()


@pytest.fixture
def queued(monkeypatch):
    seen: dict = {}

    class _Task:
        id = "reply-1"

    import tasks

    def fake_delay(thread_id, question, **kwargs):
        seen["thread_id"] = thread_id
        seen["question"] = question
        seen.update(kwargs)
        return _Task()

    monkeypatch.setattr(tasks.playground_reply, "delay", fake_delay)
    return seen


# --- creating ---------------------------------------------------------------


def test_a_thread_is_created(client):
    body = client.post("/api/v1/threads", json={}).json()
    assert len(body["thread"]["id"]) == 32
    assert body["thread"]["message_count"] == 0


def test_creating_with_a_message_takes_the_first_turn(client, queued):
    """What typing into an empty chat actually is: one call, not two."""
    body = client.post(
        "/api/v1/threads", json={"message": "What does the corpus contain?"}
    ).json()

    assert body["task_id"] == "reply-1"
    assert body["thread"]["message_count"] == 1
    assert body["thread"]["title"] == "What does the corpus contain?"
    assert queued["question"] == "What does the corpus contain?"


def test_creating_without_a_message_queues_nothing(client, queued):
    client.post("/api/v1/threads", json={"title": "empty"})
    assert queued == {}


# --- listing and reading ----------------------------------------------------


def test_threads_are_listed_newest_activity_first(client, queued):
    first = client.post("/api/v1/threads", json={"title": "first"}).json()["thread"]
    client.post("/api/v1/threads", json={"title": "second"})
    client.post(f"/api/v1/threads/{first['id']}/messages", json={"content": "bump"})

    listed = client.get("/api/v1/threads").json()
    assert listed["threads"][0]["id"] == first["id"]
    assert listed["count"] == 2


def test_a_thread_reads_back_with_its_messages(client, queued):
    made = client.post("/api/v1/threads", json={"message": "hello"}).json()
    body = client.get(f"/api/v1/threads/{made['thread']['id']}").json()

    assert body["thread"]["id"] == made["thread"]["id"]
    assert [m["role"] for m in body["messages"]] == ["user"]
    assert body["messages"][0]["content"] == "hello"


def test_an_unknown_thread_is_a_404(client):
    assert client.get("/api/v1/threads/nope").status_code == 404


def test_listing_pages(client):
    for i in range(4):
        client.post("/api/v1/threads", json={"title": f"t{i}"})
    assert len(client.get("/api/v1/threads", params={"limit": 2}).json()["threads"]) == 2


# --- posting a message ------------------------------------------------------


def test_a_message_is_stored_before_the_task_is_queued(client, queued):
    """The one that matters. A message that appears only when the agent
    finishes is one the user cannot see they sent — and if the worker then
    fails, there is no trace it was ever asked."""
    thread = client.post("/api/v1/threads", json={}).json()["thread"]
    client.post(f"/api/v1/threads/{thread['id']}/messages", json={"content": "a question"})

    stored = client.get(f"/api/v1/threads/{thread['id']}").json()["messages"]
    assert [m["content"] for m in stored] == ["a question"]


def test_posting_returns_a_task_to_poll(client, queued):
    thread = client.post("/api/v1/threads", json={}).json()["thread"]
    body = client.post(
        f"/api/v1/threads/{thread['id']}/messages", json={"content": "hello"}
    ).json()

    assert body["task_id"] == "reply-1"
    assert body["poll"] == f"/api/v1/threads/{thread['id']}/replies/reply-1"


def test_the_route_is_named_before_the_work_starts(client, queued):
    """So the UI can say "thinking" or "this takes a minute" rather than
    showing the same spinner for a five-second answer and a two-minute loop."""
    thread = client.post("/api/v1/threads", json={}).json()["thread"]

    fast = client.post(
        f"/api/v1/threads/{thread['id']}/messages", json={"content": "what is the revenue"}
    ).json()
    assert fast["path"] == "answer"

    store.append_message(thread["id"], "assistant", "…")
    slow = client.post(
        f"/api/v1/threads/{thread['id']}/messages",
        json={"content": "how many suppliers are there"},
    ).json()
    assert slow["path"] == "investigate"


def test_a_message_to_an_unknown_thread_is_a_404(client, queued):
    assert client.post(
        "/api/v1/threads/nope/messages", json={"content": "hello"}
    ).status_code == 404


def test_an_empty_message_is_refused(client, queued):
    thread = client.post("/api/v1/threads", json={}).json()["thread"]
    assert client.post(
        f"/api/v1/threads/{thread['id']}/messages", json={"content": ""}
    ).status_code == 422


def test_a_second_message_while_one_is_in_flight_is_a_409(client, queued):
    """Two agents appending to one history produce a transcript neither was
    answering."""
    thread = client.post("/api/v1/threads", json={"message": "first"}).json()["thread"]
    second = client.post(
        f"/api/v1/threads/{thread['id']}/messages", json={"content": "second"}
    )
    assert second.status_code == 409


def test_the_next_message_is_allowed_once_the_reply_lands(client, queued):
    thread = client.post("/api/v1/threads", json={"message": "first"}).json()["thread"]
    store.append_message(thread["id"], "assistant", "an answer")
    assert client.post(
        f"/api/v1/threads/{thread['id']}/messages", json={"content": "second"}
    ).status_code == 202


def test_local_only_reaches_the_task(client, queued):
    thread = client.post("/api/v1/threads", json={}).json()["thread"]
    client.post(
        f"/api/v1/threads/{thread['id']}/messages",
        json={"content": "hello", "local_only": True},
    )
    assert queued["local_only"] is True


# --- deleting ---------------------------------------------------------------


def test_deleting_removes_the_thread_and_its_messages(client, queued):
    thread = client.post("/api/v1/threads", json={"message": "hello"}).json()["thread"]
    body = client.delete(f"/api/v1/threads/{thread['id']}").json()

    assert body["messages_removed"] == 1
    assert client.get(f"/api/v1/threads/{thread['id']}").status_code == 404


def test_deleting_an_unknown_thread_is_a_404(client):
    assert client.delete("/api/v1/threads/nope").status_code == 404


# --- the hydration seam -----------------------------------------------------


def test_a_loop_with_no_history_is_unchanged():
    """The regression that matters: every existing caller passes nothing."""
    from pipeline.agents.loop import AgentLoop
    from pipeline.extract.llm.base import ToolTurn

    seen: dict = {}

    class Spy:
        name = model = "spy"

        def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
            seen.setdefault("messages", messages)
            return ToolTurn(text="ok", backend="spy", model="spy")

    AgentLoop(backend=Spy(), system="SYS").run("a question")
    assert [(m.role, m.content) for m in seen["messages"]] == [
        ("system", "SYS"),
        ("user", "a question"),
    ]


def test_history_sits_between_the_prompt_and_the_question():
    """The role's instructions still lead and the new question is still last —
    the two positions a model weights most."""
    from pipeline.agents.loop import AgentLoop
    from pipeline.extract.llm.base import Message, ToolTurn

    seen: dict = {}

    class Spy:
        name = model = "spy"

        def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
            seen.setdefault("messages", messages)
            return ToolTurn(text="ok", backend="spy", model="spy")

    AgentLoop(backend=Spy(), system="SYS").run(
        "the new one",
        history=[Message.user("earlier"), Message(role="assistant", content="replied")],
    )
    assert [(m.role, m.content) for m in seen["messages"]] == [
        ("system", "SYS"),
        ("user", "earlier"),
        ("assistant", "replied"),
        ("user", "the new one"),
    ]


def test_the_supervisor_hands_history_to_its_specialist():
    from pipeline.agents.supervisor import Supervisor
    from pipeline.extract.llm.base import Message, ToolTurn

    seen: dict = {}

    class Quiet:
        name = model = "quiet"

        def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
            seen.setdefault("messages", messages)
            return ToolTurn(text="nothing to add", backend="quiet", model="quiet")

    class Answer:
        answer, sufficient, cited = "done", True, []
        sources: list = []

    Supervisor(
        backend=Quiet(),
        answerer=lambda q: Answer(),
        verify_answer=False,
        history=[Message.user("earlier")],
    ).investigate("the new one")

    assert any(m.content == "earlier" for m in seen["messages"])


# --- what decides a turn ----------------------------------------------------


def test_the_breakdown_carries_both_decisions():
    """Path and domain are read together — "answered directly, as the insurance
    specialist" is the sentence, and either half alone explains nothing."""
    from playground.threads import breakdown

    decided = breakdown("which policies cover physiotherapy")
    assert decided["path"] == "investigate"
    assert decided["skill"] == "insurance"
    assert decided["skill_how"] == "trigger"
    assert "corpus_profile" in decided["tools"]


def test_the_breakdown_names_the_declared_roster():
    """What the side panel shows as agents created, as distinct from run."""
    from playground.threads import breakdown

    decided = breakdown("the exam timetable for this term")
    assert decided["buildable"] is True
    assert [a["name"] for a in decided["agents"]] == [
        "registrar", "timetable", "attendance", "examiner"
    ]


def test_an_unmatched_question_still_produces_a_breakdown():
    """No skill is a normal outcome, not a missing one."""
    from playground.threads import breakdown

    decided = breakdown("what is the revenue", allow_embedding=False)
    assert decided["skill"] is None
    assert decided["path"] == "answer"


def test_the_request_path_never_embeds(client, queued, monkeypatch):
    """The security-adjacent one for latency: an HTTP handler loading a 130 MB
    model puts ten seconds in front of the first message anybody sends."""
    from pipeline.embed import dense

    def explode(*args, **kwargs):
        raise AssertionError("the API loaded the embedder")

    monkeypatch.setattr(dense, "get_dense_embedder", explode)
    thread = client.post("/api/v1/threads", json={}).json()["thread"]
    response = client.post(
        f"/api/v1/threads/{thread['id']}/messages",
        json={"content": "something with no trigger word at all"},
    )
    assert response.status_code == 202
    assert response.json()["breakdown"]["skill_how"] == "undecided"


def test_a_trigger_match_is_reported_immediately(client, queued):
    """Free, so there is no reason to make the caller wait for it."""
    thread = client.post("/api/v1/threads", json={}).json()["thread"]
    body = client.post(
        f"/api/v1/threads/{thread['id']}/messages",
        json={"content": "which policies cover physiotherapy"},
    ).json()
    assert body["breakdown"]["skill"] == "insurance"


def test_the_matched_skill_composes_the_specialist(tmp_path, monkeypatch):
    """The integration. Without this the Playground is skill-blind and the
    panel has nothing to show."""
    monkeypatch.setattr(config, "PLAYGROUND_DB_PATH", str(tmp_path / "pg.db"))
    store.reset_cache()

    from playground import threads

    seen: dict = {}

    class FakeSupervisor:
        def investigate(self, question):
            class R:
                answer, sufficient = "done", True
                sources: list = []
                trace = [{"kind": "specialist", "role": "insurance"}]
                warnings: list = []

            return R()

    thread = store.create_thread()
    store.append_message(thread.id, "user", "which policies cover physiotherapy")
    result = threads.reply(
        thread.id, "which policies cover physiotherapy", supervisor=FakeSupervisor()
    )

    assert result.message.meta["skill"] == "insurance"
    assert result.message.meta["ran"] == ["insurance"]
    store.reset_cache()
