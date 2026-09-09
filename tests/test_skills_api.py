"""The skills surface: listing, dry-running the routing, and queueing a task.

Nothing here runs an agent. The endpoints are a transport -- they match an
intent, name the skill in the reply, and hand the name to the queue -- and what
goes wrong in a transport is that something is lost on the way through. In
particular a permission: ``allow_write`` must reach the task as the caller sent
it, because a skill file is not allowed to supply one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app as api


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def queued(monkeypatch):
    """Capture what would have been enqueued, without a broker."""
    seen: dict = {}

    class _Task:
        id = "task-abc"

    import tasks

    def fake_delay(question, **kwargs):
        seen["question"] = question
        seen.update(kwargs)
        return _Task()

    monkeypatch.setattr(tasks.investigate, "delay", fake_delay)
    return seen


# --- listing ---------------------------------------------------------------


def test_the_shipped_skills_are_listed(client):
    body = client.get("/api/v1/skills").json()
    assert body["count"] == 4
    assert {skill["name"] for skill in body["skills"]} == {
        "health", "ecommerce", "insurance", "school"
    }


def test_a_listed_skill_says_what_it_may_do(client):
    body = client.get("/api/v1/skills").json()
    one = next(s for s in body["skills"] if s["name"] == "insurance")
    assert one["requires"] == "read"
    assert "search_corpus" in one["tools"]
    assert one["schema"]


# --- the dry run -----------------------------------------------------------


def test_matching_explains_its_choice(client):
    body = client.get(
        "/api/v1/skills/match", params={"intent": "which policies cover physiotherapy"}
    ).json()
    assert body["skill"] == "insurance"
    assert body["how"] == "trigger"
    assert body["why"]


def test_matching_runs_nothing(client, monkeypatch):
    """It is a preview. If it queued anything it would be useless as one."""
    import tasks

    def explode(*a, **kw):
        raise AssertionError("the preview enqueued a task")

    monkeypatch.setattr(tasks.investigate, "delay", explode)
    assert client.get("/api/v1/skills/match", params={"intent": "the syllabus"}).status_code == 200


def test_an_empty_intent_is_refused_by_the_endpoint(client):
    assert client.get("/api/v1/skills/match", params={"intent": ""}).status_code == 422


# --- queueing a task -------------------------------------------------------


def test_a_task_queues_the_matched_skill(client, queued):
    body = client.post(
        "/api/v1/task", json={"intent": "which policies cover physiotherapy"}
    ).json()
    assert body["status"] == "queued"
    assert body["matched"]["skill"] == "insurance"
    assert queued["skill"] == "insurance"


def test_the_composed_agent_is_named_in_the_reply(client, queued):
    """The caller learns which specialist will run in the same response that
    gives them the task id. A routing decision they cannot see is one they
    cannot correct."""
    body = client.post("/api/v1/task", json={"intent": "the exam timetable"}).json()
    assert body["agent"]["role"] == "school"
    assert "search_corpus" in body["agent"]["tools"]


@pytest.mark.slow
def test_an_unmatched_intent_still_runs_the_generic_specialist(client, queued):
    """Marked slow: no trigger fires, so the real embedder decides this one."""
    body = client.post(
        "/api/v1/task", json={"intent": "which acquisitions does the corpus describe"}
    ).json()
    assert body["matched"]["skill"] is None
    assert body["agent"]["role"] == "corpus"
    assert queued["skill"] is None


def test_naming_a_skill_overrides_the_routing(client, queued):
    body = client.post(
        "/api/v1/task", json={"intent": "the exam timetable", "skill": "health"}
    ).json()
    assert body["matched"]["how"] == "explicit"
    assert queued["skill"] == "health"


def test_an_unknown_skill_is_a_400_not_a_500(client, queued):
    response = client.post("/api/v1/task", json={"intent": "anything", "skill": "finance"})
    assert response.status_code == 400
    assert "finance" in response.json()["detail"]


def test_permissions_reach_the_task_as_sent(client, queued):
    """The security test for this surface. A skill cannot grant an effect; the
    request does, and it must arrive unchanged."""
    client.post(
        "/api/v1/task",
        json={"intent": "the exam timetable", "allow_network": True, "allow_write": True},
    )
    assert queued["allow_network"] is True
    assert queued["allow_write"] is True


def test_effects_are_off_unless_asked_for(client, queued):
    client.post("/api/v1/task", json={"intent": "the exam timetable"})
    assert queued["allow_network"] is False
    assert queued["allow_write"] is False


def test_the_task_polls_where_investigations_do(client, queued):
    """It produces an investigation, so it is polled like one rather than
    growing a second status endpoint that says the same thing."""
    body = client.post("/api/v1/task", json={"intent": "the exam timetable"}).json()
    assert body["poll"] == "/api/v1/investigations/task-abc"


# --- extraction ------------------------------------------------------------


def test_a_skill_supplies_the_extraction_schema(client, monkeypatch):
    seen: dict = {}

    class _Task:
        id = "extract-1"

    import tasks

    def fake_delay(url, prompt, schema, **kwargs):
        seen["schema"] = schema
        seen.update(kwargs)
        return _Task()

    monkeypatch.setattr(tasks.extract_url, "delay", fake_delay)
    response = client.post(
        "/api/v1/extract",
        json={"url": "https://example.test/policy", "prompt": "the policy", "skill": "insurance"},
    )
    assert response.status_code == 202
    assert "sum_insured" in seen["schema"]
    assert seen["skill"] == "insurance"


def test_an_extraction_with_neither_schema_nor_skill_is_refused(client):
    response = client.post(
        "/api/v1/extract", json={"url": "https://example.test/a", "prompt": "anything"}
    )
    assert response.status_code == 422


def test_an_explicit_schema_is_not_overwritten_by_a_skill(client, monkeypatch):
    """Naming a skill for its graph types must not silently replace the fields
    the caller asked for."""
    seen: dict = {}

    class _Task:
        id = "extract-2"

    import tasks

    monkeypatch.setattr(
        tasks.extract_url, "delay",
        lambda url, prompt, schema, **kw: (seen.update(schema=schema), _Task())[1],
    )
    client.post(
        "/api/v1/extract",
        json={
            "url": "https://example.test/a",
            "prompt": "p",
            "schema_template": {"title": "string"},
            "skill": "insurance",
        },
    )
    assert seen["schema"] == {"title": "string"}
