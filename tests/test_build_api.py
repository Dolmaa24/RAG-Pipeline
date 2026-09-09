"""The draft, approve and build endpoints.

A transport again, so these check what a transport loses: a permission that did
not arrive as sent, an error that became a 500 instead of the 400 it was, a
draft that reached the live set without passing the gate.

No model and no broker. The synthesis backend is stubbed and the task's
``delay`` is captured.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app as api
from config import config
from pipeline.skills import loader, synth
from tests.test_skill_synth import PAYLOAD, Stub


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def skills_root(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "skills_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def stubbed(monkeypatch):
    """Draft without a model."""
    real = synth.draft
    monkeypatch.setattr(
        synth, "draft", lambda intent, **kw: real(intent, backend=Stub())
    )


@pytest.fixture
def queued(monkeypatch):
    seen: dict = {}

    class _Task:
        id = "build-1"

    import tasks

    def fake_delay(skill, **kwargs):
        seen["skill"] = skill
        seen.update(kwargs)
        return _Task()

    monkeypatch.setattr(tasks.build_project, "delay", fake_delay)
    return seen


# --- drafting --------------------------------------------------------------


def test_drafting_returns_the_file_for_review(client, skills_root, stubbed):
    body = client.post("/api/v1/skills/draft", json={"intent": "make a Baristo system"}).json()
    assert body["name"] == "barista"
    assert "name: barista" in body["text"]
    assert body["approve"].endswith("/barista/approve")


def test_a_draft_is_not_live(client, skills_root, stubbed):
    """The gate. Present on disk, absent from what an agent can be given."""
    client.post("/api/v1/skills/draft", json={"intent": "make a Baristo system"})
    assert client.get("/api/v1/skills").json()["count"] == 0
    assert client.get("/api/v1/skills/drafts").json()["drafts"] == ["barista"]


def test_the_response_says_the_body_must_be_read(client, skills_root, stubbed):
    body = client.post("/api/v1/skills/draft", json={"intent": "anything"}).json()
    assert "system prompt" in body["note"]


def test_an_empty_intent_is_refused(client, skills_root):
    assert client.post("/api/v1/skills/draft", json={"intent": ""}).status_code == 422


def test_a_draft_naming_an_unknown_tool_is_a_422(client, skills_root, monkeypatch):
    """The caller's problem to see, not a server fault."""
    real = synth.draft
    monkeypatch.setattr(
        synth, "draft",
        lambda intent, **kw: real(intent, backend=Stub({**PAYLOAD, "tools": ["brew_coffee"]})),
    )
    response = client.post("/api/v1/skills/draft", json={"intent": "anything"})
    assert response.status_code == 422
    assert "brew_coffee" in response.json()["detail"]


def test_a_draft_reads_back(client, skills_root, stubbed):
    client.post("/api/v1/skills/draft", json={"intent": "anything"})
    assert "barista" in client.get("/api/v1/skills/drafts/barista").json()["text"]


def test_a_missing_draft_is_a_404(client, skills_root):
    assert client.get("/api/v1/skills/drafts/nothing").status_code == 404


def test_a_traversing_draft_name_is_not_a_500(client, skills_root):
    """The name is a URL path segment."""
    assert client.get("/api/v1/skills/drafts/..%2F..%2Fconfig").status_code in (400, 404)


# --- approving -------------------------------------------------------------


def test_approving_makes_the_skill_live(client, skills_root, stubbed):
    client.post("/api/v1/skills/draft", json={"intent": "make a Baristo system"})
    assert client.post("/api/v1/skills/drafts/barista/approve", json={}).status_code == 200

    listed = client.get("/api/v1/skills").json()
    assert [s["name"] for s in listed["skills"]] == ["barista"]
    assert client.get("/api/v1/skills/drafts").json()["drafts"] == []


def test_an_edited_draft_is_what_installs(client, skills_root, stubbed):
    client.post("/api/v1/skills/draft", json={"intent": "anything"})
    edited = client.get("/api/v1/skills/drafts/barista").json()["text"].replace(
        "Coffee shop operations", "Everything coffee"
    )
    client.post("/api/v1/skills/drafts/barista/approve", json={"text": edited})
    listed = client.get("/api/v1/skills").json()["skills"][0]
    assert "Everything coffee" in listed["description"]


def test_approving_a_broken_edit_is_a_400(client, skills_root, stubbed):
    client.post("/api/v1/skills/draft", json={"intent": "anything"})
    response = client.post(
        "/api/v1/skills/drafts/barista/approve", json={"text": "not a skill"}
    )
    assert response.status_code == 400
    assert client.get("/api/v1/skills").json()["count"] == 0


def test_approving_a_missing_draft_is_a_400(client, skills_root):
    assert client.post("/api/v1/skills/drafts/nothing/approve", json={}).status_code == 400


def test_discarding_removes_it(client, skills_root, stubbed):
    client.post("/api/v1/skills/draft", json={"intent": "anything"})
    assert client.delete("/api/v1/skills/drafts/barista").status_code == 200
    assert client.get("/api/v1/skills/drafts").json()["drafts"] == []


@pytest.mark.slow
def test_an_unmatched_task_says_a_draft_is_possible(client, monkeypatch):
    """Otherwise the operator has to know the endpoint exists."""
    import tasks

    class _T:
        id = "t"

    monkeypatch.setattr(tasks.investigate, "delay", lambda q, **kw: _T())
    body = client.post(
        "/api/v1/task", json={"intent": "which acquisitions does the corpus describe"}
    ).json()
    assert body["can_draft_skill"] is True


# --- building --------------------------------------------------------------


BUILDABLE = """\
---
name: barista
description: A cafe.
tools: [search_corpus]
triggers: [barista, espresso]
agents:
  - name: receptionist
    purpose: Takes orders.
    writes: [orders.py]
---

Body.
"""


@pytest.fixture
def installed(skills_root, monkeypatch):
    monkeypatch.setattr(config, "BUILD_ENABLED", True)
    folder = skills_root / "barista"
    folder.mkdir()
    (folder / "SKILL.md").write_text(BUILDABLE)
    loader.reset()
    return skills_root


def test_a_build_queues_the_skill(client, installed, queued):
    body = client.post("/api/v1/build", json={"intent": "the espresso queue"}).json()
    assert body["skill"] == "barista"
    assert queued["skill"] == "barista"
    assert [a["name"] for a in body["agents"]] == ["receptionist"]


def test_execution_is_off_unless_asked_for(client, installed, queued):
    body = client.post("/api/v1/build", json={"skill": "barista"}).json()
    assert body["will_run_tests"] is False
    assert queued["allow_execute"] is False


def test_execution_reaches_the_task_as_sent(client, installed, queued):
    """A skill cannot grant this; the request does, and it must arrive."""
    body = client.post(
        "/api/v1/build", json={"skill": "barista", "allow_execute": True}
    ).json()
    assert body["will_run_tests"] is True
    assert queued["allow_execute"] is True


def test_building_is_refused_when_the_switch_is_off(client, installed, monkeypatch):
    monkeypatch.setattr(config, "BUILD_ENABLED", False)
    response = client.post("/api/v1/build", json={"skill": "barista"})
    assert response.status_code == 403
    assert "BUILD_ENABLED" in response.json()["detail"]


def test_a_skill_with_no_roster_is_refused_before_queueing(client, skills_root, monkeypatch, queued):
    monkeypatch.setattr(config, "BUILD_ENABLED", True)
    folder = skills_root / "plain"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: plain\ndescription: d\ntools: [search_corpus]\n---\n\nBody.\n"
    )
    loader.reset()
    response = client.post("/api/v1/build", json={"skill": "plain"})
    assert response.status_code == 400
    assert "no agents" in response.json()["detail"]
    assert queued == {}


@pytest.mark.slow
def test_a_domain_nobody_anticipated_is_queued_anyway(client, installed, queued):
    """The limitation this replaced: four skills shipped, and a build of a
    library, a gym or a restaurant was refused outright.

    No skill is named in the response because there is not one yet — the worker
    writes the domain down before building it, drafting being a model call and
    an HTTP handler being the wrong place for one.

    Marked slow: no trigger fires, so the real embedder settles that there is
    no match before the fallback takes over.
    """
    response = client.post(
        "/api/v1/build", json={"intent": "a library book lending system"}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["skill"] is None
    assert body["writing_skill"] is True
    assert queued["skill"] is None
    assert queued["intent"] == "a library book lending system"


@pytest.mark.slow
def test_with_auto_create_off_an_unmatched_intent_points_at_drafting(
    client, installed, queued, monkeypatch
):
    """For anyone who would rather approve every skill by hand."""
    monkeypatch.setattr(config, "SKILLS_AUTO_CREATE", False)
    response = client.post(
        "/api/v1/build", json={"intent": "a library book lending system"}
    )
    assert response.status_code == 400
    assert "/api/v1/skills/draft" in response.json()["detail"]


def test_an_unknown_skill_is_a_400(client, installed, queued):
    response = client.post("/api/v1/build", json={"skill": "finance"})
    assert response.status_code == 400


def test_every_routed_queue_has_a_worker_that_consumes_it():
    """A task routed to a queue nothing listens on is queued for ever.

    The build queue was added and its worker was not, so POST /api/v1/build
    returned 202 and the Build tab spun until the poll timed out — a failure
    with no error anywhere, which is the worst shape a failure can take.
    """
    import re
    from pathlib import Path

    from celery_app import celery_app

    run_sh = (Path(__file__).resolve().parents[1] / "run.sh").read_text()
    served = set(re.findall(r"--queues=([a-z,]+)", run_sh))
    served = {queue for group in served for queue in group.split(",")}

    routed = {route["queue"] for route in celery_app.conf.task_routes.values()}
    assert routed <= served, f"no worker consumes: {sorted(routed - served)}"


def test_health_reports_depth_for_every_queue(client):
    """A queue missing from here hides the failure this endpoint exists for:
    work piling up because nothing is consuming it."""
    from celery_app import celery_app

    depths = client.get("/health").json()["queue_depth"]
    routed = {route["queue"] for route in celery_app.conf.task_routes.values()}
    assert routed <= set(depths), f"not reported: {sorted(routed - set(depths))}"
