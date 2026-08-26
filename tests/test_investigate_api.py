"""The surfaces: a queue of its own, an endpoint, and what it hands back.

No test here runs an investigation. The task is a transport — it builds a
budget from the request's permissions, hands it to the supervisor and shapes
the result — and what can go wrong in a transport is that a permission is lost
on the way through. That is what these check.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app as api
from celery_app import celery_app
from config import config


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture
def queued(monkeypatch):
    """Capture what would have been enqueued, without a broker."""
    seen: dict = {}

    class _Task:
        id = "task-123"

    import tasks

    def fake_delay(question, **kwargs):
        seen["question"] = question
        seen.update(kwargs)
        return _Task()

    monkeypatch.setattr(tasks.investigate, "delay", fake_delay)
    return seen


def test_investigations_have_their_own_queue():
    # Not io: a ninety-second run would sit in a thread sixteen fetches are
    # queued behind. Not cpu: it would block behind a transcription.
    route = celery_app.conf.task_routes["tasks.investigate"]
    assert route["queue"] == config.AGENTS_QUEUE
    assert config.AGENTS_QUEUE not in (config.IO_QUEUE, config.CPU_QUEUE)


def test_the_task_does_not_retry():
    # Retrying is for work the network broke. A failed investigation already
    # spent its budget on model calls and would spend it again the same way.
    import tasks

    assert not getattr(tasks.investigate, "autoretry_for", ())


def test_a_plain_request_grants_nothing(client, queued):
    response = client.post("/api/v1/investigate", json={"question": "what is x?"})

    assert response.status_code == 202
    assert queued["allow_network"] is False
    assert queued["allow_write"] is False


def test_permissions_are_passed_through(client, queued):
    client.post(
        "/api/v1/investigate",
        json={"question": "q", "allow_network": True, "allow_write": True},
    )
    assert queued["allow_network"] is True
    assert queued["allow_write"] is True


def test_the_response_says_where_to_poll(client, queued):
    payload = client.post("/api/v1/investigate", json={"question": "q"}).json()

    assert payload["status"] == "queued"
    assert payload["poll"] == f"/api/v1/investigations/{payload['task_id']}"


def test_an_empty_question_is_refused_before_it_is_queued(client, queued):
    assert client.post("/api/v1/investigate", json={"question": ""}).status_code == 422
    assert queued == {}


def test_rounds_are_bounded_by_the_schema(client, queued):
    assert client.post(
        "/api/v1/investigate", json={"question": "q", "max_rounds": 99}
    ).status_code == 422


def test_a_broker_that_cannot_be_reached_is_a_503(client, monkeypatch):
    import tasks

    def explode(*args, **kwargs):
        raise ConnectionError("redis is down")

    monkeypatch.setattr(tasks.investigate, "delay", explode)
    response = client.post("/api/v1/investigate", json={"question": "q"})

    assert response.status_code == 503
    assert "broker" in response.json()["detail"]


def _budget_for(monkeypatch, **permissions):
    """Run the task with the supervisor stubbed, and return the budget it built."""
    captured: dict = {}

    class FakeSupervisor:
        def __init__(self, *, budget, **kwargs):
            captured["budget"] = budget
            captured.update(kwargs)

        def investigate(self, question):
            class _Result:
                @staticmethod
                def to_dict():
                    return {"answer": "", "question": question}

            return _Result()

    import pipeline.agents.supervisor as supervisor

    monkeypatch.setattr(supervisor, "Supervisor", FakeSupervisor)
    from tasks import investigate

    investigate.apply(args=["q"], kwargs=permissions)
    return captured


def test_without_permission_the_effects_are_not_merely_capped(monkeypatch):
    from pipeline.agents.tools import Effect

    budget = _budget_for(monkeypatch)["budget"]

    assert budget.network_calls == 0
    assert budget.write_calls == 0
    # The counter is the grant, so zero means the tools are absent from what
    # the model is shown rather than present and refused.
    assert budget.effects() == [Effect.READ]


def test_allowing_the_network_does_not_allow_writing(monkeypatch):
    from pipeline.agents.tools import Effect

    budget = _budget_for(monkeypatch, allow_network=True)["budget"]

    assert budget.network_calls == config.AGENT_NETWORK_CALLS
    assert budget.write_calls == 0
    assert Effect.WRITE not in budget.effects()


def test_allowing_both_budgets_both(monkeypatch):
    budget = _budget_for(monkeypatch, allow_network=True, allow_write=True)["budget"]

    assert budget.network_calls == config.AGENT_NETWORK_CALLS
    assert budget.write_calls == config.AGENT_WRITE_CALLS


def test_the_task_wires_a_progress_reporter(monkeypatch):
    # A run takes half a minute or more. Without this the caller watches a
    # spinner and cannot tell a slow graph query from a hung model.
    assert _budget_for(monkeypatch)["on_progress"] is not None


def _status(monkeypatch, *, state, info=None, successful=True, ready=True):
    class FakeResult:
        status = state

        def __init__(self, *args, **kwargs):
            self.info = info
            self.result = info

        def ready(self):
            return ready

        def successful(self):
            return successful

    monkeypatch.setattr(api, "AsyncResult", FakeResult)


def test_progress_is_reported_while_it_runs(client, monkeypatch):
    _status(monkeypatch, state="PROGRESS", info={"stage": "gather", "role": "corpus"}, ready=False)
    payload = client.get("/api/v1/investigations/abc").json()

    assert payload["done"] is False
    assert payload["progress"]["stage"] == "gather"


def test_a_pending_task_is_not_done(client, monkeypatch):
    _status(monkeypatch, state="PENDING", ready=False)
    assert client.get("/api/v1/investigations/abc").json()["done"] is False


def test_a_finished_task_returns_its_result(client, monkeypatch):
    _status(monkeypatch, state="SUCCESS", info={"answer": "42"})
    payload = client.get("/api/v1/investigations/abc").json()

    assert payload["done"] is True
    assert payload["result"]["answer"] == "42"


def test_a_failed_task_reports_the_error_rather_than_a_result(client, monkeypatch):
    _status(monkeypatch, state="FAILURE", info="it broke", successful=False)
    payload = client.get("/api/v1/investigations/abc").json()

    assert payload["done"] is True
    assert "it broke" in payload["error"]
    assert "result" not in payload
