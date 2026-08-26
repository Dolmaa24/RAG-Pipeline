"""Routing a question by its shape.

`bench/agents.py` measured both paths over the same thirteen questions:

    kind           ask once      investigate
    single_hop     4/4 · 6.6s    3/4 · 20.7s
    multi_hop      2/3           2/3
    enumeration    1/3           3/3
    unanswerable   3/3           3/3

The loop earns its four-fold latency on enumeration and nowhere else, so that
is what this routes on. Being wrong is cheap in one direction and expensive in
the other: sending a genuine enumeration to the fast path costs one partial
answer, and sending an ordinary question to the loop costs twenty seconds of
somebody's attention on every such question. Anything unrecognised therefore
takes the fast path.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app as api
from pipeline.agents.route import route


SET_SHAPED = [
    "Which acquisitions does the corpus describe?",
    "Which two acquisitions are described, and what were the amounts?",
    "Which 3 suppliers were used?",
    "How many acquisitions are there?",
    "List every acquisition in the corpus.",
    "List all the regions covered.",
    "What kinds of relationship exist?",
    "What types of document are indexed?",
    "Which companies acquired which other companies?",
    "Name all the suppliers mentioned.",
    "Enumerate the entities in the graph.",
]

ORDINARY = [
    "What did Acme Corporation acquire?",
    "How much did Northwind pay for Fabrikam Freight?",
    "Where is Fabrikam Freight headquartered?",
    "What was the quarterly revenue?",
    "What is the capital of Mongolia?",
    "Who founded Acme Corporation?",
    "What does the report say about margins?",
    "When did the acquisition complete?",
    "Summarise the board memo.",
]


@pytest.mark.parametrize("question", SET_SHAPED)
def test_a_question_asking_for_a_set_is_investigated(question):
    assert route(question).investigate, question


@pytest.mark.parametrize("question", ORDINARY)
def test_an_ordinary_question_is_answered_directly(question):
    assert not route(question).investigate, question


@pytest.mark.parametrize(
    "question",
    [
        # "s" that is not a plural. The first version of the pattern matched
        # "what **is**" and "what **was**", routing two of the benchmark's own
        # lookups to a twenty-second path.
        "What is the capital of Mongolia?",
        "What was the quarterly revenue?",
        "What has the company announced?",
        "What does the memo say?",
        "What is the address of the business?",
        "What was the status of the process?",
    ],
)
def test_a_verb_ending_in_s_is_not_a_plural_noun(question):
    assert not route(question).investigate, question


@pytest.mark.parametrize(
    "question",
    [
        "Which company has the most employees?",
        "Which supplier offered the lowest price?",
        "Which of these is the largest?",
    ],
)
def test_a_superlative_asks_which_one_not_which_ones(question):
    assert not route(question).investigate, question


def test_an_empty_question_is_not_investigated():
    decision = route("   ")
    assert not decision.investigate
    assert "empty" in decision.reason


def test_routing_can_be_turned_off(monkeypatch):
    import pipeline.agents.route as module

    monkeypatch.setattr(module.config, "AGENT_ROUTE_ENUMERATION", False)
    decision = route("Which acquisitions does the corpus describe?")

    assert not decision.investigate
    assert "off" in decision.reason


def test_every_decision_explains_itself():
    for question in (*SET_SHAPED, *ORDINARY, ""):
        assert route(question).reason


@pytest.fixture
def client():
    return TestClient(api.app)


def test_route_can_be_previewed_without_taking_it(client):
    payload = client.get(
        "/api/v1/route", params={"question": "How many suppliers are there?"}
    ).json()

    assert payload["path"] == "investigate"
    assert payload["reason"]


def test_ask_queues_an_investigation_for_a_set_shaped_question(client, monkeypatch):
    class _Task:
        id = "task-1"

    import tasks

    monkeypatch.setattr(tasks.investigate, "delay", lambda *a, **k: _Task())
    payload = client.post(
        "/api/v1/ask", json={"query": "Which acquisitions does the corpus describe?"}
    ).json()

    assert payload["path"] == "investigate"
    assert payload["task_id"] == "task-1"
    # The two paths return different shapes and this says which happened, so a
    # caller can handle both without guessing.
    assert "poll" in payload


def test_ask_answers_an_ordinary_question_directly(client, monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("an ordinary question must not reach the loop")

    import tasks

    monkeypatch.setattr(tasks.investigate, "delay", never)
    monkeypatch.setattr(
        api, "answer", lambda request: {"answer": "Beta Industries", "sufficient": True}
    )

    payload = client.post(
        "/api/v1/ask", json={"query": "What did Acme Corporation acquire?"}
    ).json()

    assert payload["path"] == "answer"
    assert payload["answer"] == "Beta Industries"
