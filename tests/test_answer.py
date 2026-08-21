"""Answering: grounded in the evidence, and honest when there is none."""

from __future__ import annotations

import pytest

from pipeline.graph.schema import Triple
from pipeline.retrieve.answer import answer_question
from pipeline.retrieve.hybrid import ScoredChunk
from pipeline.retrieve.orchestrator import RetrievalResult


def _result(chunks=None, triples=None) -> RetrievalResult:
    return RetrievalResult(
        query="q",
        chunks=chunks if chunks is not None else [
            ScoredChunk(id="c1", document="Revenue was 41.2 million in Q1.",
                        score=0.9, metadata={"source": "https://x.test/q1"}),
            ScoredChunk(id="c2", document="Headcount grew to 300.",
                        score=0.8, metadata={"source": "https://x.test/hr"}),
        ],
        triples=triples or [],
    )


def test_the_answer_and_its_citations_come_back(fake_backend):
    backend = fake_backend(
        {"answer": "Revenue was 41.2 million [1].", "sufficient": True, "citations": [1]}
    )
    reply = answer_question("what was revenue", result=_result(), backend=backend)

    assert reply.sufficient is True
    assert "41.2 million" in reply.answer
    assert reply.cited == [1]
    assert [s.number for s in reply.sources] == [1, 2]


def test_citations_are_read_from_the_text_not_the_field(fake_backend):
    """What the model wrote into the answer is what it actually used."""
    backend = fake_backend(
        {"answer": "Both [1] and [2] are relevant.", "sufficient": True, "citations": [1]}
    )
    reply = answer_question("q", result=_result(), backend=backend)
    assert reply.cited == [1, 2]


def test_a_citation_pointing_nowhere_is_dropped(fake_backend):
    backend = fake_backend(
        {"answer": "As shown in [7].", "sufficient": True, "citations": [7]}
    )
    reply = answer_question("q", result=_result(), backend=backend)
    assert reply.cited == []


def test_an_uncited_answer_is_flagged(fake_backend):
    """Grounded prose with no citation is prose, and should not look grounded."""
    backend = fake_backend({"answer": "Revenue grew.", "sufficient": True, "citations": []})
    reply = answer_question("q", result=_result(), backend=backend)
    assert any("cites no source" in w for w in reply.warnings)


def test_insufficient_evidence_is_a_real_outcome(fake_backend):
    backend = fake_backend(
        {"answer": "The sources do not mention the merger date.", "sufficient": False}
    )
    reply = answer_question("when did the merger close", result=_result(), backend=backend)
    assert reply.sufficient is False
    assert "do not mention" in reply.answer


def test_no_evidence_means_no_model_call(fake_backend):
    """Nothing retrieved is answered by saying so, not by asking a model."""
    backend = fake_backend({"answer": "should not be used", "sufficient": True})
    reply = answer_question("q", result=_result(chunks=[]), backend=backend)

    assert backend.calls == 0
    assert reply.sufficient is False
    assert "not been ingested" in reply.answer


def test_the_graph_is_one_numbered_source(fake_backend):
    backend = fake_backend({"answer": "See [3].", "sufficient": True})
    triples = [Triple(source="A", relation="ACQUIRED", target="B"),
               Triple(source="B", relation="OWNS", target="C")]
    reply = answer_question("q", result=_result(triples=triples), backend=backend)

    graph_sources = [s for s in reply.sources if s.kind == "graph"]
    assert len(graph_sources) == 1
    assert graph_sources[0].number == 3
    assert "ACQUIRED" in graph_sources[0].text
    assert "OWNS" in graph_sources[0].text


def test_passages_are_truncated_before_the_prompt(fake_backend, monkeypatch):
    from config import config

    monkeypatch.setattr(config, "ANSWER_MAX_PASSAGE_CHARS", 20)
    backend = fake_backend({"answer": "ok [1]", "sufficient": True})
    reply = answer_question("q", result=_result(), backend=backend)
    assert all(len(s.text) <= 20 for s in reply.sources if s.kind == "passage")


def test_only_so_many_passages_reach_the_model(fake_backend, monkeypatch):
    from config import config

    monkeypatch.setattr(config, "ANSWER_MAX_PASSAGES", 1)
    backend = fake_backend({"answer": "ok [1]", "sufficient": True})
    reply = answer_question("q", result=_result(), backend=backend)
    assert len([s for s in reply.sources if s.kind == "passage"]) == 1


def test_a_dead_model_keeps_the_evidence(fake_backend):
    """Failing to write prose must not throw away what was retrieved."""
    backend = fake_backend(raises=RuntimeError("ollama down"))
    reply = answer_question("q", result=_result(), backend=backend)

    assert reply.answer == ""
    assert len(reply.sources) == 2
    assert any("could not generate" in w for w in reply.warnings)


def test_an_empty_question_is_handled(fake_backend):
    reply = answer_question("   ", result=_result(), backend=fake_backend({}))
    assert reply.warnings == ["empty question"]


def test_the_reply_serialises(fake_backend):
    import json

    backend = fake_backend({"answer": "ok [1]", "sufficient": True})
    reply = answer_question("q", result=_result(), backend=backend)
    payload = json.loads(json.dumps(reply.to_dict()))
    assert payload["sources"][0]["number"] == 1
    assert payload["retrieval"]["chunks"]


def test_an_empty_answer_says_something_useful(fake_backend):
    """A small model often returns the verdict and no prose. Silence reads as
    a broken feature, so the verdict is spoken instead."""
    backend = fake_backend({"answer": "", "sufficient": False, "citations": [1, 2]})
    reply = answer_question("what was revenue", result=_result(), backend=backend)

    assert reply.answer
    assert "do not answer this" in reply.answer
    assert reply.sufficient is False
    # No text used them, so nothing is cited.
    assert reply.cited == []


def test_an_empty_answer_marked_sufficient_is_not_trusted(fake_backend):
    backend = fake_backend({"answer": "   ", "sufficient": True, "citations": [1]})
    reply = answer_question("q", result=_result(), backend=backend)
    assert reply.sufficient is False
    assert reply.cited == []
