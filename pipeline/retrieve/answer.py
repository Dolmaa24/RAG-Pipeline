"""Answering the question, from the evidence and nothing else.

Retrieval finds passages and graph facts. This turns them into the thing a
person actually asked for — an answer — while keeping the property that makes a
retrieval system worth having over a chat model: **every claim is traceable to a
document you can open.**

Three rules shape the prompt, and each one exists because of a specific way
grounded answering goes wrong:

* **Cite by number.** Sources are numbered in the prompt and the answer refers
  to them as ``[2]``. Without this the answer is prose you have to fact-check
  from scratch, which is most of the work the retrieval was supposed to save.
* **Say when the evidence does not answer it.** A model handed six passages
  will compose something from them whether or not they are relevant. The
  ``sufficient`` flag is asked for separately so "the corpus does not cover
  this" is a first-class outcome rather than a paragraph that hedges.
* **Do not answer from general knowledge.** A model that knows the answer
  anyway will happily supply it, and the result looks identical to a grounded
  answer while being unsupported by anything in the store.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from config import config
from observability import get_logger, metrics

from pipeline.retrieve.filters import MetadataFilter
from pipeline.retrieve.orchestrator import RetrievalResult, retrieve

log = get_logger("answer")

_PROMPT = """Answer the question using ONLY the numbered sources below.

Rules:
- Answer every part of the question. A question asking two things is not
  answered by one of them: if it asks what happened and for how much, give both.
- Cite the sources you used inline, as [1], [2]. Every factual claim needs one.
- If the sources do not contain the answer, set sufficient to false and say
  plainly what is missing. Do not answer from your own knowledge.
- If the sources disagree, say so and cite both.
- Be direct. No preamble, no restating the question.

Sources:
{sources}"""

_SCHEMA_HINT = {
    "answer": "string",
    "sufficient": "boolean",
    "citations": "list of numbers",
}

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sufficient": {"type": "boolean"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answer", "sufficient"],
}

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
    """One numbered thing the answer was allowed to use."""

    number: int
    kind: str  # "passage" or "graph"
    text: str
    origin: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "kind": self.kind,
            "text": self.text,
            "origin": self.origin,
            "score": round(self.score, 6),
            "metadata": self.metadata,
        }


@dataclass
class Answer:
    """What the model concluded, and everything it was allowed to see."""

    question: str
    answer: str = ""
    sufficient: bool = False
    sources: list[Source] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    retrieval: Optional[RetrievalResult] = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "question": self.question,
            "answer": self.answer,
            "sufficient": self.sufficient,
            "sources": [s.to_dict() for s in self.sources],
            "cited": self.cited,
            "timings_ms": self.timings_ms,
            "warnings": self.warnings,
        }
        if self.retrieval is not None:
            payload["retrieval"] = self.retrieval.to_dict()
        return payload


def _build_sources(result: RetrievalResult) -> list[Source]:
    """Number the evidence, passages first, then the graph."""
    sources: list[Source] = []

    for chunk in result.chunks[: config.ANSWER_MAX_PASSAGES]:
        metadata = chunk.metadata or {}
        sources.append(
            Source(
                number=len(sources) + 1,
                kind="passage",
                text=chunk.document[: config.ANSWER_MAX_PASSAGE_CHARS],
                origin=str(metadata.get("source", "")),
                score=chunk.score,
                metadata=metadata,
            )
        )

    triples = result.triples[: config.ANSWER_MAX_TRIPLES]
    if triples:
        # The graph goes in as one numbered source rather than one per edge:
        # a subgraph is a single body of evidence, and twenty numbers pointing
        # at one-line facts crowds out the passages.
        rendered = "\n".join(t.render() for t in triples)
        origins = {t.source_url for t in triples if t.source_url}
        sources.append(
            Source(
                number=len(sources) + 1,
                kind="graph",
                text=rendered,
                origin=", ".join(sorted(origins)) or "knowledge graph",
                metadata={"triples": len(triples)},
            )
        )

    return sources


def _render(sources: list[Source]) -> str:
    parts = []
    for source in sources:
        label = source.origin or "unknown source"
        parts.append(f"[{source.number}] ({label})\n{source.text}")
    return "\n\n".join(parts)


def answer_question(
    question: str,
    *,
    filters: Optional[MetadataFilter] = None,
    limit: Optional[int] = None,
    use_graph: Optional[bool] = None,
    rerank_results: Optional[bool] = None,
    rewrite: Optional[bool] = None,
    local_only: bool = False,
    result: Optional[RetrievalResult] = None,
    extra_queries: Optional[list[str]] = None,
    backend=None,
) -> Answer:
    """Retrieve, then answer from what was retrieved.

    ``result`` lets a caller who has already retrieved reuse it rather than
    searching twice.
    """
    started = time.perf_counter()
    reply = Answer(question=question)

    if not question or not question.strip():
        reply.warnings.append("empty question")
        return reply

    # --- retrieve ------------------------------------------------------- #
    stage = time.perf_counter()
    if result is None:
        result = retrieve(
            question,
            filters=filters,
            limit=limit,
            use_graph=use_graph,
            rerank_results=rerank_results,
            rewrite=rewrite,
            local_only=local_only,
            extra_queries=extra_queries,
        )
    reply.retrieval = result
    reply.warnings.extend(result.warnings)
    reply.timings_ms["retrieve"] = round((time.perf_counter() - stage) * 1000, 2)

    reply.sources = _build_sources(result)
    if not reply.sources:
        reply.answer = (
            "Nothing in the indexed corpus matches that question. Either the "
            "documents that would answer it have not been ingested, or the "
            "filters excluded them."
        )
        reply.sufficient = False
        reply.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
        log.info("answer.no_evidence", question=question[:60])
        return reply

    # --- answer --------------------------------------------------------- #
    stage = time.perf_counter()
    try:
        if backend is None:
            from pipeline.extract.llm import INTERACTIVE, get_backend

            backend = get_backend(local_only=local_only, role=INTERACTIVE)

        with metrics.timer("answer.generate"):
            response = backend.complete_json(
                prompt=_PROMPT.format(sources=_render(reply.sources)),
                content=question,
                schema_hint=_SCHEMA_HINT,
                json_schema=_JSON_SCHEMA,
            )
        data = response.data or {}
    except Exception as exc:
        # The evidence is real and useful on its own; failing to write prose
        # about it should not throw the passages away.
        log.warning("answer.generation_failed", error=repr(exc))
        metrics.incr("answer.failed")
        reply.warnings.append(f"could not generate an answer: {exc}")
        reply.answer = ""
        reply.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
        return reply

    reply.answer = str(data.get("answer") or "").strip()
    reply.sufficient = bool(data.get("sufficient", False))
    reply.timings_ms["generate"] = round((time.perf_counter() - stage) * 1000, 2)

    if not reply.answer:
        # A small model asked about something the sources do not cover often
        # returns the verdict and no prose. Silence reads as a broken feature,
        # so say the thing the verdict means — and claim no citations, because
        # there is no text that used them.
        reply.answer = (
            "The indexed documents do not answer this. What was retrieved is "
            "below; if the answer should be in there, the wording may not match."
            if not reply.sufficient
            else "The model returned no text for this question."
        )
        reply.sufficient = False
        reply.cited = []
        log.info("answer.empty_text", question=question[:60])
        reply.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
        return reply

    # Trust the text over the field: a citation the model wrote into the answer
    # is one it actually used, while the list is a second chance to get it wrong.
    valid = {s.number for s in reply.sources}
    reply.cited = sorted(
        {int(n) for n in _CITATION.findall(reply.answer) if int(n) in valid}
    )
    if not reply.cited:
        for raw in data.get("citations") or []:
            try:
                number = int(raw)
            except (TypeError, ValueError):
                continue
            if number in valid:
                reply.cited.append(number)
        reply.cited = sorted(set(reply.cited))

    if reply.sufficient and not reply.cited:
        reply.warnings.append(
            "the answer cites no source — treat it as unverified"
        )

    reply.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 2)
    metrics.incr("answer.questions")
    log.info(
        "answer.done",
        question=question[:60],
        sufficient=reply.sufficient,
        sources=len(reply.sources),
        cited=len(reply.cited),
        ms=reply.timings_ms["total"],
    )
    return reply


__all__ = ["Answer", "Source", "answer_question"]
