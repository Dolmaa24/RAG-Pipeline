"""One turn of a conversation, from stored history to a stored reply.

The service layer both the API and the Celery task call, so that "what happens
when someone sends a message" is written once. The API appends the user's
message and queues; the worker hydrates, answers and appends the reply. Split
that logic across the two and they drift.

**Which path a turn takes is not decided here.** ``pipeline.agents.route``
already chooses between answering directly and running the loop, from the
question's own words and with no model call, and it was benchmarked over
thirteen questions to make that choice. A conversation is exactly where its
bias matters: most turns are ordinary lookups that should come back in seconds,
and the loop earns its four-fold latency only on questions asking for a set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config import config
from observability import get_logger, metrics

from playground import context, store

log = get_logger("playground.threads")


@dataclass(slots=True)
class Reply:
    """What one turn produced."""

    thread_id: str
    message: Optional[store.Message] = None
    path: str = ""
    sufficient: bool = False
    sources: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "message": self.message.to_dict() if self.message else None,
            "path": self.path,
            "sufficient": self.sufficient,
            "sources": self.sources,
            "trace": self.trace,
            "warnings": self.warnings,
            "seconds": round(self.seconds, 2),
        }


def plan(question: str) -> str:
    """Which path this turn will take: "answer" or "investigate"."""
    from pipeline.agents.route import route

    return route(question).path


def start_turn(
    thread_id: str, content: str, *, path: Optional[Path] = None
) -> store.Message:
    """Record what the user said, before anything slow happens.

    Separate from :func:`reply` and called by the API rather than the worker.
    A message that only appears once the agent has finished is a message the
    user cannot see they sent — and if the worker then fails, it never existed
    at all.
    """
    return store.append_message(thread_id, "user", content, path=path)


def in_flight(thread_id: str, *, path: Optional[Path] = None) -> bool:
    """Whether this thread is waiting on a reply.

    One turn at a time per thread. Two agents appending to one history
    interleave into a transcript neither of them was answering.
    """
    last = store.last_message(thread_id, path=path)
    return bool(last and last.role == "user")


def reply(
    thread_id: str,
    question: str,
    *,
    local_only: bool = False,
    on_progress=None,
    path: Optional[Path] = None,
    supervisor=None,
    answerer=None,
) -> Reply:
    """Answer the newest message in this thread, and store the answer."""
    import time

    started = time.perf_counter()
    chosen = plan(question)
    hydration = context.hydrate(thread_id, path=path)

    # Summarise before answering, so what the answer sees includes anything
    # that just fell out of the window rather than losing it for one turn.
    if hydration.needs_summary:
        hydration.summary = context.summarise(
            thread_id, hydration, local_only=local_only, path=path
        )

    if on_progress:
        on_progress({"stage": chosen, "replayed": len(hydration.messages)})

    if chosen == "investigate":
        result = _investigate(
            thread_id, question, hydration,
            local_only=local_only, on_progress=on_progress, supervisor=supervisor,
        )
    else:
        result = _answer(
            question, hydration, local_only=local_only, answerer=answerer
        )

    answer_text, sufficient, sources, trace, warnings = result
    stored = store.append_message(
        thread_id,
        "assistant",
        answer_text or "I could not produce an answer for that.",
        meta={
            "path": chosen,
            "sufficient": sufficient,
            "sources": sources,
            "replayed": len(hydration.messages),
            "summarised": bool(hydration.summary),
        },
        path=path,
    )

    log.info(
        "playground.replied",
        thread=thread_id,
        path=chosen,
        replayed=len(hydration.messages),
        sources=len(sources),
        seconds=round(time.perf_counter() - started, 1),
    )
    metrics.incr(f"playground.reply.{chosen}")
    return Reply(
        thread_id=thread_id,
        message=stored,
        path=chosen,
        sufficient=sufficient,
        sources=sources,
        trace=trace,
        warnings=warnings,
        seconds=time.perf_counter() - started,
    )


def _investigate(thread_id, question, hydration, *, local_only, on_progress, supervisor):
    """The loop, with the thread's history in the specialist's messages."""
    from pipeline.agents.supervisor import Supervisor

    runner = supervisor or Supervisor(
        local_only=local_only,
        history=hydration.as_messages(),
        on_progress=on_progress,
    )
    outcome = runner.investigate(question)
    return (
        outcome.answer,
        outcome.sufficient,
        outcome.sources,
        outcome.trace,
        outcome.warnings,
    )


def _answer(question, hydration, *, local_only, answerer):
    """The fast path, with history folded into the question.

    ``answer_question`` retrieves and answers from a question string — it takes
    no message list, so there is nowhere to put a history. Prepending the recent
    exchanges is the only shape it accepts, and it is why this is a different
    branch rather than the same call with an extra argument.
    """
    from pipeline.retrieve.answer import answer_question

    preamble = hydration.as_text()
    asked = (
        f"{preamble}\n\nUser: {question}" if preamble else question
    )

    call = answerer or answer_question
    outcome = call(asked, local_only=local_only)
    return (
        outcome.answer,
        bool(getattr(outcome, "sufficient", False)),
        [source.to_dict() for source in getattr(outcome, "sources", [])],
        [],
        [],
    )


__all__ = ["Reply", "in_flight", "plan", "reply", "start_turn"]
