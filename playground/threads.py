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

**Every turn is also matched to a domain.** ``pipeline.skills.match`` reads the
message and picks the skill whose vocabulary it belongs to, again without a
model call, and the matched skill composes the specialist that answers. A turn
that matches nothing runs the generic corpus role — which is what every
question got before skills existed, so the floor is unchanged.

Both decisions are recorded on the stored message rather than left in a log.
A conversation with two speeds and a rotating cast of specialists is
unreadable if you cannot see, per answer, what decided it — so the breakdown
travels with the message and the interface renders it.
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


def breakdown(
    question: str, *, allow_embedding: bool = True, skill: Optional[str] = None
) -> dict[str, Any]:
    """Everything decided about a message before any model is called.

    Two routing decisions and no model between them, which is what makes this
    cheap enough to run before the work starts and show to the caller: the
    path from ``route``, and the domain from ``match``. Returned as one object
    because they are read together — "answered directly, as the insurance
    specialist" is the sentence, and either half alone explains nothing.
    """
    from pipeline.agents.route import route

    decision = route(question)
    found = _match(question, allow_embedding=allow_embedding)
    chosen = found.skill if found else None

    # A skill resolved by the caller — ensure() having just written one, or a
    # person naming it — is what runs, whatever the matcher would have said.
    if skill and (chosen is None or chosen.name != skill):
        try:
            from pipeline.skills import get as get_skill

            chosen = get_skill(skill)
            found = None
        except Exception as exc:
            log.warning("playground.skill_missing", skill=skill, error=repr(exc))
    skill_object = chosen

    return {
        "path": decision.path,
        "why_path": decision.reason,
        "skill": skill_object.name if skill_object else None,
        "skill_how": found.how if found else ("resolved" if skill_object else "none"),
        "skill_confidence": round(found.confidence, 3) if found else 0.0,
        "why_skill": (
            found.explain() if found
            else (f"{skill_object.name} was written for this request"
                  if skill_object else "no skill matched")
        ),
        "runners_up": [
            {"skill": name, "score": round(score, 3)}
            for name, score in (found.runners_up if found else [])
        ],
        "tools": list(skill_object.tools) if skill_object else [],
        "agents": (
            [agent.to_dict() for agent in skill_object.agents] if skill_object else []
        ),
        "buildable": bool(skill_object and skill_object.buildable),
        "generated": bool(skill_object and skill_object.generated),
    }


def _match(question: str, *, allow_embedding: bool = True):
    """The matched skill, or None when skills are off or unavailable.

    Never raises. A conversation must not fail because a skill file was being
    edited while someone was typing.
    """
    try:
        from pipeline.skills import match as match_intent

        return match_intent(question, allow_embedding=allow_embedding)
    except Exception as exc:
        log.warning("playground.match_failed", error=repr(exc))
        return None


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


def established(thread_id: str, *, path: Optional[Path] = None) -> Optional[str]:
    """The domain this conversation is already in, if any.

    A thread holds its domain. Only the opening message describes a subject
    area; everything after it is a question *within* that subject, and matching
    each one independently produces two bad outcomes.

    It drifts: "which classes are most popular with members" does not look like
    the gym question that started the thread, so a turn later the agent is
    answering as a different specialist with different tools.

    And with a skill written for an unmatched intent, it manufactures rubbish.
    Measured: a gym thread whose follow-up asked about classes had a `classes`
    skill written for it — name, description and triggers all lifted from the
    question — which then matched anything mentioning a class.

    So the domain is decided once, by the message that opened the thread, and
    held. A person who wants a different one starts a different conversation,
    which is also how they would think about it.
    """
    if not config.SKILLS_HOLD_THREAD_DOMAIN:
        return None
    for message in reversed(store.messages(thread_id, path=path)):
        if message.role != "assistant":
            continue
        named = (message.meta or {}).get("skill")
        if named:
            return str(named)
    return None


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
    skill: Optional[str] = None,
    local_only: bool = False,
    on_progress=None,
    path: Optional[Path] = None,
    supervisor=None,
    answerer=None,
) -> Reply:
    """Answer the newest message in this thread, and store the answer."""
    import time

    started = time.perf_counter()

    # Resolved here, in the worker, rather than in the API: this may write a
    # skill, which is a model call, and an HTTP handler is the wrong place for
    # one. An existing skill is reused; a domain nothing covers is written down
    # once and reused by every question after it.
    created = False
    if not skill:
        skill = established(thread_id, path=path)
        if skill:
            log.debug("playground.domain_held", thread=thread_id, skill=skill)
        else:
            try:
                from pipeline.skills import synth

                chosen_skill, created = synth.ensure(question, local_only=local_only)
                skill = chosen_skill.name if chosen_skill else None
            except Exception as exc:
                log.warning("playground.ensure_failed", error=repr(exc))

    decided = breakdown(question, skill=skill)
    decided["skill_created"] = created
    chosen = decided["path"]
    hydration = context.hydrate(thread_id, path=path)

    # Summarise before answering, so what the answer sees includes anything
    # that just fell out of the window rather than losing it for one turn.
    if hydration.needs_summary:
        hydration.summary = context.summarise(
            thread_id, hydration, local_only=local_only, path=path
        )

    if on_progress:
        on_progress({
            "stage": chosen,
            "skill": decided["skill"],
            "replayed": len(hydration.messages),
        })

    if chosen == "investigate":
        result = _investigate(
            thread_id, question, hydration,
            skill=decided["skill"],
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
            **decided,
            "sufficient": sufficient,
            "sources": sources,
            "replayed": len(hydration.messages),
            "summarised": bool(hydration.summary),
            # Which specialists actually ran, as opposed to which the skill
            # declares. They differ: acquisition joins a run that needed it,
            # and a worker the roster names may never have been reached.
            "ran": [
                step.get("role")
                for step in trace
                if step.get("kind") == "specialist" and step.get("role")
            ],
        },
        path=path,
    )

    log.info(
        "playground.replied",
        thread=thread_id,
        path=chosen,
        skill=decided["skill"],
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


def _investigate(
    thread_id, question, hydration, *, skill, local_only, on_progress, supervisor
):
    """The loop, with the thread's history and the domain's own specialist."""
    from pipeline.agents.supervisor import Supervisor

    role = None
    if skill:
        try:
            from pipeline.skills import get as get_skill

            role = get_skill(skill).as_role()
        except Exception as exc:
            # The generic specialist is a worse answer than the right one and a
            # much better one than a failed turn.
            log.warning("playground.skill_unavailable", skill=skill, error=repr(exc))

    runner = supervisor or Supervisor(
        local_only=local_only,
        role=role,
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


__all__ = ["Reply", "breakdown", "established", "in_flight", "plan", "reply", "start_turn"]
