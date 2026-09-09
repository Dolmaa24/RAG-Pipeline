"""Fitting a conversation into the window the model actually has.

The limit people reach for here is the wrong one. ``OLLAMA_NUM_PREDICT`` and
``GROQ_MAX_OUTPUT_TOKENS`` are both 4096, and both cap the *reply*; replaying
fifty turns does not touch either. What it exhausts is ``OLLAMA_NUM_CTX``,
which is 8192 for input **and** output together -- so with the reply's 4096
reserved, everything sent has to fit in roughly 4096 tokens. On Groq the same
pressure arrives as the 8000 tokens-per-minute allowance, which this project
has already measured a build hitting repeatedly.

Against the corpus role that leaves about 2600 tokens for history: eight to
twelve short exchanges, not fifty.

So: **a sliding window, plus one summary of what falls out of it.**

The window is the newest turns that fit. Everything older is evicted -- but
evicted *once*. The first time a turn falls out, one model call folds
everything up to that point into ``threads.summary``, and
``summarised_through`` records how far it reaches so the next turn does not pay
again. A ten-turn conversation never summarises at all; a fifty-turn one does
it a handful of times. Summarising on every message would be simpler and would
put a second model call in front of every answer, competing with it for the
same per-minute budget.

**Tool messages are stored and not replayed.** They are the bulkiest thing in a
thread, the agent has already acted on them, and feeding yesterday's search
results back invites the model to treat them as current.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import config
from observability import get_logger, metrics

from pipeline.extract.llm.base import Message as LLMMessage
from playground import store

log = get_logger("playground.context")

#: Roles worth replaying. A stored `system` message is a note about the thread
#: rather than the role's own prompt, which the loop supplies itself.
_REPLAYED = ("user", "assistant")

_SUMMARY_PROMPT = """\
Summarise this conversation so it can be carried forward, in at most 150 words.

Keep: what was asked, what was established as true, any names, numbers and
sources that later questions might refer back to, and anything the user stated
about what they want.

Drop: pleasantries, restatements, and your own reasoning about how you searched.

Write it as notes, in the third person, not as a reply to anyone."""

_SUMMARY_HEADER = "Earlier in this conversation:"


def estimate_tokens(text: str) -> int:
    """Roughly, and deliberately on the high side.

    A real tokenizer is a dependency and a model download for a number that
    only needs to be approximately right in the safe direction: overestimating
    sends less history than would fit, which costs a little recall.
    Underestimating overflows the window, which costs the whole turn.
    """
    per_token = max(1, config.PLAYGROUND_CHARS_PER_TOKEN)
    return (len(text or "") + per_token - 1) // per_token


@dataclass(slots=True)
class Hydration:
    """One thread, ready to be handed to a loop."""

    #: Replayable history, oldest first, already within budget.
    messages: list[LLMMessage] = field(default_factory=list)
    #: The rolling summary, or "" when the thread has never overflowed.
    summary: str = ""
    #: Messages that did not fit and are not covered by the summary yet.
    evicted: list[store.Message] = field(default_factory=list)
    #: Rough token count of what `messages` and `summary` will cost.
    tokens: int = 0

    @property
    def needs_summary(self) -> bool:
        return bool(self.evicted)

    def as_messages(self) -> list[LLMMessage]:
        """History plus the summary, in the order a loop should receive them.

        The summary leads, as a system message: it is background rather than
        something anyone said, and putting it after the turns it precedes would
        read as the most recent thing in the conversation.
        """
        if not self.summary:
            return list(self.messages)
        return [
            LLMMessage.system(f"{_SUMMARY_HEADER}\n{self.summary}"),
            *self.messages,
        ]

    def as_text(self, limit: int = 4) -> str:
        """The tail of the conversation as prose, for the fast answer path.

        ``answer_question`` retrieves and answers from a question string; it has
        no message history to be given. So a routed-to-answer turn gets the
        recent exchanges folded into the question instead, which is the only
        shape that path accepts.
        """
        parts = []
        if self.summary:
            parts.append(f"{_SUMMARY_HEADER}\n{self.summary}")
        recent = self.messages[-(limit * 2):] if limit else self.messages
        for message in recent:
            who = "User" if message.role == "user" else "You"
            parts.append(f"{who}: {message.content}")
        return "\n\n".join(parts)


def hydrate(
    thread_id: str,
    *,
    budget: Optional[int] = None,
    path: Optional[Path] = None,
) -> Hydration:
    """The most recent history that fits, and what fell out of it.

    Pure over the rows: no model is called here, so what to send and whether to
    pay for a summary are separable, and the first is testable on its own.
    """
    limit = config.PLAYGROUND_HISTORY_TOKENS if budget is None else budget
    thread = store.get_thread(thread_id, path=path)
    rows = store.messages(thread_id, path=path)

    summary = thread.summary or ""
    spent = estimate_tokens(summary) if summary else 0

    replayable = [row for row in rows if row.role in _REPLAYED]
    # Anything the summary already covers is not replayed again; it is in there.
    fresh = [row for row in replayable if row.id > thread.summarised_through]

    kept: list[store.Message] = []
    for row in reversed(fresh):
        cost = estimate_tokens(row.content) + 4  # role and framing overhead
        if kept and spent + cost > limit:
            break
        # The newest message is kept whatever it costs. A turn that cannot fit
        # its own immediate predecessor is still better than one with no
        # history at all, and truncating mid-conversation is the caller's
        # problem to notice rather than something to silently produce.
        spent += cost
        kept.append(row)

    kept.reverse()
    survived = {row.id for row in kept}
    evicted = [row for row in fresh if row.id not in survived]

    messages = [
        LLMMessage(role=row.role, content=row.content) for row in kept
    ]
    log.debug(
        "playground.hydrated",
        thread=thread_id,
        replayed=len(messages),
        evicted=len(evicted),
        tokens=spent,
    )
    return Hydration(messages=messages, summary=summary, evicted=evicted, tokens=spent)


def summarise(
    thread_id: str,
    hydration: Hydration,
    *,
    backend=None,
    local_only: bool = False,
    path: Optional[Path] = None,
) -> str:
    """Fold what fell out of the window into the thread's rolling summary.

    Called only when something was actually evicted, and it advances
    ``summarised_through`` so the same turns are never paid for twice. Returns
    the new summary, or the old one when nothing changed.

    A failure here is not a failed turn. The conversation continues with the
    window it has and the summary it had; losing the beginning of a long thread
    is worse than an error only in the sense that the user should be told, and
    the caller is.
    """
    if not hydration.evicted:
        return hydration.summary
    if not config.PLAYGROUND_SUMMARISE:
        # Still advance the marker: without this, every turn re-evicts the same
        # messages and `needs_summary` is permanently true.
        store.set_summary(
            thread_id, hydration.summary, hydration.evicted[-1].id, path=path
        )
        return hydration.summary

    if backend is None:
        from pipeline.extract.llm import INTERACTIVE, get_backend

        backend = get_backend(role=INTERACTIVE, local_only=local_only)

    transcript = "\n\n".join(
        f"{'User' if row.role == 'user' else 'Assistant'}: {row.content}"
        for row in hydration.evicted
    )
    if hydration.summary:
        transcript = (
            f"{_SUMMARY_HEADER}\n{hydration.summary}\n\n"
            f"And then:\n\n{transcript}"
        )

    try:
        with metrics.timer("playground.summarise"):
            response = backend.complete_json(
                prompt=_SUMMARY_PROMPT,
                content=transcript,
                schema_hint={"summary": "string"},
                json_schema={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            )
        text = str((response.data or {}).get("summary") or "").strip()
    except Exception as exc:
        log.warning("playground.summary_failed", thread=thread_id, error=repr(exc))
        metrics.incr("playground.summary.failed")
        return hydration.summary

    if not text:
        return hydration.summary

    store.set_summary(thread_id, text, hydration.evicted[-1].id, path=path)
    log.info(
        "playground.summarised",
        thread=thread_id,
        folded=len(hydration.evicted),
        through=hydration.evicted[-1].id,
    )
    metrics.incr("playground.summarised")
    return text


__all__ = ["Hydration", "estimate_tokens", "hydrate", "summarise"]
