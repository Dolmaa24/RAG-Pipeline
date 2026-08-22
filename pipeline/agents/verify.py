"""Checking a drafted answer against the passages it claims to rest on.

This is where "high precision" stops being a claim about the retrieval stack and
becomes something the system actually does. Retrieval being good does not stop a
model writing a sentence the passages do not support; it only makes it less
likely, and less likely is not a property you can show anyone.

One model call, and a deliberately narrow question. The verifier is not asked
whether the answer is *true* — it has no way to know that and would guess. It is
asked, for each sentence, whether the supplied passages say it. A sentence the
passages do not support is flagged rather than deleted: the draft is evidence
about what the system did, and silently rewriting it destroys the only record of
the failure.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from observability import get_logger, metrics

log = get_logger("agents.verify")

#: Sentence splitting that keeps abbreviations and decimals intact. Answers are
#: full of "42.5 million" and "Inc." and splitting those apart produces
#: fragments the verifier then judges as unsupported, which is a false alarm
#: caused by the splitter rather than by the model.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")

#: Where one sentence carries two claims. A true first half was found to carry
#: a false second half through unchallenged: "Acme acquired Beta Industries and
#: Northwind acquired Contoso Shipping" was vouched for in full, and the same
#: two claims as separate sentences had the invented one flagged. The model
#: judges a sentence as a unit, so a unit holding two claims is judged on
#: whichever it notices.
_CLAUSE = re.compile(r"\s*(?:;|,?\s+and|,?\s+but|,?\s+while|,?\s+whereas)\s+", re.IGNORECASE)

#: Both halves must be long enough to be a claim rather than a fragment.
#: "Revenue was 42.5 million dollars and rising" is one claim, not two.
_MIN_CLAUSE = 25

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "verification",
    "properties": {
        "supported": {
            "type": "array",
            "description": "Numbers of sentences the passages do support.",
            "items": {"type": "integer"},
        },
        "note": {"type": ["string", "null"], "description": "Optional one line."},
    },
    "required": ["supported"],
}

#: Asks which sentences **are** supported, and inverts. The first version asked
#: for the unsupported ones, and on three sentences whose evidence was verbatim
#: in the passages, qwen2.5:3b named the wrong one and llama3.2:3b named none;
#: asked positively, both named exactly the right two. Small models invert
#: negated selection tasks often enough that a verifier built on one is not a
#: verifier. Measured, not guessed — see bench/verify.py.
#:
#: The inversion also fails closed: a sentence the model does not list is
#: flagged. For a check whose output is an annotation rather than a deletion,
#: erring towards flagging is the right direction.
_INSTRUCTION = """\
For each numbered sentence, decide whether the PASSAGES below say it.

Judge only what the passages contain. Do not use your own knowledge, and do not
judge whether a sentence is true — only whether these passages support it. A
sentence that restates a passage in different words is supported. A sentence
adding a number, a name, a date or a causal claim the passages do not contain
is not.

Return the numbers of the sentences that ARE supported. List every one of them.
"""


#: What the check is worth, in the words a reader needs to act on it. Measured
#: by bench/verify.py against the local model: it catches most fabrications and
#: rejects roughly one supported sentence in three, almost always a paraphrase.
#: Stating that alongside the flags is the difference between a reader
#: discounting a flag and a reader discounting all of them.
LOCAL_CAVEAT = (
    "Advisory. The local checker catches about 85% of unsupported claims and "
    "wrongly flags about a third of supported ones, usually when the answer "
    "restates a passage in different words. Read a flag as 'worth checking', "
    "not as 'wrong'."
)


@dataclass(slots=True)
class Verdict:
    """What survived the check, and what did not."""

    answer: str
    unsupported: list[str] = field(default_factory=list)
    checked: int = 0
    note: str = ""
    seconds: float = 0.0
    #: How far to trust the flags. Empty when nothing was flagged, since a
    #: caveat on a clean result is noise.
    caveat: str = ""
    #: True when the check could not run at all. Distinguished from "everything
    #: was supported", because those must not look the same to a caller.
    skipped: bool = False
    reason: str = ""

    @property
    def clean(self) -> bool:
        return not self.unsupported and not self.skipped

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "unsupported": self.unsupported,
            "clean": self.clean,
            "skipped": self.skipped,
            "caveat": self.caveat,
            "reason": self.reason,
            "note": self.note,
            "seconds": round(self.seconds, 2),
        }


def split_sentences(text: str) -> list[str]:
    """The units to check, one claim each where that can be told apart."""
    units: list[str] = []
    for sentence in _SENTENCE.split(text.strip()):
        sentence = sentence.strip()
        if sentence:
            units.extend(_clauses(sentence))
    return units


def _clauses(sentence: str) -> list[str]:
    """One sentence, split where it plainly carries more than one claim.

    Conservative on purpose. Splitting too eagerly produces fragments the model
    then cannot support from any passage, and a false alarm teaches a reader to
    ignore the flags — which costs the catches their value.
    """
    parts = [part.strip() for part in _CLAUSE.split(sentence) if part.strip()]
    if len(parts) < 2 or any(len(part) < _MIN_CLAUSE for part in parts):
        return [sentence]
    return parts


def annotate(answer: str, unsupported: list[str]) -> str:
    """The answer with unsupported sentences marked in place.

    Marked, not removed. A reader can see what the model claimed and that the
    evidence did not carry it, which is more useful than a shorter answer with
    no explanation of what is missing from it.
    """
    if not unsupported:
        return answer

    marked = answer
    for sentence in unsupported:
        marked = marked.replace(sentence, f"{sentence} [unsupported]", 1)
    return marked


def verify(
    answer: str,
    passages: list[str],
    *,
    backend=None,
    local_only: bool = False,
) -> Verdict:
    """Check each sentence of ``answer`` against ``passages``."""
    sentences = split_sentences(answer or "")
    if not sentences:
        return Verdict(answer=answer, skipped=True, reason="nothing to check")
    if not passages:
        # Every sentence is unsupported when there is no evidence at all, but
        # saying so sentence by sentence implies a check that did not happen.
        return Verdict(
            answer=answer,
            skipped=True,
            reason="no passages were retrieved, so nothing could be checked",
        )

    if backend is None:
        from pipeline.extract.llm import VERIFY, get_backend

        # Its own role, defaulting to the extraction model rather than the
        # agent one. Measured separately (bench/verify.py): qwen2.5:3b catches
        # 18 of 22 unsupported claims where llama3.2:3b catches 10 — it agrees
        # with almost anything, the same disposition that makes it stop when
        # told to and a liability here. Set LLM_VERIFY_BACKEND=groq to put the
        # one call that most needs a bigger model on one.
        backend = get_backend(local_only=local_only, role=VERIFY)

    numbered = "\n".join(f"{n}. {text}" for n, text in enumerate(sentences, 1))
    evidence = "\n\n".join(f"[{n}] {text}" for n, text in enumerate(passages, 1))

    started = time.perf_counter()
    try:
        response = backend.complete_json(
            prompt=_INSTRUCTION,
            content=f"SENTENCES:\n{numbered}\n\nPASSAGES:\n{evidence}",
            schema_hint={"unsupported": "list of integers", "note": "string"},
            json_schema=_SCHEMA,
        )
    except Exception as exc:
        # A verifier that fails must not fail the answer. It must say it did not
        # run, so the caller knows the answer is unchecked rather than clean.
        log.warning("agents.verify.failed", error=repr(exc))
        metrics.incr("agents.verify.failed")
        return Verdict(answer=answer, skipped=True, reason=f"the check failed: {exc}")

    elapsed = time.perf_counter() - started
    flagged = _unsupported(response.data.get("supported"), sentences)

    log.info(
        "agents.verify.done",
        sentences=len(sentences),
        unsupported=len(flagged),
        seconds=round(elapsed, 1),
    )
    metrics.incr("agents.verify.runs")
    return Verdict(
        answer=annotate(answer, flagged),
        unsupported=flagged,
        checked=len(sentences),
        note=str(response.data.get("note") or "").strip(),
        seconds=elapsed,
        caveat=_caveat_for(backend) if flagged else "",
    )


def _caveat_for(backend: Any) -> str:
    """How far to trust a flag from this backend.

    Only the local model's rates have been measured here. A hosted model is
    very likely better at the paraphrase case that produces most of the false
    alarms, but "likely better" is not a number, so it gets no claim either way.
    """
    name = str(getattr(backend, "name", "") or "")
    if name.startswith("ollama"):
        return LOCAL_CAVEAT
    return (
        "Advisory. Flags mark claims the cited passages did not clearly "
        "support; they are not a judgement that a claim is false."
    )


def _unsupported(raw: Any, sentences: list[str]) -> list[str]:
    """The sentences the model did not vouch for.

    Models return 0-based indices, strings, and numbers past the end. None of
    those is worth failing a run over: anything unreadable simply does not count
    as a vouch, which flags the sentence — the safe direction for a check whose
    output is an annotation.
    """
    vouched: set[int] = set()
    if isinstance(raw, list):
        for item in raw:
            try:
                vouched.add(int(item))
            except (TypeError, ValueError):
                continue

    return [
        sentence
        for number, sentence in enumerate(sentences, 1)
        if number not in vouched
    ]


__all__ = ["Verdict", "annotate", "split_sentences", "verify"]
