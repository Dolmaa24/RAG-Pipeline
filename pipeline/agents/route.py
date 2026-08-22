"""Deciding whether a question is worth an investigation.

`bench/agents.py` measured both paths over the same thirteen questions and the
result was narrower than this project had been assuming:

    kind           ask once      investigate
    single_hop     4/4 · 6.6s    3/4 · 20.7s
    multi_hop      2/3           2/3
    enumeration    1/3           3/3
    unanswerable   3/3           3/3

The loop earns its four-fold latency on **enumeration** and nowhere else.
"Which acquisitions does the corpus describe" names no entity to start from, so
one retrieval answers it partially or not at all, and the loop can list the
edges instead. Everywhere else the two are within a case of each other.

So this routes: a question asking *which*, *how many*, *list*, or *what kinds*
goes to the loop; everything else is answered directly. No model call — the
decision is made from the question's own words, because paying a model to
decide whether to pay for a model is the most avoidable latency there is, and
because a wrong answer here is cheap in one direction and expensive in the
other.

**Bias towards answering directly.** Routing a genuine enumeration to the fast
path costs one partial answer. Routing an ordinary question to the loop costs
twenty seconds of somebody's attention, on every such question. The patterns
below are therefore narrow, and anything unrecognised takes the fast path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config import config

#: Words ending in "s" that are not plural nouns. Without this the pattern for
#: "which <plural>" matches "what **is** the revenue" and "what **was** the
#: price", routing ordinary lookups to a twenty-second path — which the first
#: version of this file did, for two of the benchmark's own questions.
_NOT_A_PLURAL = (
    "is|was|has|does|goes|says|means|its|his|hers|this|thus|plus|less|"
    "as|us|yes|else|whose|across|business|address|process|status|analysis"
)

#: Determiners and counts that can sit between "which" and the noun, so that
#: "which **two** acquisitions" reads the same as "which acquisitions". Counts
#: are spelled out as well as written: the benchmark's own phrasing is "which
#: two acquisitions", and a digits-only pattern missed it.
_QUANTIFIER = (
    r"(?:\d+\s+|two\s+|three\s+|four\s+|five\s+|six\s+|seven\s+|eight\s+|"
    r"nine\s+|ten\s+|both\s+|several\s+|the\s+|these\s+|those\s+|other\s+|"
    r"remaining\s+|different\s+)*"
)

#: Asking for a set rather than a fact.
_ENUMERATION = re.compile(
    rf"""
    \b(?:
        how\s+many
      | (?:which|what)\s+(?:kinds?|types?|sorts?|categories)\s+of
      | (?:which|what)\s+{_QUANTIFIER}(?!(?:{_NOT_A_PLURAL})\b)[a-z]{{3,}}s\b
      | list\s+(?:all|every|the|each|out)
      | enumerate\b
      | name\s+(?:all|every|each)
      | (?:all|every)\s+(?:of\s+the\s+)?[a-z]{{3,}}s\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: Words that make a plural noun a comparison rather than a set to enumerate.
#: "Which company has the most employees" wants one answer, not a list.
_NOT_ENUMERATION = re.compile(
    r"\b(?:most|least|largest|smallest|highest|lowest|best|worst|cheapest|"
    r"biggest|greatest|fewest)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Route:
    """Where a question should go, and why."""

    investigate: bool
    reason: str

    @property
    def path(self) -> str:
        return "investigate" if self.investigate else "answer"


def route(question: str) -> Route:
    """Whether ``question`` is worth the loop.

    Never raises, and never calls a model. An unrecognised question takes the
    fast path, which is the cheap direction to be wrong in.
    """
    text = (question or "").strip()
    if not text:
        return Route(False, "empty question")

    if not config.AGENT_ROUTE_ENUMERATION:
        return Route(False, "routing is off; answering directly")

    if _NOT_ENUMERATION.search(text):
        return Route(False, "asks which one, not which ones")

    if _ENUMERATION.search(text):
        return Route(True, "asks for a set, which one retrieval answers partially")

    return Route(False, "one retrieval answers this as well as the loop would")


__all__ = ["Route", "route"]
