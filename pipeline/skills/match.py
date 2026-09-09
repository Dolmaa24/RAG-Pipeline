"""Deciding which domain an intent belongs to.

Two steps, and no model call in either. ``pipeline/agents/route.py`` already
argues the reason at length: paying a model to decide whether to pay for a model
is the most avoidable latency there is. It applies twice over here, because a
routing decision that a model makes is a routing decision no test can assert.

**Triggers first.** A skill names the words that mean its domain outright, and
an intent containing one of them is not ambiguous. This costs microseconds and
handles most real intents, because people asking about insurance say
"insurance".

**Embeddings second**, when no trigger fires or two skills tie. The intent is
embedded and compared against each skill's description. This is what catches
"my child's attendance record" without ``school`` appearing in it. The embedder
is already resident in the agents worker (``AGENT_WARM_EMBEDDER``), so this is a
matrix multiply against a handful of vectors rather than a model load.

**No match is not a failure.** It returns a match with no skill, and the caller
runs the generic corpus specialist — the behaviour the system had before skills
existed. That makes the floor equal to today rather than worse than it, which
matters because a wrong skill is more expensive than no skill: it hands the
agent a prompt about the wrong domain and a tool subset chosen for it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from config import config
from observability import get_logger, metrics

from pipeline.skills.loader import load_all
from pipeline.skills.models import Skill

log = get_logger("skills.match")

_WORD = re.compile(r"[a-z0-9]+")

#: A phrase trigger is worth more than a word trigger. "sum insured" appearing
#: in an intent is far stronger evidence than "claim", which turns up in
#: ordinary speech about anything.
_PHRASE_WEIGHT = 2.0
_WORD_WEIGHT = 1.0


@dataclass(frozen=True, slots=True)
class Match:
    """Which skill an intent needs, how sure, and by what route."""

    skill: Optional[Skill]
    confidence: float = 0.0
    #: "trigger", "embedding", "explicit", or "none".
    how: str = "none"
    #: Everything else that scored, best first. Surfaced rather than discarded:
    #: a near-miss is the useful half of a wrong match.
    runners_up: list[tuple[str, float]] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return self.skill is not None

    def explain(self) -> str:
        if self.skill is None:
            return "no skill matched; the generic corpus specialist will run"
        if self.how == "explicit":
            return f"{self.skill.name} was named in the request"
        if self.how == "trigger":
            return f"{self.skill.name} matched on its own trigger words"
        return f"{self.skill.name} was the closest description to the intent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill.name if self.skill else None,
            "confidence": round(self.confidence, 3),
            "how": self.how,
            "why": self.explain(),
            "runners_up": [
                {"skill": name, "score": round(score, 3)} for name, score in self.runners_up
            ],
        }


def _singular(word: str) -> str:
    """A crude singular, applied to *both* sides so it need only be consistent.

    Without it "which policies cover physiotherapy" misses the trigger
    "policy" and falls through to embeddings, which is slower and less certain
    than the answer the trigger already had. A real stemmer would be a
    dependency and a model download for a rule that has to survive being
    wrong: "clas" is not a word, but "classes" and "class" both reduce to it,
    and matching is all this is for.
    """
    if len(word) <= 3 or word.endswith("ss") or word.endswith("us"):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def _normalise(text: str) -> str:
    return " ".join(_singular(word) for word in _WORD.findall((text or "").lower()))


def _trigger_scores(intent: str, skills: dict[str, Skill]) -> dict[str, float]:
    """How strongly each skill's own vocabulary appears in the intent.

    Matched on word boundaries against a normalised copy, so "policyholder"
    does not count as the trigger "policy" and "claiming" does not count as
    "claim". Substring matching looked simpler and scored `ecommerce` on
    "the school ordered new textbooks" through "order".
    """
    haystack = f" {_normalise(intent)} "
    scores: dict[str, float] = {}
    for name, skill in skills.items():
        total = 0.0
        for trigger in skill.triggers:
            cleaned = _normalise(trigger)
            if not cleaned:
                continue
            if f" {cleaned} " in haystack:
                total += _PHRASE_WEIGHT if " " in cleaned else _WORD_WEIGHT
        if total:
            scores[name] = total
    return scores


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


#: Description vectors, keyed by embedder name and description text, so an
#: edited skill re-embeds and a swapped embedder does not compare vectors from
#: two different models.
_vectors: dict[tuple[str, str], list[float]] = {}


def _embedding_scores(intent: str, skills: dict[str, Skill], embedder) -> dict[str, float]:
    if embedder is None:
        from pipeline.embed.dense import get_dense_embedder

        embedder = get_dense_embedder()

    model = getattr(embedder, "model_name", getattr(embedder, "name", "?"))
    #: The triggers join the description: they are the domain's vocabulary, and
    #: a description alone is a sentence about the domain rather than in it.
    texts = {
        name: f"{skill.description} {' '.join(skill.triggers)}".strip()
        for name, skill in skills.items()
    }

    missing = [text for text in texts.values() if (model, text) not in _vectors]
    if missing:
        for text, vector in zip(missing, embedder.embed_documents(missing)):
            _vectors[(model, text)] = vector

    query = embedder.embed_query(intent)
    return {
        name: _cosine(query, _vectors[(model, text)]) for name, text in texts.items()
    }


def _ranked(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def match(
    intent: str,
    *,
    skill: Optional[str] = None,
    skills: Optional[dict[str, Skill]] = None,
    embedder=None,
) -> Match:
    """Which skill should handle ``intent``.

    ``skill`` names one outright and skips the routing, for when the match was
    wrong and a person is correcting it. An unknown name raises, because a
    caller who asked for a particular skill should be told it does not exist
    rather than quietly given a different one.
    """
    available = load_all() if skills is None else skills

    if skill:
        chosen = available.get(skill)
        if chosen is None:
            from pipeline.skills.models import SkillError

            names = ", ".join(sorted(available)) or "none"
            raise SkillError(f"no skill named {skill!r}; available: {names}")
        return Match(chosen, confidence=1.0, how="explicit")

    text = (intent or "").strip()
    if not text or not available:
        return Match(None)

    triggers = _trigger_scores(text, available)
    ranked = _ranked(triggers)
    #: A tie means the intent named two domains equally. That is the case
    #: embeddings are better at than counting is, so it falls through rather
    #: than picking the alphabetically-first of the two.
    if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]):
        name, score = ranked[0]
        metrics.incr("skills.match.trigger")
        log.info("skills.matched", skill=name, how="trigger", score=score)
        return Match(available[name], confidence=score, how="trigger", runners_up=ranked[1:4])

    try:
        similarity = _embedding_scores(text, available, embedder)
    except Exception as exc:
        # An unavailable embedder must not fail a task. The generic specialist
        # is a worse answer than the right skill and a much better one than an
        # error, and this is exactly the position the shared rate limiter takes
        # about Redis.
        log.warning("skills.embedding_unavailable", error=repr(exc))
        return Match(None, runners_up=ranked[:4])

    ordered = _ranked(similarity)
    best, score = ordered[0]
    #: The margin, not the score, is the test. Every skill describes a domain
    #: of documents, so any question about documents scores middlingly against
    #: all of them -- "who wrote this document" reached 0.508 against `school`,
    #: which is inside the range genuine domain intents occupy. What a real
    #: domain intent does that a generic one does not is *pull away from the
    #: rest*, and that is scale-free where an absolute cut is not.
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = score - runner_up
    if score < config.SKILLS_MIN_SIMILARITY or margin < config.SKILLS_MIN_MARGIN:
        metrics.incr("skills.match.none")
        log.info(
            "skills.unmatched",
            best=best,
            score=round(score, 3),
            margin=round(margin, 3),
        )
        return Match(None, confidence=score, runners_up=ordered[:4])

    metrics.incr("skills.match.embedding")
    log.info("skills.matched", skill=best, how="embedding", score=round(score, 3))
    return Match(available[best], confidence=score, how="embedding", runners_up=ordered[1:4])


def reset() -> None:
    """Drop cached description vectors."""
    _vectors.clear()


__all__ = ["Match", "match", "reset"]
