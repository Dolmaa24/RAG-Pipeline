"""Catching relationships the model got backwards.

``Northwind ACQUIRED Fabrikam`` and its reverse are not variations on a fact,
they are a fact and a falsehood. A graph that stores the wrong one answers "who
acquired Fabrikam" confidently and wrongly, and nothing downstream can tell —
the graph *is* the source of truth for relationships, so there is nothing to
check it against later. Measured on this pipeline's benchmark, ``llama3.2:3b``
reversed two of four directional edges and ``qwen2.5:3b`` reversed none, which
is exactly the kind of gap that closes when you stop hoping and start checking.

Two signals, because neither is sufficient alone:

**Type asymmetry.** ``FOUNDED`` runs from a person or an organisation *to* an
organisation or a project. So ``(Project Titan)-[STARTED]->(Apple)`` is
structurally impossible, and the flip is structurally fine — that is a
correction you can make without reading the sentence.

**Surface order.** Type rules cannot help when both endpoints are the same kind:
``(Fabrikam)-[ACQUIRED]->(Northwind)`` is two organisations either way. But the
sentence that produced it says "Northwind Traders acquired Fabrikam Ltd", and
the subject of an active clause precedes its verb. Passive voice inverts that —
"Fabrikam was acquired by Northwind" — so the agent after *by* is the true
source, and the check looks for that before deciding.

The rule throughout: **only act when one orientation fits and the other does
not.** An edge whose direction is genuinely ambiguous is left exactly as the
model wrote it. Flipping on a guess would replace one wrong-fact generator with
another.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from observability import get_logger, metrics

from pipeline.graph.schema import Relationship

log = get_logger("graph.validate")


class Verdict(str, Enum):
    OK = "ok"
    REVERSED = "reversed"
    #: No rule applies, or both orientations are equally consistent.
    UNKNOWN = "unknown"


#: The extractor's ``type`` field is free text, so it is bucketed before use.
_TYPE_BUCKETS: dict[str, tuple[str, ...]] = {
    "person": ("person", "people", "individual", "human", "employee", "executive"),
    "org": (
        "organization", "organisation", "company", "corporation", "corp",
        "business", "firm", "institution", "agency", "university", "team",
        "group", "employer",
    ),
    "place": ("location", "place", "city", "country", "region", "state", "address"),
    "project": ("project", "initiative", "programme", "program", "effort", "mission"),
    "thing": (
        "product", "technology", "service", "platform", "device", "software",
        "concept", "field", "vehicle", "system",
    ),
}


def bucket(raw_type: str) -> str:
    """Map a free-text entity type onto one of a handful of buckets."""
    text = (raw_type or "").strip().lower()
    if not text:
        return "unknown"
    for name, words in _TYPE_BUCKETS.items():
        if any(word in text for word in words):
            return name
    return "unknown"


@dataclass(frozen=True)
class Rule:
    """One directional relation and the shape it must have."""

    #: Matched as substrings of the relation name, which is UPPER_SNAKE_CASE.
    verbs: tuple[str, ...]
    source: frozenset[str]
    target: frozenset[str]
    #: Stems to look for in the source text for the surface-order check.
    surface: tuple[str, ...] = ()

    def matches(self, relation: str) -> bool:
        name = (relation or "").upper()
        return any(verb in name for verb in self.verbs)


_PERSON = frozenset({"person"})
_ORG = frozenset({"org"})
_ORG_OR_PERSON = frozenset({"org", "person"})
_ORG_PROJECT_THING = frozenset({"org", "project", "thing"})

RULES: tuple[Rule, ...] = (
    Rule(("ACQUIR", "BOUGHT", "PURCHAS", "TOOK_OVER"), _ORG_OR_PERSON, _ORG,
         ("acquir", "bought", "purchas")),
    Rule(("FOUND", "STARTED", "CREATED", "LAUNCH", "ESTABLISH"),
         _ORG_OR_PERSON, _ORG_PROJECT_THING,
         ("found", "started", "created", "launch", "establish")),
    Rule(("WORK", "EMPLOY", "JOIN", "HIRED_BY"), _PERSON, _ORG,
         ("work", "employ", "join")),
    Rule(("HIRED", "RECRUIT"), _ORG_OR_PERSON, _PERSON, ("hired", "recruit")),
    Rule(("LED", "LEAD", "MANAG", "HEAD", "DIRECT", "OVERSAW", "SUPERVIS"),
         _PERSON, frozenset({"org", "project", "person", "thing"}),
         ("led", "lead", "manag", "head", "oversaw", "supervis")),
    Rule(("LOCATED", "HEADQUARTER", "BASED"), frozenset({"org", "person", "project"}),
         frozenset({"place"}), ("located", "headquarter", "based")),
    Rule(("SUBSIDIARY", "PART_OF", "DIVISION_OF", "OWNED_BY"), _ORG, _ORG,
         ("subsidiary", "part of", "division of", "owned by")),
    Rule(("INVEST", "FUNDED"), _ORG_OR_PERSON, _ORG, ("invest", "funded")),
    Rule(("DEPART", "LEFT", "RESIGN"), _PERSON, _ORG, ("depart", "left", "resign")),
    Rule(("REPORT",), _PERSON, _PERSON, ("report",)),
    Rule(("DEVELOP", "BUILT", "PRODUCE", "MAKES", "MANUFACTUR"),
         _ORG_OR_PERSON, frozenset({"project", "thing"}),
         ("develop", "built", "produce", "manufactur")),
)


def rule_for(relation: str) -> Optional[Rule]:
    for rule in RULES:
        if rule.matches(relation):
            return rule
    return None


#: A passive clause puts the agent after "by": "was acquired by Northwind".
_PASSIVE_WINDOW = 24


def check_types(relation: Relationship, types: dict[str, str]) -> Verdict:
    """Whether the endpoints' types fit the relation, in this order."""
    rule = rule_for(relation.relation)
    if rule is None:
        return Verdict.UNKNOWN

    source = bucket(types.get(relation.source, ""))
    target = bucket(types.get(relation.target, ""))
    if "unknown" in (source, target):
        return Verdict.UNKNOWN

    forward = source in rule.source and target in rule.target
    reverse = target in rule.source and source in rule.target

    if forward and reverse:
        # The rule does not discriminate here — ``ACQUIRED`` between two
        # organisations is type-consistent whichever way round it is written.
        # Saying OK would be claiming evidence this check does not have, and
        # would stop the surface-order check from ever being asked.
        return Verdict.UNKNOWN
    if forward:
        return Verdict.OK
    if reverse:
        return Verdict.REVERSED
    return Verdict.UNKNOWN


def check_order(relation: Relationship, text: str) -> Verdict:
    """Whether the sentence puts the endpoints in the order the edge claims.

    Only decides when both names and the verb are actually present and one
    reading fits. Everything else is :attr:`Verdict.UNKNOWN`.
    """
    rule = rule_for(relation.relation)
    if rule is None or not rule.surface or not text:
        return Verdict.UNKNOWN

    haystack = text.lower()
    source_at = haystack.find(relation.source.lower())
    target_at = haystack.find(relation.target.lower())
    if source_at < 0 or target_at < 0 or source_at == target_at:
        return Verdict.UNKNOWN

    verb_at = -1
    verb_end = -1
    for stem in rule.surface:
        found = haystack.find(stem)
        if found >= 0 and (verb_at < 0 or found < verb_at):
            verb_at, verb_end = found, found + len(stem)
    if verb_at < 0:
        return Verdict.UNKNOWN

    # The verb has to sit between the two names for the order to mean anything.
    first, second = sorted((source_at, target_at))
    if not (first < verb_at < second):
        return Verdict.UNKNOWN

    passive = " by " in haystack[verb_end : verb_end + _PASSIVE_WINDOW]
    subject_first = source_at < target_at

    if passive:
        # "Fabrikam was acquired by Northwind" — the agent after "by" is source.
        return Verdict.REVERSED if subject_first else Verdict.OK
    return Verdict.OK if subject_first else Verdict.REVERSED


@dataclass
class Correction:
    """One edge whose direction was changed, and why."""

    relation: str
    was: tuple[str, str]
    now: tuple[str, str]
    evidence: str

    def describe(self) -> str:
        return (
            f"({self.was[0]})-[{self.relation}]->({self.was[1]}) "
            f"corrected to ({self.now[0]})-[{self.relation}]->({self.now[1]}) "
            f"by {self.evidence}"
        )


def validate(
    relationships: Iterable[Relationship],
    *,
    types: Optional[dict[str, str]] = None,
    text: str = "",
) -> tuple[list[Relationship], list[Correction]]:
    """Return the relationships with reversed ones flipped, and what changed.

    Type evidence outranks surface order: it is structural, while word order is
    a heuristic about the sentence that happened to produce the edge.
    """
    types = types or {}
    out: list[Relationship] = []
    corrections: list[Correction] = []

    for relation in relationships:
        verdict = check_types(relation, types)
        evidence = "entity types"
        if verdict is Verdict.UNKNOWN:
            verdict = check_order(relation, text)
            evidence = "word order in the source text"

        if verdict is not Verdict.REVERSED:
            out.append(relation)
            continue

        flipped = relation.model_copy(
            update={"source": relation.target, "target": relation.source}
        )
        corrections.append(
            Correction(
                relation=relation.relation,
                was=(relation.source, relation.target),
                now=(flipped.source, flipped.target),
                evidence=evidence,
            )
        )
        out.append(flipped)

    if corrections:
        metrics.incr("graph.direction_corrected", len(corrections))
        for correction in corrections:
            log.info("graph.direction_corrected", detail=correction.describe())
    return out, corrections


__all__ = [
    "Correction",
    "RULES",
    "Rule",
    "Verdict",
    "bucket",
    "check_order",
    "check_types",
    "rule_for",
    "validate",
]
