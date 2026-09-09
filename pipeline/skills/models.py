"""A domain pack, and the agent it composes.

``pipeline.agents.roles`` states the principle this builds on: a role is data,
not a class hierarchy — a name, a purpose, the tools it may use, and a prompt.
Everything else is shared. Two roles were enough while the pipeline was
domain-blind; a corpus of clinical guidance and a corpus of shop listings want
different prompts, different extraction fields and differently-typed entities,
and none of that is a code change worth making four times.

So a skill is a role read from a file, plus what the extraction and graph
stages need to know about the domain. :meth:`Skill.as_role` is the whole of
"creating an agent": it returns an ordinary :class:`~pipeline.agents.roles.Role`
that the existing supervisor, loop and budget already know how to run.

**A skill narrows; it never widens.** ``Role.catalog()`` intersects the declared
tools with the effects the run's budget permits, so a file that names
``crawl_site`` gets it stripped under a read-only budget, and a skill declaring
``requires: write`` is absent from a read-only run rather than offered and
refused. The request grants effects. A file in a folder does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pipeline.agents.roles import Role
from pipeline.agents.tools import Effect


@dataclass(frozen=True, slots=True)
class BuildAgent:
    """One worker in a domain's roster.

    A café has a receptionist, a machine, a delivery hand and a cleaner. Those
    are the domain's own actors, and they are the natural decomposition of the
    code that models it -- one module each, one model call each.

    Declared in the skill file rather than invented per run. A roster a model
    re-derives every time is a roster nothing can review, diff or assert on,
    and it would be the one part of a build that changed under you between two
    runs of the same intent.
    """

    name: str
    purpose: str
    #: The files this agent owns. One model call per file, because both
    #: backends cap output at 4096 tokens and a whole application does not fit
    #: in one -- a truncated file is the failure this avoids.
    writes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "purpose": self.purpose, "writes": list(self.writes)}


@dataclass(frozen=True, slots=True)
class Skill:
    """One domain, as loaded from its ``SKILL.md``."""

    name: str
    #: Prose, matched against an intent when no trigger word fires. Written for
    #: a similarity comparison rather than for a person, so it should name the
    #: domain's own vocabulary rather than describe the file.
    description: str
    #: Words and phrases that mean this domain outright.
    triggers: tuple[str, ...]
    tools: tuple[str, ...]
    system: str
    requires: Effect = Effect.READ
    #: Field names to types, in the friendly form ``compile_schema`` accepts.
    #: Fills ``schema_template`` for an extraction that names this skill.
    schema_hint: dict[str, Any] = field(default_factory=dict)
    #: What the graph extractor should look for in this domain.
    entity_types: tuple[str, ...] = ()
    relation_types: tuple[str, ...] = ()
    #: The workers a build spins up for this domain, in the order they run.
    #: Empty for a skill that only answers questions, which is all four of the
    #: shipped ones -- and a build of such a skill is refused rather than
    #: producing an empty workspace.
    agents: tuple[BuildAgent, ...] = ()
    path: Optional[Path] = None

    def as_role(self) -> Role:
        """This skill as a specialist the supervisor can run.

        A plain ``Role``. Nothing about the loop, the budget or the registry
        needs to know a skill exists.
        """
        return Role(
            name=self.name,
            purpose=self.description,
            tools=self.tools,
            system=self.system,
            requires=self.requires,
        )

    @property
    def buildable(self) -> bool:
        return bool(self.agents)

    def graph_guidance(self) -> str:
        """What to add to the graph extractor's prompt for this domain, if
        anything. Empty when the skill names no types, so a skill that only
        answers questions leaves graph extraction exactly as it was."""
        parts = []
        if self.entity_types:
            parts.append(
                "In this domain the entities worth naming are typically: "
                + ", ".join(self.entity_types)
                + ". Use another type when the text calls for one."
            )
        if self.relation_types:
            parts.append(
                "Relations commonly stated in this domain: "
                + ", ".join(self.relation_types)
                + ". Use another when the sentence states one."
            )
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "triggers": list(self.triggers),
            "tools": list(self.tools),
            "requires": self.requires.value,
            "schema": self.schema_hint,
            "entity_types": list(self.entity_types),
            "relation_types": list(self.relation_types),
            "agents": [agent.to_dict() for agent in self.agents],
            "buildable": self.buildable,
        }


class SkillError(ValueError):
    """A skill file could not be loaded. Always names the file."""


__all__ = ["BuildAgent", "Skill", "SkillError"]
