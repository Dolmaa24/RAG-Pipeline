"""Domain packs: a folder of SKILL.md files, matched to an intent.

See :mod:`pipeline.skills.models` for what a skill is and why it composes into
an ordinary :class:`~pipeline.agents.roles.Role` rather than a new abstraction.

Note that the ``match`` exported here is the *function*, which shadows the
submodule of the same name — ``from pipeline.skills import match`` gets the
callable. Reach for :mod:`pipeline.skills.match` by its full path when you want
the module.
"""

from pipeline.skills.loader import get, load_all, load_file, parse, skills_dir
from pipeline.skills.loader import reset as _reset_files
from pipeline.skills.match import Match, match
from pipeline.skills.match import reset as _reset_vectors
from pipeline.skills.models import Skill, SkillError


def reset() -> None:
    """Drop both caches: parsed skills, and their description vectors.

    One call, because they go stale together — an edited description that
    reloaded but kept its old vector would match as though it had not changed.
    """
    _reset_files()
    _reset_vectors()


__all__ = [
    "Match",
    "Skill",
    "SkillError",
    "get",
    "load_all",
    "load_file",
    "match",
    "parse",
    "reset",
    "skills_dir",
]
