"""Reading ``SKILL.md`` files into :class:`Skill` objects.

The format is YAML frontmatter and a markdown body, which is the convention the
rest of this project's tooling already uses and, more usefully, one a person who
does not write Python can edit. The body becomes the agent's system prompt
verbatim.

**Nothing here executes anything a skill file says.** The frontmatter goes
through ``yaml.safe_load``; the body is prompt text and is never imported,
evaluated or run. That is deliberate rather than incidental — a folder of
drop-in files that can run code is an obvious way to get code run, and a skill
buys nothing from being executable that a prompt and a tool list do not already
give it.

For the same reason skills are read only from the configured directory inside
this repository. The body reaches a model's system prompt, so a skill file
carries the trust level of ``roles.py`` itself, not that of a fetched page.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from config import config
from observability import get_logger

from pipeline.agents.tools import Effect, catalog
from pipeline.skills.models import BuildAgent, Skill, SkillError

log = get_logger("skills.loader")

#: The filename a skill folder must use. Anything else in the folder — notes, a
#: README, a sample document — is left alone.
SKILL_FILE = "SKILL.md"

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)

_REQUIRED = ("name", "description")

_EFFECTS = {effect.value: effect for effect in Effect}


def _known_tools() -> set[str]:
    """Every registered tool name, at every effect.

    Checked against the whole registry rather than against what the current
    budget permits: a skill naming ``crawl_site`` is legitimate and simply
    yields a narrower catalog on a read-only run. A skill naming
    ``search_corpuss`` is a typo, and should be a load error rather than a
    capability that silently never appears.
    """
    return {spec.name for spec in catalog(list(Effect))}


def parse(text: str, *, source: str = "<string>") -> Skill:
    """One skill from the contents of a ``SKILL.md``."""
    match = _FRONTMATTER.match(text)
    if not match:
        raise SkillError(
            f"{source}: no frontmatter. A skill starts with a '---' line, "
            "some YAML, another '---', and then the prompt."
        )

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise SkillError(f"{source}: the frontmatter is not valid YAML — {exc}") from exc

    if not isinstance(meta, dict):
        raise SkillError(f"{source}: the frontmatter must be a mapping of settings")

    body = match.group(2).strip()
    if not body:
        raise SkillError(
            f"{source}: the body is empty. It becomes the agent's system "
            "prompt, and an agent with no instructions is the generic one."
        )

    for key in _REQUIRED:
        if not str(meta.get(key) or "").strip():
            raise SkillError(f"{source}: '{key}' is required and must not be empty")

    name = str(meta["name"]).strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise SkillError(
            f"{source}: name {name!r} must be lowercase letters, digits, "
            "hyphens and underscores — it is used in URLs and API arguments"
        )

    effect_name = str(meta.get("requires", "read")).strip().lower()
    if effect_name not in _EFFECTS:
        raise SkillError(
            f"{source}: requires must be one of {', '.join(sorted(_EFFECTS))}; "
            f"got {effect_name!r}"
        )

    tools = tuple(_strings(meta.get("tools"), source, "tools"))
    if not tools:
        raise SkillError(f"{source}: 'tools' must name at least one tool")

    unknown = [tool for tool in tools if tool not in _known_tools()]
    if unknown:
        raise SkillError(
            f"{source}: no tool named {', '.join(repr(t) for t in unknown)}. "
            f"Available: {', '.join(sorted(_known_tools()))}"
        )

    extraction = meta.get("extraction") or {}
    if not isinstance(extraction, dict):
        raise SkillError(f"{source}: 'extraction' must be a mapping")

    schema_hint = extraction.get("schema") or {}
    if not isinstance(schema_hint, dict):
        raise SkillError(f"{source}: 'extraction.schema' must map field names to types")

    return Skill(
        agents=_agents(meta.get("agents"), source),
        name=name,
        description=" ".join(str(meta["description"]).split()),
        triggers=tuple(_strings(meta.get("triggers"), source, "triggers")),
        tools=tools,
        system=body,
        requires=_EFFECTS[effect_name],
        schema_hint=schema_hint,
        entity_types=tuple(_strings(extraction.get("entity_types"), source, "entity_types")),
        relation_types=tuple(
            _strings(extraction.get("relation_types"), source, "relation_types")
        ),
    )


def _agents(value: Any, source: str) -> tuple[BuildAgent, ...]:
    """The build roster, in declaration order.

    Order is the running order, and it matters: a later agent is shown what the
    earlier ones wrote. Sorting these would quietly change what each one sees.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise SkillError(f"{source}: 'agents' must be a list")

    roster: list[BuildAgent] = []
    seen: set[str] = set()
    for index, entry in enumerate(value):
        where = f"{source}: agents[{index}]"
        if not isinstance(entry, dict):
            raise SkillError(f"{where} must be a mapping with name and purpose")
        name = str(entry.get("name") or "").strip()
        purpose = str(entry.get("purpose") or "").strip()
        if not name or not purpose:
            raise SkillError(f"{where} needs both a name and a purpose")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
            raise SkillError(f"{where}: name {name!r} must be lowercase and url-safe")
        if name in seen:
            raise SkillError(f"{where}: {name!r} is declared twice")
        seen.add(name)
        roster.append(
            BuildAgent(
                name=name,
                purpose=" ".join(purpose.split()),
                writes=tuple(_strings(entry.get("writes"), where, "writes")),
            )
        )
    return tuple(roster)


def _strings(value: Any, source: str, field: str) -> list[str]:
    """A YAML list of strings, tolerating a single string and stray blanks."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise SkillError(f"{source}: '{field}' must be a list")
    return [str(entry).strip() for entry in value if str(entry).strip()]


def load_file(path: Path) -> Skill:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillError(f"{path}: could not be read — {exc}") from exc
    skill = parse(text, source=str(path))
    return Skill(**{**_fields(skill), "path": path})


def _fields(skill: Skill) -> dict[str, Any]:
    return {
        "name": skill.name,
        "description": skill.description,
        "triggers": skill.triggers,
        "tools": skill.tools,
        "system": skill.system,
        "requires": skill.requires,
        "schema_hint": skill.schema_hint,
        "entity_types": skill.entity_types,
        "relation_types": skill.relation_types,
        "agents": skill.agents,
    }


def skills_dir() -> Path:
    """Where skills live. Relative paths resolve against the repository root,
    so a worker started from anywhere reads the same folder."""
    configured = Path(config.SKILLS_DIR).expanduser()
    if configured.is_absolute():
        return configured
    return (Path(__file__).resolve().parents[2] / configured).resolve()


def drafts_dir(directory: Optional[Path] = None) -> Path:
    """Where a synthesized skill waits to be read."""
    return (directory or skills_dir()) / config.SKILLS_DRAFTS_DIR


def discover(directory: Optional[Path] = None) -> list[Path]:
    """Every ``SKILL.md`` under ``directory``, one folder deep, sorted.

    Folders whose name starts with an underscore are skipped. ``_drafts`` holds
    bare ``<name>.md`` files and so would be passed over anyway — but a draft is
    a model's proposal for an agent's own system prompt, and "invisible because
    of how it happens to be named" is not the property to rest that on.
    """
    root = directory or skills_dir()
    if not root.is_dir():
        return []
    return sorted(
        child / SKILL_FILE
        for child in root.iterdir()
        if child.is_dir()
        and not child.name.startswith("_")
        and (child / SKILL_FILE).is_file()
    )


#: Parsed skills, keyed by directory, with the fingerprint they were read at.
_cache: dict[str, tuple[Any, dict[str, Skill]]] = {}


def _fingerprint(paths: Iterable[Path]) -> tuple:
    """What must change for a reload. Path and mtime, not content — an edited
    skill should appear without a restart, and hashing every file on every
    lookup would put a read on the path that answers every request."""
    return tuple((str(path), path.stat().st_mtime_ns) for path in paths)


def load_all(directory: Optional[Path] = None, *, force: bool = False) -> dict[str, Skill]:
    """Every valid skill, by name.

    One bad file does not hide the others: it is logged and skipped, because a
    typo in an experimental skill should not take the whole system's domain
    knowledge offline. A caller that wants the error should use
    :func:`load_file`.
    """
    if not config.SKILLS_ENABLED:
        return {}

    root = directory or skills_dir()
    paths = discover(root)
    key = str(root)
    stamp = _fingerprint(paths)

    if not force:
        cached = _cache.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]

    loaded: dict[str, Skill] = {}
    for path in paths:
        try:
            skill = load_file(path)
        except SkillError as exc:
            log.warning("skills.invalid", path=str(path), error=str(exc))
            continue
        if skill.name in loaded:
            log.warning(
                "skills.duplicate",
                name=skill.name,
                path=str(path),
                kept=str(loaded[skill.name].path),
                effect="the later file is ignored",
            )
            continue
        loaded[skill.name] = skill

    _cache[key] = (stamp, loaded)
    log.info("skills.loaded", count=len(loaded), directory=key)
    return loaded


def get(name: str, directory: Optional[Path] = None) -> Skill:
    skills = load_all(directory)
    try:
        return skills[name]
    except KeyError:
        available = ", ".join(sorted(skills)) or "none"
        raise SkillError(f"no skill named {name!r}; available: {available}") from None


def reset() -> None:
    """Drop the cache. For tests, and for a reload after editing a file."""
    _cache.clear()


__all__ = [
    "SKILL_FILE",
    "discover",
    "drafts_dir",
    "get",
    "load_all",
    "load_file",
    "parse",
    "reset",
    "skills_dir",
]
