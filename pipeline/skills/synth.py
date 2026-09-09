"""Drafting a skill for a domain nothing covers, and installing it once read.

The matcher's honest answer to "make a Baristo system" is that no skill fits,
and the generic specialist then searches a corpus that holds no café. The gap
is not the matcher's: there is no barista skill, and nothing ever makes one.

This makes one. A model reads the intent and proposes the domain -- its
vocabulary, the fields worth extracting from its pages, the entities its graph
would hold, the prompt its specialist should carry, and the roster of workers a
build would spin up. The proposal is then **written to a drafts folder the
loader ignores**, and becomes a real skill only when a person moves it across.

That gate is the point. A skill's body is an agent's system prompt, so
auto-installing a synthesized one would hand the model authorship of its own
instructions -- and there is no reviewing that after the fact, because by then
it has already run.

**JSON in, markdown out.** The model is asked for an object against a real JSON
Schema, not for a ``SKILL.md``. The frontmatter is then rendered here by code
that cannot produce broken YAML, which asking a 3B model to write by hand is a
failure mode with no upside.

The object is fetched through the **tool-calling** path rather than
``complete_json``, and that is not an arbitrary choice. ``complete_json`` is an
extraction interface, and its system prompt says so in as many words: *"Copy
values from the content. Do not infer, calculate, or complete them from your own
knowledge"*, and *"if a value is genuinely not present in the content, use
null"*. Asked to describe a café through that door, ``openai/gpt-oss-120b``
returned every field null -- correctly, because "make a Baristo system" contains
no triggers and no roster to copy out. Synthesis is precisely the case those
rules forbid. A tool call is the generative door: the arguments are constrained
by the same schema, and the framing asks the model to compose rather than to
find.

Two things a draft may never do, enforced here rather than asked for in the
prompt:

* **``requires`` is forced to read.** A model may not author a skill that
  reaches the network or writes to the corpus. Editing the file afterwards is
  how a person grants that, deliberately.
* **Tools must exist.** The catalog is given in the prompt and checked on the
  way back, so an invented tool name is a generation failure with the name in
  the message, not a skill that silently never works.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from config import config
from observability import get_logger, metrics

from pipeline.agents.tools import Effect, catalog
from pipeline.skills import loader
from pipeline.skills.models import BuildAgent, Skill, SkillError

log = get_logger("skills.synth")

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "triggers": {"type": "array", "items": {"type": "string"}},
        "tools": {"type": "array", "items": {"type": "string"}},
        "system": {"type": "string"},
        "schema": {"type": "object"},
        "entity_types": {"type": "array", "items": {"type": "string"}},
        "relation_types": {"type": "array", "items": {"type": "string"}},
        "agents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "purpose": {"type": "string"},
                    "writes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "purpose", "writes"],
            },
        },
    },
    "required": [
        "name", "description", "triggers", "tools", "system",
        "schema", "entity_types", "relation_types", "agents",
    ],
}

#: The friendly form, for the Ollama path and for the prompt.
_SCHEMA_HINT = {
    "name": "string",
    "description": "string",
    "triggers": "list of strings",
    "tools": "list of strings",
    "system": "string",
    "schema": "object",
    "entity_types": "list of strings",
    "relation_types": "list of strings",
    "agents": "list of objects",
}

#: The schema, as the one tool the model is given. Constrained decoding without
#: the extraction framing.
_PROPOSE_TOOL = {
    "name": "propose_skill",
    "description": (
        "Propose a complete description of a subject domain: its vocabulary, "
        "the fields worth extracting from its pages, the entities its knowledge "
        "graph would hold, its specialist's instructions, and the workers its "
        "software would be built from."
    ),
    "input_schema": _JSON_SCHEMA,
}

_SYSTEM = """\
You describe subject domains. You are composing a description from what you
know about the subject, not copying one out of a document — there is no
document. Every field must be filled with something real and specific to the
domain. Never return null, and never return an empty list.

Call propose_skill exactly once."""

_PROMPT = """\
You are describing a subject domain so that a retrieval system can specialise in
it, and so that a code generator can decompose it into workers.

name — one lowercase word for the domain. Letters, digits, hyphens, underscores.

description — one sentence naming what the domain is about, written in the
domain's own vocabulary. It is compared against future requests by meaning, so
the words that belong to this subject matter more than a tidy sentence does.

triggers — 10 to 20 words and short phrases that mean this domain and little
else. Prefer terms that would be strange in another subject. Singular forms;
plurals are matched automatically.

tools — choose from the list below, by exact name. Nothing else exists.
{tools}

system — the system prompt for a specialist answering questions about this
domain from an indexed corpus. Write it as instructions to that specialist:
what to search for, which of its tools suit which question, the vocabulary the
documents use where it differs from the vocabulary a person would use, and what
it must not do. Several paragraphs. Do not mention this task or these rules.

schema — the fields worth extracting from a page in this domain, as field name
to type. Types are "string", "number", "integer", "list of strings".

entity_types — the kinds of thing this domain's knowledge graph would hold,
capitalised singular nouns.

relation_types — the connections between them, UPPERCASE_SNAKE_CASE verbs.

agents — the workers this domain's software would be built from. Use the
domain's own actors, not job titles from software: a café has a receptionist
taking orders, a machine brewing them, a hand delivering them and a cleaner
sweeping up, and each of those is one module. Three to six of them. Each needs a
lowercase name, a one-sentence purpose, and the source files it owns — one to
two Python files each, because one file is one generation and a long file gets
truncated.

The request:
{intent}"""


def _arguments(turn) -> dict[str, Any]:
    """The proposal out of a turn, whichever way the model expressed it."""
    for call in getattr(turn, "calls", []) or []:
        if call.name == _PROPOSE_TOOL["name"] and isinstance(call.arguments, dict):
            return call.arguments
    # A model that answered with the object as text rather than as a call. The
    # shim already handles this for backends without native tools; this covers
    # a native one that did it anyway.
    text = (getattr(turn, "text", "") or "").strip()
    if text.startswith("{"):
        import json

        from pipeline.extract.llm.base import extract_json_object

        try:
            return json.loads(extract_json_object(text))
        except Exception:
            return {}
    return {}


def _tool_listing() -> str:
    return "\n".join(
        f"  {spec.name} — {spec.description.splitlines()[0]}"
        for spec in catalog([Effect.READ])
    )


@dataclass(frozen=True, slots=True)
class Drafted:
    """A proposal: the parsed skill, and the file it came from.

    Both, because the caller needs both and re-rendering from the parsed skill
    would be a second implementation of the renderer that could disagree with
    the first. The text is what was validated and what gets written.
    """

    skill: Skill
    text: str


def draft(intent: str, *, backend=None, local_only: bool = False) -> Drafted:
    """Propose a skill for ``intent``. Never installs it."""
    text = (intent or "").strip()
    if not text:
        raise SkillError("an intent is required to draft a skill")

    if backend is None:
        from pipeline.extract.llm import INTERACTIVE, get_agent_backend

        # The agent backend, because this goes through the tool-calling path —
        # which also means a model without native tool support is shimmed into
        # constrained JSON rather than failing here.
        backend = get_agent_backend(local_only=local_only, role=INTERACTIVE)

    from pipeline.extract.llm.base import Message

    with metrics.timer("skills.synth"):
        turn = backend.complete_with_tools(
            messages=[
                Message.system(_SYSTEM),
                Message.user(_PROMPT.format(tools=_tool_listing(), intent=text)),
            ],
            tools=[_PROPOSE_TOOL],
            # There is one tool and calling it is the whole task, so the model
            # is not asked to decide whether to.
            tool_choice="required",
        )

    data = _arguments(turn)
    if not isinstance(data, dict) or not data:
        raise SkillError(
            "the model did not propose a skill"
            + (f" — it replied: {turn.text[:200]}" if turn.text else "")
        )

    drafted = _from_payload(data, intent=text)
    log.info(
        "skills.drafted",
        name=drafted.skill.name,
        tools=len(drafted.skill.tools),
        agents=len(drafted.skill.agents),
        backend=getattr(turn, "backend", "?"),
    )
    metrics.incr("skills.synth.ok")
    return drafted


def _from_payload(data: dict[str, Any], *, intent: str) -> Drafted:
    """Turn the model's object into a skill, by rendering and re-parsing it.

    Deliberately round-tripped through :func:`loader.parse` rather than
    constructed directly. That is the same validation a hand-written file gets —
    unknown tools, a bad name, an empty body, a malformed roster — so a drafted
    skill cannot be accepted on a path a hand-written one would fail.
    """
    rendered = render(data, intent=intent)
    try:
        return Drafted(loader.parse(rendered, source="<drafted>"), rendered)
    except SkillError as exc:
        raise SkillError(f"the drafted skill is not valid — {exc}") from exc


def render(data: dict[str, Any], *, intent: str = "") -> str:
    """A SKILL.md from the model's object. Pure, and the only writer of YAML."""
    name = _slug(str(data.get("name") or "").strip())
    frontmatter: dict[str, Any] = {
        "name": name,
        "description": " ".join(str(data.get("description") or "").split()),
        "triggers": _clean_list(data.get("triggers")),
        "tools": _clean_list(data.get("tools")),
        # Never taken from the model. A drafted skill reads and nothing more.
        "requires": "read",
    }

    extraction: dict[str, Any] = {}
    if isinstance(data.get("schema"), dict) and data["schema"]:
        extraction["schema"] = {
            str(k): str(v) for k, v in data["schema"].items() if str(k).strip()
        }
    for field in ("entity_types", "relation_types"):
        values = _clean_list(data.get(field))
        if values:
            extraction[field] = values
    if extraction:
        frontmatter["extraction"] = extraction

    roster = _clean_agents(data.get("agents"))
    if roster:
        frontmatter["agents"] = roster

    # Provenance belongs in the frontmatter, not above the body. The body *is*
    # the agent's system prompt, verbatim — a note to a human reviewer placed
    # there would be sent to the model as part of its instructions on every
    # run, and it would also make an empty prompt look like a non-empty file.
    if intent.strip():
        frontmatter["drafted_from"] = " ".join(intent.split())

    body = str(data.get("system") or "").strip()
    header = yaml.safe_dump(
        frontmatter, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).strip()
    return f"---\n{header}\n---\n\n{body}\n"


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")
    return cleaned or "untitled"


def _clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for entry in value:
        text = " ".join(str(entry).split())
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out


def _clean_agents(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        name = _slug(str(entry.get("name") or "").strip())
        purpose = " ".join(str(entry.get("purpose") or "").split())
        if not purpose or name == "untitled":
            continue
        out.append(
            {"name": name, "purpose": purpose, "writes": _clean_list(entry.get("writes"))}
        )
    return out


# --- the drafts folder -----------------------------------------------------


def _draft_path(name: str, directory: Optional[Path] = None) -> Path:
    """Where a draft by this name lives, refusing anything that is not a name.

    ``name`` reaches this from a URL path segment, so a traversal here would let
    a request read or delete a file anywhere the process can reach.
    """
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name or ""):
        raise SkillError(f"{name!r} is not a skill name")
    return loader.drafts_dir(directory) / f"{name}.md"


def save_draft(drafted: Drafted, directory: Optional[Path] = None) -> Path:
    skill, text = drafted.skill, drafted.text
    path = _draft_path(skill.name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    log.info("skills.draft_saved", name=skill.name, path=str(path))
    return path


def list_drafts(directory: Optional[Path] = None) -> list[str]:
    folder = loader.drafts_dir(directory)
    if not folder.is_dir():
        return []
    return sorted(path.stem for path in folder.glob("*.md"))


def read_draft(name: str, directory: Optional[Path] = None) -> str:
    path = _draft_path(name, directory)
    if not path.is_file():
        raise SkillError(f"no draft named {name!r}")
    return path.read_text(encoding="utf-8")


def approve(name: str, directory: Optional[Path] = None, *, text: Optional[str] = None) -> Path:
    """Install a draft as a real skill.

    ``text`` is the reviewed content when a person edited it before approving,
    which is the common case — the draft on disk is a starting point. It is
    validated again either way: what is approved is what is installed, and an
    edit that broke the file must fail here rather than at the next request.
    """
    body = text if text is not None else read_draft(name, directory)
    skill = loader.parse(body, source=f"<approving {name}>")

    root = directory or loader.skills_dir()
    destination = root / skill.name / loader.SKILL_FILE
    if destination.exists():
        raise SkillError(
            f"{skill.name!r} already exists at {destination}. Edit that file, or "
            "rename the draft."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(body, encoding="utf-8")

    draft_path = _draft_path(name, directory)
    if draft_path.is_file():
        draft_path.unlink()

    from pipeline.skills import reset

    reset()
    log.info("skills.approved", name=skill.name, path=str(destination))
    metrics.incr("skills.synth.approved")
    return destination


def discard(name: str, directory: Optional[Path] = None) -> None:
    path = _draft_path(name, directory)
    if not path.is_file():
        raise SkillError(f"no draft named {name!r}")
    path.unlink()
    log.info("skills.draft_discarded", name=name)


__all__ = [
    "Drafted",
    "approve",
    "discard",
    "draft",
    "list_drafts",
    "read_draft",
    "render",
    "save_draft",
]
