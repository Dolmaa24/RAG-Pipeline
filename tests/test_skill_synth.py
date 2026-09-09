"""Drafting a skill for a domain nothing covers, and the gate before it is live.

The gate is the point of this module, so it is what most of these test. A
skill's body becomes an agent's system prompt; a drafted skill that installed
itself would be a model writing its own instructions, and there is no reviewing
that afterwards because by then it has run.

The model is a stub throughout. What is being tested is the rendering, the
validation and the approval path — not whether a particular model writes a good
café description, which is not a property a test can hold.
"""

from __future__ import annotations

import json

import pytest

from pipeline.agents.tools import Effect
from pipeline.extract.llm.base import ToolRequest, ToolTurn
from pipeline.skills import loader, synth
from pipeline.skills.models import SkillError

PAYLOAD = {
    "name": "barista",
    "description": "Coffee shop operations: orders, drinks, the machine, the queue.",
    "triggers": ["barista", "espresso", "latte", "cafe", "order queue"],
    "tools": ["corpus_profile", "search_corpus", "answer_from_corpus"],
    "system": "You answer questions about cafe operations from an indexed corpus.",
    "schema": {"drink": "string", "price": "number"},
    "entity_types": ["Drink", "Order"],
    "relation_types": ["ORDERED_BY"],
    "agents": [
        {"name": "receptionist", "purpose": "Takes orders.", "writes": ["orders.py"]},
        {"name": "machine", "purpose": "Brews drinks.", "writes": ["machine.py"]},
    ],
}


class Stub:
    """Returns whatever payload the test hands it, as a tool call.

    A tool call rather than a JSON body, because that is the door synthesis
    goes through: ``complete_json`` is an extraction interface whose prompt
    forbids the model from using its own knowledge, and a real model asked to
    describe a café through it correctly returns nulls for every field.
    """

    name, model = "stub", "stub-1"

    def __init__(self, payload=None, *, as_text: bool = False):
        self.payload = PAYLOAD if payload is None else payload
        self.prompt = ""
        self.tool_choice = ""
        self.as_text = as_text

    def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
        self.prompt = "\n\n".join(m.content for m in messages)
        self.tool_choice = tool_choice
        if self.as_text:
            return ToolTurn(text=json.dumps(self.payload), backend=self.name, model=self.model)
        return ToolTurn(
            calls=[ToolRequest(name="propose_skill", arguments=self.payload, call_id="1")],
            backend=self.name,
            model=self.model,
        )


@pytest.fixture
def skills_root(tmp_path, monkeypatch):
    """A skills folder of its own, so a test never installs into the repo's."""
    monkeypatch.setattr(loader, "skills_dir", lambda: tmp_path)
    return tmp_path


# --- rendering -------------------------------------------------------------


def test_a_payload_renders_to_a_file_that_parses():
    """The whole reason the model is asked for JSON and not for markdown."""
    text = synth.render(PAYLOAD, intent="make a Baristo system")
    skill = loader.parse(text)
    assert skill.name == "barista"
    assert skill.tools == ("corpus_profile", "search_corpus", "answer_from_corpus")
    assert skill.schema_hint == {"drink": "string", "price": "number"}


def test_the_roster_survives_the_round_trip():
    skill = loader.parse(synth.render(PAYLOAD))
    assert [agent.name for agent in skill.agents] == ["receptionist", "machine"]
    assert skill.agents[0].writes == ("orders.py",)
    assert skill.buildable


def test_the_intent_is_recorded_in_the_frontmatter():
    """A reviewer three weeks later needs to know what was asked for."""
    text = synth.render(PAYLOAD, intent="make a Baristo system")
    header = text.split("---")[1]
    assert "make a Baristo system" in header


def test_the_body_is_only_the_system_prompt():
    """It is sent to the model verbatim on every run, so a note to a human
    reviewer placed there becomes part of the agent's instructions."""
    skill = loader.parse(synth.render(PAYLOAD, intent="make a Baristo system"))
    assert skill.system == PAYLOAD["system"]
    assert "Baristo" not in skill.system
    assert "<!--" not in skill.system


def test_a_name_is_made_url_safe():
    text = synth.render({**PAYLOAD, "name": "Coffee Shop"})
    assert loader.parse(text).name == "coffee-shop"


def test_an_agent_name_is_made_url_safe():
    payload = {**PAYLOAD, "agents": [{"name": "Machine Worker", "purpose": "Brews.", "writes": []}]}
    assert loader.parse(synth.render(payload)).agents[0].name == "machine-worker"


def test_duplicate_triggers_are_dropped():
    payload = {**PAYLOAD, "triggers": ["cafe", "Cafe", "CAFE", "latte"]}
    assert loader.parse(synth.render(payload)).triggers == ("cafe", "latte")


def test_a_folded_description_becomes_one_line():
    payload = {**PAYLOAD, "description": "Coffee shop\noperations\nand drinks."}
    assert "\n" not in loader.parse(synth.render(payload)).description


def test_an_agent_without_a_purpose_is_dropped():
    """The purpose is the whole of that worker's brief. A blank one would send
    a model to write a file with nothing to go on."""
    payload = {**PAYLOAD, "agents": [
        {"name": "receptionist", "purpose": "Takes orders.", "writes": ["a.py"]},
        {"name": "ghost", "purpose": "", "writes": ["b.py"]},
    ]}
    assert [a.name for a in loader.parse(synth.render(payload)).agents] == ["receptionist"]


# --- what a draft is never allowed to be -----------------------------------


def test_a_drafted_skill_is_always_read_only():
    """The security test. A model may not author a skill that reaches the
    network or writes to the corpus, however it fills the field."""
    text = synth.render({**PAYLOAD, "requires": "write", "tools": ["extract_url"]})
    assert loader.parse(text).requires is Effect.READ
    assert "requires: read" in text


def test_an_invented_tool_fails_at_generation():
    """Not at approval, and not silently at the first run."""
    stub = Stub({**PAYLOAD, "tools": ["search_corpuss", "brew_coffee"]})
    with pytest.raises(SkillError, match="brew_coffee"):
        synth.draft("make a Baristo system", backend=stub)


def test_an_empty_body_fails_at_generation():
    with pytest.raises(SkillError, match="body is empty"):
        synth.draft("anything", backend=Stub({**PAYLOAD, "system": ""}))


def test_a_reply_with_no_proposal_is_reported():
    class Silent:
        name = model = "silent"

        def complete_with_tools(self, *, messages, tools, tool_choice="auto"):
            return ToolTurn(text="I would rather not.", backend="silent", model="silent")

    with pytest.raises(SkillError, match="did not propose"):
        synth.draft("anything", backend=Silent())


def test_the_object_is_accepted_as_text_too():
    """A native tool-caller that answered with the object anyway. The shim
    already covers backends without tools; this covers one that has them."""
    assert synth.draft("anything", backend=Stub(as_text=True)).skill.name == "barista"


def test_the_model_is_not_asked_whether_to_propose():
    """There is one tool and calling it is the whole task."""
    stub = Stub()
    synth.draft("anything", backend=stub)
    assert stub.tool_choice == "required"


def test_an_empty_intent_is_refused():
    with pytest.raises(SkillError, match="intent is required"):
        synth.draft("   ", backend=Stub())


def test_the_prompt_names_only_the_read_tools():
    """A drafted skill choosing write_source would be a skill that writes code
    on a retrieval run. The model is never shown that it exists."""
    stub = Stub()
    synth.draft("make a Baristo system", backend=stub)
    assert "search_corpus" in stub.prompt
    assert "write_source" not in stub.prompt
    assert "crawl_site" not in stub.prompt


# --- the drafts folder -----------------------------------------------------


def test_a_draft_is_saved_where_the_loader_does_not_look(skills_root):
    drafted = synth.draft("make a Baristo system", backend=Stub())
    path = synth.save_draft(drafted)

    assert path.is_file()
    assert synth.list_drafts() == ["barista"]
    # The gate: present on disk, absent from the live set.
    assert loader.load_all(skills_root, force=True) == {}


def test_an_underscore_folder_is_never_loaded_as_a_skill(skills_root):
    """Belt and braces. Drafts are bare .md files so they would be skipped
    anyway, and "skipped by accident" is not the property to rest this on."""
    hidden = skills_root / "_drafts" / "barista"
    hidden.mkdir(parents=True)
    (hidden / "SKILL.md").write_text(synth.render(PAYLOAD))
    assert loader.load_all(skills_root, force=True) == {}


def test_a_draft_reads_back(skills_root):
    synth.save_draft(synth.draft("anything", backend=Stub()))
    assert "name: barista" in synth.read_draft("barista")


def test_reading_a_missing_draft_says_so(skills_root):
    with pytest.raises(SkillError, match="no draft named"):
        synth.read_draft("nothing")


def test_a_draft_name_cannot_traverse(skills_root):
    """The name arrives from a URL path segment."""
    for name in ("../config", "/etc/passwd", "a/b", ""):
        with pytest.raises(SkillError, match="not a skill name"):
            synth.read_draft(name)


def test_discarding_removes_it(skills_root):
    synth.save_draft(synth.draft("anything", backend=Stub()))
    synth.discard("barista")
    assert synth.list_drafts() == []


# --- approval --------------------------------------------------------------


def test_approving_installs_the_skill(skills_root):
    synth.save_draft(synth.draft("make a Baristo system", backend=Stub()))
    path = synth.approve("barista", skills_root)

    assert path == skills_root / "barista" / "SKILL.md"
    assert "barista" in loader.load_all(skills_root, force=True)


def test_approving_removes_the_draft(skills_root):
    synth.save_draft(synth.draft("anything", backend=Stub()))
    synth.approve("barista", skills_root)
    assert synth.list_drafts() == []


def test_an_edited_draft_is_what_gets_installed(skills_root):
    """The common case: the draft is a starting point, not a finished file."""
    synth.save_draft(synth.draft("anything", backend=Stub()))
    edited = synth.read_draft("barista").replace(
        "Coffee shop operations", "Everything about coffee"
    )
    synth.approve("barista", skills_root, text=edited)
    assert "Everything about coffee" in loader.load_all(skills_root, force=True)["barista"].description


def test_an_edit_that_broke_the_file_is_refused(skills_root):
    """Validated on the way in, so a bad edit fails here rather than at the
    next request."""
    synth.save_draft(synth.draft("anything", backend=Stub()))
    with pytest.raises(SkillError):
        synth.approve("barista", skills_root, text="no frontmatter here")
    assert loader.load_all(skills_root, force=True) == {}


def test_approving_does_not_overwrite_an_existing_skill(skills_root):
    """A silently replaced prompt is the worst outcome available here."""
    existing = skills_root / "barista"
    existing.mkdir()
    (existing / "SKILL.md").write_text(synth.render({**PAYLOAD, "system": "The original."}))

    synth.save_draft(synth.draft("anything", backend=Stub()))
    with pytest.raises(SkillError, match="already exists"):
        synth.approve("barista", skills_root)
    assert "The original." in (existing / "SKILL.md").read_text()


def test_approving_a_missing_draft_says_so(skills_root):
    with pytest.raises(SkillError, match="no draft named"):
        synth.approve("nothing", skills_root)


def test_the_installed_skill_matches_the_intent_that_drafted_it(skills_root):
    """The end of the loop: an intent that matched nothing now matches."""
    from pipeline.skills.match import match

    synth.save_draft(synth.draft("make a Baristo system", backend=Stub()))
    synth.approve("barista", skills_root)

    installed = loader.load_all(skills_root, force=True)
    assert match("the espresso order queue", skills=installed).skill.name == "barista"
