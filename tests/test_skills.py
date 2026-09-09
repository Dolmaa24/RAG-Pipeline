"""Loading a domain pack, and what it is and is not allowed to do.

The permission tests here are the important ones. A skill is a file someone
drops in a folder, and the pipeline's standing rule is that effects are granted
by the request and never by anything the run itself supplies -- ``budget.py``
calls its counters "both a ceiling and a grant". A skill must therefore be able
to narrow what an agent can reach and never to widen it, and that has to be
asserted rather than assumed, because the failure is silent: a skill that
quietly kept ``crawl_site`` on a read-only run would look exactly like a skill
that did not.
"""

from __future__ import annotations

import textwrap

import pytest

from pipeline.agents.budget import Budget
from pipeline.agents.roles import Role
from pipeline.agents.tools import Effect
from pipeline.skills import loader
from pipeline.skills.models import Skill, SkillError

VALID = """\
---
name: demo
description: A domain used only by the tests.
triggers: [demo, worked example]
tools: [search_corpus, answer_from_corpus]
requires: read
extraction:
  schema:
    title: string
    tags: list of strings
  entity_types: [Widget, Gadget]
  relation_types: [PART_OF]
---

You are the demonstration specialist. Answer from the tools.
"""


def written(tmp_path, text: str, name: str = "demo"):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "SKILL.md").write_text(text, encoding="utf-8")
    return folder / "SKILL.md"


def test_a_valid_skill_parses():
    skill = loader.parse(VALID)
    assert skill.name == "demo"
    assert skill.tools == ("search_corpus", "answer_from_corpus")
    assert skill.triggers == ("demo", "worked example")


def test_the_body_becomes_the_system_prompt():
    assert loader.parse(VALID).system.startswith("You are the demonstration")


def test_the_frontmatter_does_not_leak_into_the_prompt():
    """The prompt is what a model reads. A YAML block in it is noise it will
    try to interpret."""
    assert "---" not in loader.parse(VALID).system
    assert "triggers:" not in loader.parse(VALID).system


def test_the_extraction_schema_is_read():
    skill = loader.parse(VALID)
    assert skill.schema_hint == {"title": "string", "tags": "list of strings"}
    assert skill.entity_types == ("Widget", "Gadget")
    assert skill.relation_types == ("PART_OF",)


def test_a_description_is_collapsed_onto_one_line():
    """YAML folded scalars keep their newlines, and the description is embedded
    and shown in a list. A three-line description is a formatting bug in both."""
    skill = loader.parse(VALID.replace(
        "description: A domain used only by the tests.",
        "description: >\n  A domain used\n  only by the tests.",
    ))
    assert "\n" not in skill.description


def test_a_file_with_no_frontmatter_is_refused():
    with pytest.raises(SkillError, match="frontmatter"):
        loader.parse("Just a prompt, no settings.")


def test_broken_yaml_is_refused_with_the_source_named():
    broken = "---\nname: demo\n  bad: [indent\n---\n\nBody.\n"
    with pytest.raises(SkillError, match="skills/demo"):
        loader.parse(broken, source="skills/demo/SKILL.md")


def test_a_missing_name_is_refused():
    with pytest.raises(SkillError, match="'name' is required"):
        loader.parse(VALID.replace("name: demo", "name: ''"))


def test_a_missing_description_is_refused():
    """The description is what an intent is compared against when no trigger
    fires. Without one the skill can only ever be reached by name."""
    with pytest.raises(SkillError, match="'description' is required"):
        loader.parse(VALID.replace("description: A domain used only by the tests.", "description: ''"))


def test_an_empty_body_is_refused():
    with pytest.raises(SkillError, match="body is empty"):
        loader.parse(VALID.split("---\n\n")[0] + "---\n\n   \n")


def test_a_name_that_is_not_url_safe_is_refused():
    """The name is an API argument and a path segment."""
    with pytest.raises(SkillError, match="must be lowercase"):
        loader.parse(VALID.replace("name: demo", "name: My Demo"))


def test_an_unknown_tool_is_refused_at_load():
    """A typo must be a startup error, not a capability that never appears."""
    with pytest.raises(SkillError, match="no tool named 'search_corpuss'"):
        loader.parse(VALID.replace("search_corpus,", "search_corpuss,"))


def test_naming_no_tools_is_refused():
    with pytest.raises(SkillError, match="at least one tool"):
        loader.parse(VALID.replace("tools: [search_corpus, answer_from_corpus]", "tools: []"))


def test_an_unknown_effect_is_refused():
    with pytest.raises(SkillError, match="requires must be one of"):
        loader.parse(VALID.replace("requires: read", "requires: everything"))


def test_a_skill_may_name_a_write_tool():
    """Legitimate, and the run's budget is what decides whether it appears."""
    skill = loader.parse(
        VALID.replace("tools: [search_corpus, answer_from_corpus]", "tools: [extract_url]")
        .replace("requires: read", "requires: write")
    )
    assert skill.requires is Effect.WRITE


# --- permission: a skill narrows, and never widens -------------------------


def _writing_skill() -> Skill:
    return loader.parse(
        VALID.replace(
            "tools: [search_corpus, answer_from_corpus]",
            "tools: [search_corpus, extract_url, crawl_site]",
        )
    )


def test_a_read_only_budget_strips_the_write_tools_a_skill_named():
    """The security test. A file in a folder does not grant an effect.

    Written against ``Role.catalog``, which is what the supervisor actually
    calls -- asserting on ``skill.tools`` would pass while the agent was handed
    crawl_site anyway.
    """
    role = _writing_skill().as_role()
    offered = {tool["name"] for tool in role.catalog(Budget().effects())}
    assert offered == {"search_corpus"}
    assert "crawl_site" not in offered
    assert "extract_url" not in offered


def test_the_same_skill_keeps_them_when_the_request_budgeted_for_them():
    """The other half: narrowing must not be unconditional, or the grant means
    nothing either."""
    budget = Budget(network_calls=2, write_calls=2)
    offered = {tool["name"] for tool in _writing_skill().as_role().catalog(budget.effects())}
    assert {"search_corpus", "extract_url", "crawl_site"} <= offered


def test_a_write_skill_is_unavailable_under_a_read_only_budget():
    from pipeline.agents.roles import available

    skill = loader.parse(
        VALID.replace("tools: [search_corpus, answer_from_corpus]", "tools: [extract_url]")
        .replace("requires: read", "requires: write")
    )
    assert skill.as_role().requires is Effect.WRITE
    assert skill.as_role() not in available(Budget())


def test_as_role_returns_an_ordinary_role():
    """No new abstraction: the loop, the budget and the registry are unchanged."""
    role = loader.parse(VALID).as_role()
    assert isinstance(role, Role)
    assert role.name == "demo"
    assert role.system == loader.parse(VALID).system


# --- graph guidance --------------------------------------------------------


def test_graph_guidance_names_the_domains_types():
    guidance = loader.parse(VALID).graph_guidance()
    assert "Widget" in guidance and "PART_OF" in guidance


def test_a_skill_with_no_types_adds_no_guidance():
    """So graph extraction is exactly what it was before skills existed."""
    without = VALID.split("  entity_types")[0] + "---\n\nBody.\n"
    assert loader.parse(without).graph_guidance() == ""


# --- the folder ------------------------------------------------------------


def test_load_all_reads_every_folder(tmp_path):
    written(tmp_path, VALID, "demo")
    written(tmp_path, VALID.replace("name: demo", "name: other"), "other")
    assert sorted(loader.load_all(tmp_path, force=True)) == ["demo", "other"]


def test_a_broken_skill_does_not_hide_the_others(tmp_path):
    """A typo in an experimental skill must not take the rest offline."""
    written(tmp_path, VALID, "demo")
    written(tmp_path, "not a skill at all", "broken")
    assert sorted(loader.load_all(tmp_path, force=True)) == ["demo"]


def test_a_folder_without_a_skill_file_is_ignored(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "README.md").write_text("just notes")
    written(tmp_path, VALID, "demo")
    assert sorted(loader.load_all(tmp_path, force=True)) == ["demo"]


def test_a_duplicate_name_keeps_the_first_and_warns(tmp_path):
    written(tmp_path, VALID, "a-folder")
    written(tmp_path, VALID, "b-folder")
    loaded = loader.load_all(tmp_path, force=True)
    assert list(loaded) == ["demo"]
    assert loaded["demo"].path.parent.name == "a-folder"


def test_a_missing_directory_is_empty_not_an_error(tmp_path):
    assert loader.load_all(tmp_path / "nothing-here", force=True) == {}


def test_the_switch_turns_skills_off(tmp_path, monkeypatch):
    from config import config

    written(tmp_path, VALID, "demo")
    monkeypatch.setattr(config, "SKILLS_ENABLED", False)
    assert loader.load_all(tmp_path, force=True) == {}


def test_an_edited_skill_is_reloaded(tmp_path):
    """uvicorn --reload sees a new file; a cache keyed on nothing would not."""
    path = written(tmp_path, VALID, "demo")
    assert loader.load_all(tmp_path)["demo"].description.startswith("A domain")
    path.write_text(VALID.replace("A domain used only by the tests.", "Rewritten."))
    assert loader.load_all(tmp_path)["demo"].description == "Rewritten."


def test_get_names_what_is_available_when_asked_for_something_missing(tmp_path):
    written(tmp_path, VALID, "demo")
    with pytest.raises(SkillError, match="available: demo"):
        loader.get("nope", tmp_path)


# --- the four that ship ----------------------------------------------------


def test_the_shipped_skills_all_load():
    """They are read by the API on every request; a broken one is a 500."""
    shipped = loader.load_all(force=True)
    assert set(shipped) == {"health", "ecommerce", "insurance", "school"}


def test_no_shipped_skill_asks_for_more_than_reading():
    """A domain pack answers questions. Fetching and writing are decisions the
    request makes, and a skill that shipped with `requires: write` would be
    granting itself one by being installed."""
    for skill in loader.load_all(force=True).values():
        assert skill.requires is Effect.READ, skill.name


def test_every_shipped_skill_has_an_extraction_schema():
    for skill in loader.load_all(force=True).values():
        assert skill.schema_hint, skill.name
        assert skill.entity_types, skill.name
