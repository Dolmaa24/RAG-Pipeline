"""Routing an intent to a domain.

Two behaviours, tested separately because they fail differently. Trigger
matching is exact and deterministic and its bugs are about word boundaries.
The embedding fallback is a threshold, and its bugs are about calibration --
so those tests drive a stub embedder whose scores the test chooses, rather
than asserting on what BGE happens to produce for a phrase. A test that
asserts a model's similarity score is a test that fails when the model is
swapped, which says nothing about this module.

The calibration itself was measured, not guessed. Over five domain intents
carrying no trigger word and six deliberately generic questions, the raw
similarities overlapped completely (0.480-0.672 against 0.508-0.620) while the
margins over the runner-up did not (0.022-0.132 against 0.009-0.062). That is
why the margin is the test and the absolute score is only a floor.
"""

from __future__ import annotations

import pytest

# By symbol from the submodule: ``from pipeline.skills import match`` gets the
# function the package re-exports, not this module — the same trap conftest.py
# documents about ``understand``.
from pipeline.skills.loader import parse
from pipeline.skills.match import match, reset as reset_vectors
from pipeline.skills.models import SkillError

TEMPLATE = """\
---
name: {name}
description: {description}
triggers: [{triggers}]
tools: [search_corpus]
---

Body for {name}.
"""


def skill(name: str, triggers: str, description: str = "A test domain."):
    return parse(TEMPLATE.format(
        name=name, triggers=triggers, description=description
    ))


@pytest.fixture(autouse=True)
def _no_cached_vectors():
    reset_vectors()
    yield
    reset_vectors()


@pytest.fixture
def two():
    return {
        "insurance": skill("insurance", "insurance, policy, sum insured"),
        "school": skill("school", "school, syllabus, attendance"),
    }


class StubEmbedder:
    """Scores chosen by the test.

    Vectors are two-dimensional and hand-built so the cosine against the query
    is exactly the number the test asked for -- which is the only way to test a
    threshold without testing the model behind it.
    """

    name = "stub"
    model_name = "stub-embedder"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.queries: list[str] = []

    def embed_documents(self, texts):
        out = []
        for text in texts:
            name = next((n for n in self.scores if n in text), None)
            out.append(_unit(self.scores.get(name, 0.0)))
        return out

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0]


def _unit(cosine: float) -> list[float]:
    """A unit vector whose cosine against [1, 0] is ``cosine``."""
    return [cosine, (1.0 - cosine**2) ** 0.5]


# --- triggers --------------------------------------------------------------


def test_a_trigger_word_routes_the_intent(two):
    result = match("does the policy cover this", skills=two)
    assert result.skill.name == "insurance"
    assert result.how == "trigger"


def test_a_plural_still_matches_its_singular_trigger(two):
    """"which policies cover physiotherapy" is the ordinary way to ask, and
    exact word matching misses it."""
    assert match("which policies cover physiotherapy", skills=two).how == "trigger"


def test_a_phrase_outweighs_a_single_word():
    """"sum insured" appearing is far stronger evidence than "school"."""
    skills = {
        "insurance": skill("insurance", "sum insured"),
        "school": skill("school", "school"),
    }
    result = match("what is the sum insured under the school plan", skills=skills)
    assert result.skill.name == "insurance"


def test_a_trigger_does_not_match_inside_a_longer_word():
    """Substring matching scored `ecommerce` on "the school ordered textbooks"
    through the trigger "order"."""
    skills = {"ecommerce": skill("ecommerce", "order"), "school": skill("school", "school")}
    assert match("the school reordered textbooks", skills=skills).skill.name == "school"


def test_more_triggers_beat_fewer(two):
    result = match("the school syllabus and attendance policy", skills=two)
    assert result.skill.name == "school"


def test_runners_up_are_reported(two):
    result = match("the school syllabus and attendance policy", skills=two)
    assert result.runners_up and result.runners_up[0][0] == "insurance"


# --- the embedding fallback ------------------------------------------------


def test_no_trigger_falls_through_to_embeddings(two):
    stub = StubEmbedder({"insurance": 0.8, "school": 0.4})
    result = match("what am I covered for", skills=two, embedder=stub)
    assert result.how == "embedding"
    assert result.skill.name == "insurance"
    assert stub.queries == ["what am I covered for"]


def test_a_tie_on_triggers_falls_through_to_embeddings(two):
    """The intent named both domains equally. Counting cannot break that, and
    picking the alphabetically-first of the two would be arbitrary."""
    stub = StubEmbedder({"insurance": 0.4, "school": 0.9})
    result = match("the school policy", skills=two, embedder=stub)
    assert result.how == "embedding"
    assert result.skill.name == "school"


def test_too_close_to_the_runner_up_matches_nothing(two):
    """The generic case: every skill describes a domain of documents, so any
    question about documents scores middlingly against all of them."""
    stub = StubEmbedder({"insurance": 0.60, "school": 0.58})
    result = match("who wrote this document", skills=two, embedder=stub)
    assert result.skill is None
    assert result.how == "none"


def test_a_clear_lead_matches_even_at_a_middling_score(two):
    """A domain intent pulls away from the rest; the absolute score is only a
    floor, because it separates nothing on its own."""
    stub = StubEmbedder({"insurance": 0.50, "school": 0.30})
    assert match("what am I covered for", skills=two, embedder=stub).skill.name == "insurance"


def test_a_score_below_the_floor_matches_nothing(two):
    stub = StubEmbedder({"insurance": 0.20, "school": 0.02})
    assert match("qwerty asdf", skills=two, embedder=stub).skill is None


def test_an_unavailable_embedder_matches_nothing_rather_than_failing(two):
    """A task must not fail because the embedder could not load. The generic
    specialist is a worse answer than the right skill and a far better one than
    an error -- the position the shared rate limiter takes about Redis."""

    class Broken:
        name = model_name = "broken"

        def embed_documents(self, texts):
            raise RuntimeError("no model")

        def embed_query(self, text):
            raise RuntimeError("no model")

    result = match("what am I covered for", skills=two, embedder=Broken())
    assert result.skill is None


# --- naming one outright ---------------------------------------------------


def test_naming_a_skill_skips_the_routing(two):
    result = match("who wrote this document", skill="school", skills=two)
    assert result.skill.name == "school"
    assert result.how == "explicit"
    assert result.confidence == 1.0


def test_naming_an_unknown_skill_is_an_error(two):
    """A caller who asked for one skill must not be quietly given another."""
    with pytest.raises(SkillError, match="available: insurance, school"):
        match("anything", skill="finance", skills=two)


# --- edges -----------------------------------------------------------------


def test_an_empty_intent_matches_nothing(two):
    assert match("   ", skills=two).skill is None


def test_no_skills_installed_matches_nothing():
    assert match("which policies cover physiotherapy", skills={}).skill is None


def test_an_unmatched_result_explains_the_fallback(two):
    assert "generic" in match("", skills=two).explain()


def test_the_dict_form_carries_the_reason(two):
    payload = match("does the policy cover this", skills=two).to_dict()
    assert payload["skill"] == "insurance"
    assert payload["how"] == "trigger"
    assert "trigger" in payload["why"]


# --- against the four that ship --------------------------------------------


@pytest.mark.parametrize(
    "intent, expected",
    [
        ("which policies cover physiotherapy", "insurance"),
        ("what are the side effects of this medicine", "health"),
        ("compare the price and rating of these two laptops", "ecommerce"),
        ("what is the syllabus for the second semester", "school"),
    ],
)
def test_a_plainly_worded_intent_reaches_its_own_skill(intent, expected):
    """No embedder: each of these carries a trigger word, which is the point --
    the fallback is a backstop, not the mechanism."""
    result = match(intent)
    assert result.skill is not None and result.skill.name == expected
    assert result.how == "trigger"


@pytest.mark.slow
@pytest.mark.parametrize(
    "intent",
    [
        "which acquisitions does the corpus describe",
        "what did the report say about revenue",
        "who wrote this document",
    ],
)
def test_a_generic_question_reaches_no_skill(intent):
    """Marked slow: no trigger fires, so this is the one place the real
    embedder has to run. It is also the only test that checks the shipped
    calibration against the real model rather than a stub. Measured: all three
    scored 0.508-0.586 against their nearest skill, inside the range genuine
    domain intents occupy. The margin is what refuses them."""
    assert match(intent).skill is None
