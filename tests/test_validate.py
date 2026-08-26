"""Direction validation: catching the edges a model wrote backwards."""

from __future__ import annotations

import pytest

from pipeline.graph.schema import Relationship
from pipeline.graph.validate import (
    Verdict,
    bucket,
    check_order,
    check_types,
    rule_for,
    validate,
)


def _rel(source: str, target: str, relation: str) -> Relationship:
    return Relationship(source=source, target=target, relation=relation, description="")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Person", "person"), ("PERSON", "person"), ("Executive", "person"),
        ("Organization", "org"), ("Company", "org"), ("Corporation", "org"),
        ("Location", "place"), ("City", "place"),
        ("Project", "project"), ("Initiative", "project"),
        ("Product", "thing"), ("Technology", "thing"),
        ("", "unknown"), ("Sasquatch", "unknown"),
    ],
)
def test_types_bucket(raw, expected):
    assert bucket(raw) == expected


def test_a_type_impossible_edge_is_flipped():
    """(Project Titan)-[STARTED]->(Apple): a project cannot start a company."""
    fixed, corrections = validate(
        [_rel("Project Titan", "Apple", "STARTED")],
        types={"Project Titan": "Project", "Apple": "Organization"},
    )
    assert (fixed[0].source, fixed[0].target) == ("Apple", "Project Titan")
    assert corrections[0].evidence == "entity types"


def test_word_order_catches_what_types_cannot():
    """Two organisations: ACQUIRED is type-consistent either way round."""
    fixed, corrections = validate(
        [_rel("Fabrikam Ltd", "Northwind Traders", "ACQUIRED")],
        types={"Fabrikam Ltd": "Organization", "Northwind Traders": "Organization"},
        text="Northwind Traders acquired Fabrikam Ltd in 2025.",
    )
    assert (fixed[0].source, fixed[0].target) == ("Northwind Traders", "Fabrikam Ltd")
    assert corrections[0].evidence == "word order in the source text"


def test_passive_voice_is_not_mistaken_for_a_reversal():
    """"B was acquired by A" puts the agent last, and it is still the source."""
    fixed, corrections = validate(
        [_rel("Northwind Traders", "Fabrikam Ltd", "ACQUIRED")],
        types={"Fabrikam Ltd": "Organization", "Northwind Traders": "Organization"},
        text="Fabrikam Ltd was acquired by Northwind Traders in 2025.",
    )
    assert corrections == []
    assert (fixed[0].source, fixed[0].target) == ("Northwind Traders", "Fabrikam Ltd")


def test_a_correct_edge_is_left_alone():
    fixed, corrections = validate(
        [_rel("Doug Field", "Tesla", "WORKED_AT")],
        types={"Doug Field": "Person", "Tesla": "Organization"},
        text="Doug Field worked at Tesla as Senior VP of Engineering.",
    )
    assert corrections == []
    assert (fixed[0].source, fixed[0].target) == ("Doug Field", "Tesla")


def test_a_symmetric_relation_is_never_flipped_on_types():
    """Person REPORTS_TO person fits both ways, so types decide nothing."""
    assert check_types(
        _rel("Priya Raman", "Alan Doyle", "REPORTS_TO"),
        {"Priya Raman": "Person", "Alan Doyle": "Person"},
    ) is Verdict.UNKNOWN


def test_an_unknown_verb_is_left_alone():
    fixed, corrections = validate(
        [_rel("Apple", "Tesla", "RELATED_TO")],
        types={"Apple": "Organization", "Tesla": "Organization"},
        text="Apple and Tesla are companies.",
    )
    assert corrections == []


def test_an_unknown_type_decides_nothing():
    assert check_types(
        _rel("A", "B", "ACQUIRED"), {"A": "Sasquatch", "B": "Organization"}
    ) is Verdict.UNKNOWN


def test_word_order_needs_the_verb_between_the_names():
    """Without the verb between them, order says nothing about direction."""
    assert check_order(
        _rel("Apple", "Tesla", "ACQUIRED"), "Apple and Tesla. Someone acquired something."
    ) is Verdict.UNKNOWN


def test_word_order_needs_both_names_present():
    assert check_order(
        _rel("Apple", "Missing Co", "ACQUIRED"), "Apple acquired something in 2025."
    ) is Verdict.UNKNOWN


def test_no_text_means_no_order_evidence():
    assert check_order(_rel("A", "B", "ACQUIRED"), "") is Verdict.UNKNOWN


def test_a_rule_is_found_by_substring():
    assert rule_for("ACQUIRED") is not None
    assert rule_for("WAS_ACQUIRED_BY") is not None
    assert rule_for("SOMETHING_ELSE") is None


def test_corrections_describe_what_changed():
    _, corrections = validate(
        [_rel("Project Titan", "Apple", "STARTED")],
        types={"Project Titan": "Project", "Apple": "Organization"},
    )
    described = corrections[0].describe()
    assert "(Project Titan)-[STARTED]->(Apple)" in described
    assert "(Apple)-[STARTED]->(Project Titan)" in described
    assert "entity types" in described


def test_an_empty_list_is_handled():
    assert validate([]) == ([], [])


def test_the_extractor_applies_validation(fake_backend, monkeypatch):
    """End to end: a reversed edge from the model is stored the right way."""
    from config import config
    from pipeline.graph.cache import GraphCache
    from pipeline.graph.extractor import GraphExtractor

    monkeypatch.setattr(config, "GRAPH_ENTITY_BACKEND", "llm")
    backend = fake_backend(
        {
            "entities": [
                {"name": "Project Titan", "type": "Project", "description": "a car project"},
                {"name": "Apple", "type": "Organization", "description": "a company"},
            ],
            "relationships": [
                {
                    "source": "Project Titan",
                    "target": "Apple",
                    "relation": "STARTED",
                    "description": "",
                }
            ],
        }
    )
    result = GraphExtractor(backend=backend, cache=GraphCache()).extract(
        "Project Titan is an initiative started by Apple."
    )
    assert (result.relationships[0].source, result.relationships[0].target) == (
        "Apple",
        "Project Titan",
    )


def test_validation_can_be_turned_off(fake_backend, monkeypatch):
    from config import config
    from pipeline.graph.cache import GraphCache
    from pipeline.graph.extractor import GraphExtractor

    monkeypatch.setattr(config, "GRAPH_ENTITY_BACKEND", "llm")
    monkeypatch.setattr(config, "GRAPH_VALIDATE_DIRECTION", False)
    backend = fake_backend(
        {
            "entities": [
                {"name": "Project Titan", "type": "Project", "description": ""},
                {"name": "Apple", "type": "Organization", "description": ""},
            ],
            "relationships": [
                {"source": "Project Titan", "target": "Apple", "relation": "STARTED",
                 "description": ""}
            ],
        }
    )
    result = GraphExtractor(backend=backend, cache=GraphCache()).extract("text")
    assert result.relationships[0].source == "Project Titan"
