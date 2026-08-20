"""Filter compilation, and what happens to a filter a model invented."""

from __future__ import annotations

from pipeline.retrieve.filters import MetadataFilter


def test_no_constraints_compiles_to_nothing():
    assert MetadataFilter().compile() is None
    assert MetadataFilter().is_empty()


def test_single_value_uses_equality():
    assert MetadataFilter(department=["finance"]).compile() == "department = 'finance'"


def test_several_values_use_in():
    compiled = MetadataFilter(department=["finance", "legal"]).compile()
    assert compiled == "department IN ('finance', 'legal')"


def test_clauses_are_anded():
    compiled = MetadataFilter(department=["finance"], language=["en"]).compile()
    assert compiled == "department = 'finance' AND language = 'en'"


def test_date_range_becomes_comparisons():
    compiled = MetadataFilter(date_from="2026-01-01", date_to="2026-03-31").compile()
    assert compiled == "date >= '2026-01-01' AND date <= '2026-03-31'"


def test_a_written_out_date_is_normalised():
    assert MetadataFilter(date_from="March 3, 2026").compile() == "date >= '2026-03-03'"


def test_an_unparsable_date_is_dropped_not_passed_through():
    assert MetadataFilter(date_from="whenever").compile() is None


def test_a_quote_in_a_value_cannot_break_out():
    """The escape is doubling, and the value stays one literal."""
    compiled = MetadataFilter(author=["O'Brien"]).compile()
    assert compiled == "author = 'O''Brien'"


def test_an_injection_attempt_stays_inside_the_literal():
    compiled = MetadataFilter(department=["x' OR '1'='1"]).compile()
    assert compiled == "department = 'x'' OR ''1''=''1'"


def test_empty_strings_are_ignored():
    assert MetadataFilter(department=["", "  "]).compile() is None


def test_a_field_the_model_invented_is_dropped():
    """A hallucinated key must not reach a predicate."""
    built = MetadataFilter.from_model(
        {"department": "finance", "sensitivity": "high", "team": ["x"]}
    )
    assert built.department == ["finance"]
    assert built.compile() == "department = 'finance'"
    assert not hasattr(built, "sensitivity")


def test_from_model_accepts_a_bare_string_or_a_list():
    assert MetadataFilter.from_model({"language": "en"}).language == ["en"]
    assert MetadataFilter.from_model({"language": ["en", "fr"]}).language == ["en", "fr"]


def test_from_model_survives_junk():
    assert MetadataFilter.from_model("not a dict").is_empty()
    assert MetadataFilter.from_model({"department": None, "author": []}).is_empty()
