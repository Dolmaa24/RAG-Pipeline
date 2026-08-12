"""Normalisation: text, dates, and money."""

from __future__ import annotations

from decimal import Decimal

import pytest

from pipeline.normalize.dates import is_ambiguous, normalize_date_field, parse_date
from pipeline.normalize.numbers import (
    detect_currency,
    normalize_money,
    parse_decimal,
    parse_int,
)
from pipeline.normalize.text import clean_text, normalize_record, strip_invisibles, summarize


class TestText:
    def test_collapses_whitespace(self):
        assert clean_text("  a   b\t\tc  ") == "a b c"

    def test_strips_zero_width_characters(self):
        """Category Cf survives NFKC and is not matched by \\s."""
        poisoned = "pri​ce﻿: 19­.99"
        assert strip_invisibles(poisoned) == "price: 19.99"

    def test_two_visually_identical_strings_become_equal(self):
        """The reason this matters: a dedupe key built from one of them misses."""
        assert clean_text("café") == clean_text("café")

    def test_newlines_are_preserved_as_structure(self):
        assert clean_text("a\n\nb") == "a\n\nb"

    def test_nfkc_normalisation(self):
        assert clean_text("ﬁle") == "file"

    def test_empty_becomes_none_in_a_record(self):
        assert normalize_record({"a": "   "})["a"] is None

    def test_summarize_cuts_at_a_sentence(self):
        text = "First sentence here. Second sentence follows. Third one too."
        assert summarize(text, max_chars=40).endswith("…")


class TestDates:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("2026-03-04", "2026-03-04"),
            ("2026-03-04T10:30:00Z", "2026-03-04"),
            ("March 4, 2026", "2026-03-04"),
            ("4 March 2026", "2026-03-04"),
            ("2026/03/04", "2026-03-04"),
            ("Wed, 04 Mar 2026 10:30:00 +0000", "2026-03-04"),
        ],
    )
    def test_formats(self, value, expected):
        assert parse_date(value).date().isoformat() == expected

    def test_ambiguous_dates_follow_the_stated_convention(self):
        assert parse_date("03/04/2026", dayfirst=False).month == 3
        assert parse_date("03/04/2026", dayfirst=True).month == 4

    def test_ambiguity_is_flagged_rather_than_hidden(self):
        assert is_ambiguous("03/04/2026")
        assert not is_ambiguous("25/04/2026")  # only one reading possible

    def test_relative_dates(self):
        from datetime import datetime, timezone

        parsed = parse_date("3 days ago")
        assert (datetime.now(timezone.utc) - parsed).days == 3

    def test_the_original_string_always_survives(self):
        result = normalize_date_field("March 4, 2026")
        assert result["raw"] == "March 4, 2026"
        assert result["date"] == "2026-03-04"

    def test_unparseable_input(self):
        assert parse_date("sometime next spring") is None
        assert parse_date("") is None

    def test_result_is_timezone_aware(self):
        assert parse_date("2026-03-04").tzinfo is not None


class TestNumbers:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("19.99", "19.99"),
            ("$19.99", "19.99"),
            ("1,234.56", "1234.56"),      # US grouping
            ("1.234,56", "1234.56"),      # European grouping
            ("1 234,56", "1234.56"),      # French grouping
            ("-42", "-42"),
            ("£51.77", "51.77"),
        ],
    )
    def test_separator_conventions(self, value, expected):
        assert parse_decimal(value) == Decimal(expected)

    def test_multipliers(self):
        assert parse_decimal("4.2M") == Decimal("4200000.0")
        assert parse_decimal("3 billion") == Decimal("3000000000")
        assert parse_decimal("5 crore") == Decimal("50000000")

    def test_money_is_a_string_not_a_float(self):
        """A price that round-trips as 19.989999999999998 is a bug in an invoice."""
        money = normalize_money("$19.99")
        assert money["amount"] == "19.99"
        assert isinstance(money["amount"], str)
        assert Decimal(money["amount"]) == Decimal("19.99")

    @pytest.mark.parametrize(
        "value,code",
        [("$19.99", "USD"), ("£51.77", "GBP"), ("€42,00", "EUR"), ("₹500", "INR"),
         ("19.99 USD", "USD"), ("C$25", "CAD")],
    )
    def test_currency_detection(self, value, code):
        assert detect_currency(value) == code

    def test_longest_symbol_wins(self):
        """C$ must not be read as $."""
        assert detect_currency("C$25.00") == "CAD"

    def test_unparseable(self):
        assert parse_decimal("call for pricing") is None
        assert parse_int("") is None


class TestTypedRecord:
    def test_price_gets_an_exact_companion_and_the_raw_stays(self):
        record = normalize_record({"price": "$1,234.56"})
        assert record["price"] == "$1,234.56"
        assert record["price_amount"] == "1234.56"
        assert record["price_currency"] == "USD"

    def test_date_gets_an_iso_companion(self):
        record = normalize_record({"published_date": "March 4, 2026"})
        assert record["published_date"] == "March 4, 2026"
        assert record["published_date_iso"].startswith("2026-03-04")

    def test_ambiguous_dates_are_marked(self):
        assert normalize_record({"date": "03/04/2026"})["date_ambiguous"] is True

    def test_relative_urls_are_made_absolute(self):
        record = normalize_record({"image_url": "/img/a.png"}, base_url="https://a.test/dir/page")
        assert record["image_url"] == "https://a.test/img/a.png"

    def test_counts_become_integers(self):
        assert normalize_record({"review_count": "1,284 reviews"})["review_count_value"] == 1284

    def test_ratings_become_floats(self):
        assert normalize_record({"rating": "4.6 out of 5"})["rating_value"] == 4.6

    def test_typed_pass_can_be_disabled(self):
        record = normalize_record({"price": "$19.99"}, typed=False)
        assert "price_amount" not in record

    def test_nested_values_are_cleaned(self):
        record = normalize_record({"author": {"name": "  Ada ​ Lovelace "}})
        assert record["author"]["name"] == "Ada Lovelace"
