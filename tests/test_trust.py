"""Validation, deduplication, drift, and provenance."""

from __future__ import annotations

import pytest

from config import config
from models import ExtractionItem, ExtractionMethod, ResourceKind, RunReport
from pipeline.trust.dedupe import Deduplicator, bands_of, hamming, simhash
from pipeline.trust.drift import DriftMonitor
from pipeline.trust.validation import validate_record


class TestValidation:
    @pytest.mark.parametrize("value", ["N/A", "n/a", "none", "unknown", "TBD", "-", "null"])
    def test_placeholders_are_not_values(self, value):
        result = validate_record({"price": value}, schema_hint={"price": "string"})
        assert not result.ok

    @pytest.mark.parametrize("value", ["string", "list of strings", "number", "your answer here"])
    def test_schema_echo_is_caught(self, value):
        """A small model copying the schema instead of filling it."""
        result = validate_record({"title": value}, schema_hint={"title": "string"})
        assert not result.ok

    @pytest.mark.parametrize(
        "value",
        [
            "I'm sorry, I cannot find that information in the text.",
            "As an AI language model, I don't have access to that.",
            "Unfortunately, the page does not contain a price.",
        ],
    )
    def test_model_refusals_are_caught(self, value):
        result = validate_record({"summary": value}, schema_hint={"summary": "string"})
        assert not result.ok

    def test_a_real_value_passes(self):
        result = validate_record(
            {"title": "Blue Widget", "price": "19.99"},
            schema_hint={"title": "string", "price": "string"},
        )
        assert result.ok

    def test_nulls_are_acceptable(self):
        result = validate_record(
            {"title": "Blue Widget", "price": None},
            schema_hint={"title": "string", "price": "string"},
        )
        assert result.ok

    def test_an_entirely_empty_record_fails(self):
        result = validate_record({"a": None, "b": ""}, schema_hint={"a": "string", "b": "string"})
        assert not result.ok
        assert any("nothing was extracted" in error for error in result.errors)

    def test_whole_page_dumped_into_one_field(self):
        result = validate_record({"summary": "x" * 25_000}, schema_hint={"summary": "string"})
        assert not result.ok

    def test_lorem_ipsum(self):
        result = validate_record(
            {"body": "Lorem ipsum dolor sit amet"}, schema_hint={"body": "string"}
        )
        assert not result.ok

    def test_required_fields(self):
        result = validate_record({"a": "x"}, schema_hint={"a": "string"}, required=["b"])
        assert not result.ok

    def test_url_shaped_fields_warn_rather_than_fail(self):
        result = validate_record({"image_url": "not-a-url"}, schema_hint={"image_url": "string"})
        assert result.ok  # a warning, not an error
        assert result.warnings

    def test_lists_are_checked_element_by_element(self):
        result = validate_record({"tags": ["real", "N/A"]}, schema_hint={"tags": "list"})
        assert not result.ok

    def test_prompt_echo_is_caught(self):
        prompt = "extract the title of the article and the name of its author"
        result = validate_record(
            {"title": "Extract the title of the article and the name of its author"},
            schema_hint={"title": "string"},
            prompt=prompt,
        )
        assert not result.ok


class TestSimhash:
    def test_identical_text_has_distance_zero(self):
        text = "The heron stood still for eleven minutes beside the slow river."
        assert hamming(simhash(text), simhash(text)) == 0

    #: A page-length body. Simhash distance scales with the *fraction* of
    #: shingles that changed, so the thresholds only mean anything at realistic
    #: document lengths — on two sentences, one added clause is a third of the
    #: document and lands far outside any sane threshold.
    ARTICLE = (
        "The heron stood still for eleven minutes beside the slow brown river, "
        "waiting for a fish that never came, while the light failed behind the alders. "
        "Nothing moved on the far bank except the reeds, which bent and straightened "
        "in a wind too small to feel. A cyclist passed on the towpath without looking up. "
        "By the time the bird moved, the water had gone the colour of pewter and the "
        "midges had come out over the shallows near the weir. It took two steps, folded "
        "itself upward, and was gone downstream before the ripples reached the bank. "
        "The fish, if there had ever been one, was never seen at all."
    )

    def test_a_small_edit_stays_close(self):
        with_ad = self.ARTICLE + " Advertisement. Subscribe to our newsletter today."
        assert hamming(simhash(self.ARTICLE), simhash(with_ad)) <= 6

    def test_distance_grows_with_the_share_of_changed_text(self):
        one_line = self.ARTICLE + " A single extra sentence at the end."
        many_lines = self.ARTICLE + " Different text. " * 30
        near = hamming(simhash(self.ARTICLE), simhash(one_line))
        far = hamming(simhash(self.ARTICLE), simhash(many_lines))
        assert near < far

    def test_different_documents_are_far_apart(self):
        a = "The heron stood still for eleven minutes beside the slow brown river."
        b = "Quarterly revenue rose fourteen percent on strong demand in the Asian market."
        assert hamming(simhash(a), simhash(b)) > 10

    def test_empty_text(self):
        assert simhash("") == 0

    def test_bands_are_position_tagged(self):
        """Band 0 being 0xABCD is a different fact from band 2 being 0xABCD."""
        bands = bands_of(0xABCD_ABCD_ABCD_ABCD)
        assert len(bands) == 4
        assert len(set(bands)) == 4


class TestDeduplicator:
    def test_exact_duplicate_by_content_hash(self):
        dedupe = Deduplicator()
        text = "a document with enough words in it to fingerprint meaningfully at all"
        dedupe.add(text, "hash1", "https://a.test/1")
        verdict = dedupe.check(text, "hash1", "https://a.test/2")
        assert verdict.is_duplicate and verdict.kind == "exact"

    def test_near_duplicate_is_found(self):
        dedupe = Deduplicator(max_distance=6)
        original = (
            "The heron stood still for eleven minutes beside the slow brown river, "
            "waiting for a fish that never came, while the light failed behind the alders. "
            "Nothing moved on the far bank except the reeds."
        )
        dedupe.add(original, "hash1", "https://a.test/story")
        syndicated = original + " Advertisement. Sign up for our newsletter."
        verdict = dedupe.check(syndicated, "hash2", "https://b.test/story")
        assert verdict.is_duplicate and verdict.kind == "near"

    def test_unrelated_documents_are_not_duplicates(self):
        dedupe = Deduplicator()
        dedupe.add("The heron stood still beside the slow brown river at dusk.", "h1", "https://a/1")
        verdict = dedupe.check(
            "Quarterly revenue rose fourteen percent on demand in Asia.", "h2", "https://a/2"
        )
        assert not verdict.is_duplicate

    def test_the_same_url_is_not_its_own_duplicate(self):
        dedupe = Deduplicator()
        text = "a document with enough words in it to fingerprint meaningfully"
        dedupe.add(text, "h1", "https://a.test/1")
        assert not dedupe.check(text, "h1", "https://a.test/1").is_duplicate

    def test_disabling_dedupe(self, monkeypatch):
        monkeypatch.setattr(config, "DEDUPE_ENABLED", False, raising=False)
        dedupe = Deduplicator()
        dedupe.add("text here", "h1", "https://a/1")
        assert not dedupe.check("text here", "h1", "https://a/2").is_duplicate


class TestDriftMonitor:
    def _fill(self, monitor, count, record):
        alerts = []
        for _ in range(count):
            alerts.extend(monitor.observe("https://a.test/x", "schema1", record))
        return alerts

    def test_no_alert_before_a_baseline_exists(self):
        monitor = DriftMonitor(min_samples=10)
        assert self._fill(monitor, 5, {"price": None}) == []

    def test_a_collapsing_field_alerts(self):
        monitor = DriftMonitor(min_samples=10, max_drop=0.3)
        self._fill(monitor, 10, {"price": "19.99", "title": "Widget"})
        alerts = self._fill(monitor, 20, {"price": None, "title": "Widget"})
        assert any(alert.field == "price" for alert in alerts)
        assert not any(alert.field == "title" for alert in alerts)

    def test_a_steady_field_never_alerts(self):
        monitor = DriftMonitor(min_samples=10, max_drop=0.3)
        assert self._fill(monitor, 40, {"price": "19.99"}) == []

    def test_a_field_that_was_never_reliable_is_not_alerted_on(self):
        """No baseline worth defending means no alert worth raising."""
        monitor = DriftMonitor(min_samples=5, max_drop=0.2)
        for index in range(5):
            monitor.observe("https://a.test/x", "s", {"rare": "v" if index == 0 else None})
        alerts = []
        for _ in range(20):
            alerts.extend(monitor.observe("https://a.test/x", "s", {"rare": None}))
        assert alerts == []

    def test_disabling_drift(self, monkeypatch):
        monkeypatch.setattr(config, "DRIFT_ENABLED", False, raising=False)
        monitor = DriftMonitor(min_samples=1)
        assert monitor.observe("https://a.test/x", "s", {"a": "v"}) == []


class TestProvenance:
    def test_every_record_carries_which_tier_answered(self):
        item = ExtractionItem(url="https://a.test/x", raw_bytes=b"body")
        item.compute_content_hash()
        item.kind = ResourceKind.HTML
        item.method = ExtractionMethod.STRUCTURED_DATA
        item.tier = 1
        item.confidence = 0.9
        item.status_code = 200

        provenance = item.provenance(schema_hash="s1", prompt_hash="p1")
        assert provenance.method is ExtractionMethod.STRUCTURED_DATA
        assert provenance.tier == 1
        assert provenance.content_hash == item.content_hash
        assert provenance.schema_hash == "s1"

    def test_content_hash_covers_bytes_or_text(self):
        with_bytes = ExtractionItem(url="https://a.test/x", raw_bytes=b"body")
        assert with_bytes.compute_content_hash()

        # A transcript has no bytes: the audio was deleted with its temp dir.
        transcript_only = ExtractionItem(url="https://a.test/y", cleaned_text="a transcript")
        assert transcript_only.compute_content_hash()

        assert ExtractionItem(url="https://a.test/z").compute_content_hash() is None


class TestRunReport:
    def test_counts_and_tier_breakdown(self):
        report = RunReport()
        for tier, method in ((1, ExtractionMethod.STRUCTURED_DATA), (3, ExtractionMethod.LLM)):
            item = ExtractionItem(url=f"https://a.test/{tier}")
            item.method, item.tier = method, tier
            report.record(item)
        failed = ExtractionItem(url="https://a.test/bad")
        from models import Stage

        failed.fail(Stage.FETCH, "404")
        report.record(failed)

        assert (report.submitted, report.succeeded, report.failed) == (3, 2, 1)
        assert report.by_method["structured"] == 1
        assert report.errors[0]["stage"] == "FETCH"

    def test_llm_avoidance_rate_is_the_headline_number(self):
        report = RunReport()
        for index in range(9):
            item = ExtractionItem(url=f"https://a.test/{index}")
            item.method, item.tier = ExtractionMethod.STRUCTURED_DATA, 1
            report.record(item)
        item = ExtractionItem(url="https://a.test/llm")
        item.method, item.tier = ExtractionMethod.LLM, 3
        report.record(item)

        assert report.llm_avoidance_rate == 0.9

    def test_no_successes_means_no_rate(self):
        assert RunReport().llm_avoidance_rate == 0.0
