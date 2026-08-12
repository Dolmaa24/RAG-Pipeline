"""The pipeline end to end, with the network stubbed."""

from __future__ import annotations

import io
import zipfile

import httpx
import pytest
import respx

from config import config
from models import ExtractionItem, ExtractionMethod, ResourceKind
from pipeline.extract.cascade import ExtractionCascade
from pipeline.runner import Pipeline

SCHEMA = {"title": "string", "price": "string"}
PROMPT = "Extract the title and price."


@pytest.fixture
def pipeline(exploding_backend):
    """A pipeline whose model tier fails the test if it is ever reached."""
    return Pipeline(cascade=ExtractionCascade(backend=exploding_backend))


def run(pipeline: Pipeline, url: str, **options) -> ExtractionItem:
    return pipeline.run_item(ExtractionItem(url=url), PROMPT, SCHEMA, **options)


class TestHappyPath:
    @respx.mock
    def test_html_page_with_structured_data(self, pipeline, product_html):
        respx.get("https://shop.test/p/1").mock(
            return_value=httpx.Response(
                200, content=product_html, headers={"content-type": "text/html"}
            )
        )
        item = run(pipeline, "https://shop.test/p/1")
        assert item.ok
        assert item.kind is ResourceKind.HTML
        assert item.tier == 1
        assert item.normalized_data["title"] == "Blue Widget"
        assert item.normalized_data["price"] == "19.99"

    @respx.mock
    def test_typed_companions_are_added(self, pipeline, product_html):
        respx.get("https://shop.test/p/1").mock(
            return_value=httpx.Response(200, content=product_html)
        )
        item = run(pipeline, "https://shop.test/p/1")
        assert item.normalized_data["price_amount"] == "19.99"

    @respx.mock
    def test_provenance_records_the_whole_journey(self, pipeline, product_html):
        respx.get("https://shop.test/p/1").mock(
            return_value=httpx.Response(200, content=product_html, headers={"etag": '"v1"'})
        )
        item = run(pipeline, "https://shop.test/p/1")
        provenance = item.provenance()
        assert provenance.http_status == 200
        assert provenance.method is ExtractionMethod.STRUCTURED_DATA
        assert provenance.content_hash
        assert provenance.kind is ResourceKind.HTML

    @respx.mock
    def test_timings_are_recorded_per_stage(self, pipeline, product_html):
        respx.get("https://shop.test/p/1").mock(
            return_value=httpx.Response(200, content=product_html)
        )
        item = run(pipeline, "https://shop.test/p/1")
        assert "fetch" in item.timings_ms
        assert "total" in item.timings_ms


class TestNonHtmlInputs:
    @respx.mock
    def test_a_pdf_url_is_detected_from_its_bytes(self, pipeline):
        import pymupdf

        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 100), "Quarterly Report")
        page.insert_text((72, 130), "Price: 42.00")
        pdf = document.tobytes()
        document.close()

        # Served with a lying Content-Type from a URL with no extension.
        respx.get("https://a.test/download?id=8412").mock(
            return_value=httpx.Response(200, content=pdf, headers={"content-type": "text/html"})
        )
        item = pipeline.run_item(
            ExtractionItem(url="https://a.test/download?id=8412"),
            PROMPT, SCHEMA, allowed_tiers={1},
        )
        assert item.kind is ResourceKind.DOCUMENT
        assert "Quarterly Report" in item.cleaned_text

    @respx.mock
    def test_a_csv_answers_from_its_own_rows(self, pipeline):
        respx.get("https://a.test/rows").mock(
            return_value=httpx.Response(200, content=b"title,price\nWidget,19.99\nGadget,5.00\n")
        )
        item = pipeline.run_item(
            ExtractionItem(url="https://a.test/rows"), PROMPT, SCHEMA, allowed_tiers={1}
        )
        assert item.kind is ResourceKind.TABULAR
        assert item.tier == 1


class TestRecursion:
    @respx.mock
    def test_archive_members_are_each_run_through_the_pipeline(self, pipeline):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("a.csv", b"title,price\nWidget,19.99\n")
            archive.writestr("b.csv", b"title,price\nGadget,5.00\n")

        respx.get("https://a.test/bundle.zip").mock(
            return_value=httpx.Response(200, content=buffer.getvalue())
        )
        item = pipeline.run_item(
            ExtractionItem(url="https://a.test/bundle.zip"), PROMPT, SCHEMA, allowed_tiers={1}
        )
        assert item.kind is ResourceKind.ARCHIVE
        assert len(item.children) == 2
        assert all(child.kind is ResourceKind.TABULAR for child in item.children)
        assert any(child.normalized_data for child in item.children)

    @respx.mock
    def test_children_run_even_when_the_container_extracts_nothing(self, pipeline):
        """A ZIP has no title and no price. Its members are the whole point."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("a.csv", b"title,price\nWidget,19.99\n")
        respx.get("https://a.test/d.zip").mock(
            return_value=httpx.Response(200, content=buffer.getvalue())
        )
        item = pipeline.run_item(
            ExtractionItem(url="https://a.test/d.zip"), PROMPT, SCHEMA, allowed_tiers={1}
        )
        assert not item.ok, "the container itself has nothing matching the schema"
        assert item.children[0].normalized_data["title"] == "Widget"

    @respx.mock
    def test_child_bytes_are_released_after_the_run(self, pipeline):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("a.csv", b"title,price\nWidget,19.99\n")
        respx.get("https://a.test/b.zip").mock(
            return_value=httpx.Response(200, content=buffer.getvalue())
        )
        item = pipeline.run_item(
            ExtractionItem(url="https://a.test/b.zip"), PROMPT, SCHEMA, allowed_tiers={1}
        )
        assert all(child.raw_bytes is None for child in item.children)

    @respx.mock
    def test_feed_urls_are_discovered_not_followed_inline(self, pipeline):
        """Following 100 links inside one task defeats the queue split."""
        rss = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>Notes</title>
        <item><title>One</title><link>https://a.test/1</link></item>
        <item><title>Two</title><link>https://a.test/2</link></item></channel></rss>"""
        respx.get("https://a.test/feed.xml").mock(
            return_value=httpx.Response(200, content=rss, headers={"content-type": "application/xml"})
        )
        entry = respx.get("https://a.test/1").mock(return_value=httpx.Response(200, content=b"x"))

        item = pipeline.run_item(
            ExtractionItem(url="https://a.test/feed.xml"), PROMPT, SCHEMA, allowed_tiers={1}
        )
        assert item.kind is ResourceKind.FEED
        assert item.metadata["discovered_urls"] == ["https://a.test/1", "https://a.test/2"]
        assert entry.call_count == 0, "a link must not be fetched inside the parent's task"

    @respx.mock
    def test_recursion_can_be_switched_off(self, pipeline):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("a.csv", b"title,price\nWidget,19.99\n")
        respx.get("https://a.test/c.zip").mock(
            return_value=httpx.Response(200, content=buffer.getvalue())
        )
        item = pipeline.run_item(
            ExtractionItem(url="https://a.test/c.zip"),
            PROMPT, SCHEMA, allowed_tiers={1}, follow_children=False,
        )
        assert all(child.normalized_data is None for child in item.children)


class TestFailureHandling:
    @respx.mock
    def test_one_failure_does_not_stop_a_batch(self, pipeline, product_html):
        respx.get("https://a.test/ok").mock(
            return_value=httpx.Response(200, content=product_html)
        )
        respx.get("https://a.test/gone").mock(return_value=httpx.Response(404))

        report = pipeline.run_batch(
            ["https://a.test/ok", "https://a.test/gone"], PROMPT, SCHEMA
        )
        assert report.succeeded == 1
        assert report.failed == 1
        assert report.errors[0]["stage"] == "FETCH"

    @respx.mock
    def test_the_report_shows_the_tier_breakdown(self, pipeline, product_html):
        respx.get("https://a.test/1").mock(
            return_value=httpx.Response(200, content=product_html)
        )
        respx.get("https://a.test/2").mock(
            return_value=httpx.Response(200, content=product_html)
        )
        report = pipeline.run_batch(["https://a.test/1", "https://a.test/2"], PROMPT, SCHEMA)
        assert report.by_method["structured"] == 2
        assert report.llm_avoidance_rate == 1.0


class TestTrustIntegration:
    @respx.mock
    def test_a_repeated_page_is_flagged_as_a_duplicate(self, pipeline, product_html):
        respx.get("https://a.test/1").mock(
            return_value=httpx.Response(200, content=product_html)
        )
        respx.get("https://a.test/2").mock(
            return_value=httpx.Response(200, content=product_html)
        )
        first = run(pipeline, "https://a.test/1")
        second = run(pipeline, "https://a.test/2")
        assert "simhash" in first.metadata
        assert second.metadata.get("duplicate_of") == "https://a.test/1"

    @respx.mock
    def test_validation_failures_are_recorded_not_hidden(self, product_html, fake_backend):
        backend = fake_backend({"title": "N/A", "price": "unknown"})
        pipeline = Pipeline(cascade=ExtractionCascade(backend=backend))
        respx.get("https://a.test/x").mock(
            return_value=httpx.Response(200, content=b"<html><body><main><p>"
                                        b"Some prose with no structure at all in it.</p></main></body></html>")
        )
        item = pipeline.run_item(
            ExtractionItem(url="https://a.test/x"), PROMPT, SCHEMA, allowed_tiers={3}
        )
        assert item.validation_failures
        assert item.ok, "a flagged record is still returned by default"

    @respx.mock
    def test_rejection_mode_fails_the_item(self, product_html, fake_backend, monkeypatch):
        monkeypatch.setattr(config, "REJECT_ON_VALIDATION_FAILURE", True, raising=False)
        backend = fake_backend({"title": "N/A", "price": "N/A"})
        pipeline = Pipeline(cascade=ExtractionCascade(backend=backend))
        respx.get("https://a.test/y").mock(
            return_value=httpx.Response(200, content=b"<html><body><main><p>"
                                        b"Some prose with no structure at all in it.</p></main></body></html>")
        )
        item = pipeline.run_item(
            ExtractionItem(url="https://a.test/y"), PROMPT, SCHEMA, allowed_tiers={3}
        )
        assert not item.ok


class TestProgress:
    @respx.mock
    def test_stages_are_reported(self, product_html, exploding_backend):
        seen: list[str] = []
        pipeline = Pipeline(
            cascade=ExtractionCascade(backend=exploding_backend),
            on_progress=lambda label, item: seen.append(label),
        )
        respx.get("https://a.test/x").mock(
            return_value=httpx.Response(200, content=product_html)
        )
        pipeline.run_item(ExtractionItem(url="https://a.test/x"), PROMPT, SCHEMA)
        assert "Fetching" in seen
        assert any("Reading" in label for label in seen)

    @respx.mock
    def test_a_broken_progress_callback_never_breaks_the_run(self, product_html, exploding_backend):
        def explode(label, item):
            raise RuntimeError("the UI fell over")

        pipeline = Pipeline(
            cascade=ExtractionCascade(backend=exploding_backend), on_progress=explode
        )
        respx.get("https://a.test/x").mock(
            return_value=httpx.Response(200, content=product_html)
        )
        assert pipeline.run_item(ExtractionItem(url="https://a.test/x"), PROMPT, SCHEMA).ok
