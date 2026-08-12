"""The crawler: scope, traps, the frontier, and termination."""

from __future__ import annotations

import httpx
import pytest
import respx

from models import ExtractionItem, ResourceKind
from pipeline.detect import TypeRouter
from pipeline.discover import (
    CrawlScope,
    LinkVerdict,
    MemoryFrontier,
    describe_plan,
    harvest_links,
)
from pipeline.handlers import registry

PDF_SCOPE = dict(collect_extensions=["pdf"], max_depth=2, max_pages=100)


def scope(url: str = "https://site.test/", **options) -> CrawlScope:
    return CrawlScope.build(url, **{**PDF_SCOPE, **options})


class TestLinkClassification:
    def test_target_extension_is_collected(self):
        decision = scope().classify("https://site.test/docs/report.pdf", depth=0)
        assert decision.verdict is LinkVerdict.COLLECT

    def test_page_is_followed(self):
        decision = scope().classify("https://site.test/docs/index.html", depth=0)
        assert decision.verdict is LinkVerdict.FOLLOW

    def test_extensionless_url_is_treated_as_a_page(self):
        assert scope().classify("https://site.test/about", depth=0).verdict is LinkVerdict.FOLLOW

    def test_offsite_is_skipped(self):
        decision = scope().classify("https://elsewhere.test/report.pdf", depth=0)
        assert decision.verdict is LinkVerdict.SKIP
        assert "off-site" in decision.reason

    def test_www_is_the_same_site(self):
        decision = scope("https://site.test/").classify("https://www.site.test/a", depth=0)
        assert decision.verdict is LinkVerdict.FOLLOW

    @pytest.mark.parametrize("url", [
        "https://site.test/app.css", "https://site.test/main.js", "https://site.test/icon.ico",
    ])
    def test_page_furniture_is_skipped(self, url):
        assert scope().classify(url, depth=0).verdict is LinkVerdict.SKIP

    def test_non_target_file_is_skipped(self):
        """Looking for PDFs means a .jpg is neither a target nor a page."""
        decision = scope().classify("https://site.test/photo.jpg", depth=0)
        assert decision.verdict is LinkVerdict.SKIP

    def test_depth_limit_stops_following_but_not_collecting(self):
        """The PDF at depth 3 is still the thing you asked for."""
        deep = scope(max_depth=1)
        assert deep.classify("https://site.test/page", depth=1).verdict is LinkVerdict.SKIP
        assert deep.classify("https://site.test/a.pdf", depth=5).verdict is LinkVerdict.COLLECT

    def test_nofollow_is_honoured(self):
        decision = scope().classify("https://site.test/page", depth=0, nofollow=True)
        assert decision.verdict is LinkVerdict.SKIP

    def test_exclude_pattern(self):
        excluded = scope(exclude_patterns=[r"/archive/"])
        assert excluded.classify("https://site.test/archive/a.pdf", depth=0).verdict is LinkVerdict.SKIP
        assert excluded.classify("https://site.test/current/a.pdf", depth=0).verdict is LinkVerdict.COLLECT

    def test_include_pattern_filters_collection_but_still_walks_pages(self):
        """A page outside the pattern may still link to a file inside it."""
        included = scope(include_patterns=[r"/reports/"])
        assert included.classify("https://site.test/reports/a.pdf", depth=0).verdict is LinkVerdict.COLLECT
        assert included.classify("https://site.test/other/a.pdf", depth=0).verdict is LinkVerdict.SKIP
        assert included.classify("https://site.test/other/page", depth=0).verdict is LinkVerdict.FOLLOW

    def test_multiple_extensions(self):
        multi = scope(collect_extensions=["pdf", "xlsx"])
        assert multi.classify("https://site.test/a.xlsx", depth=0).verdict is LinkVerdict.COLLECT
        assert multi.classify("https://site.test/a.pdf", depth=0).verdict is LinkVerdict.COLLECT

    def test_no_filter_means_every_page_is_extracted(self):
        everything = CrawlScope.build("https://site.test/")
        assert everything.collects_everything
        assert everything.should_extract(ResourceKind.HTML, "https://site.test/a")

    def test_a_file_filter_excludes_html_from_extraction(self):
        """Pages are the map, not the destination."""
        hunting = scope()
        assert not hunting.should_extract(ResourceKind.HTML, "https://site.test/a")
        assert hunting.should_extract(ResourceKind.DOCUMENT, "https://site.test/a.pdf")


class TestTrapDetection:
    @pytest.mark.parametrize("url,shape", [
        ("https://site.test/a/b/a/b/a/b/", "repeated path segments"),
        ("https://site.test/2026/03/04/", "calendar-style URL"),
        ("https://site.test/p?year=2026", "calendar-style URL"),
        ("https://site.test/p?sessionid=abc", "session or view parameter"),
        ("https://site.test/p?a=1&b=2&c=3&d=4&e=5&f=6&g=7", "too many query parameters"),
        ("https://site.test/" + "x/" * 20, "path is implausibly deep"),
    ])
    def test_traps_are_named(self, url, shape):
        found = scope().is_trap(url)
        assert found is not None and shape in found

    @pytest.mark.parametrize("url", [
        "https://site.test/docs/annual-report.pdf",
        "https://site.test/catalogue/category/books/travel_2/index.html",
        "https://site.test/search?q=heron&page=2",
    ])
    def test_ordinary_urls_are_not_traps(self, url):
        assert scope().is_trap(url) is None

    def test_trap_detection_can_be_disabled(self):
        assert scope(detect_traps=False).classify(
            "https://site.test/2026/03/04/", depth=0
        ).verdict is LinkVerdict.FOLLOW


class TestFrontier:
    def test_a_url_is_claimed_only_once(self):
        """Without this, a shared nav bar re-enqueues every page from every page."""
        frontier = MemoryFrontier("c1", "https://site.test/", max_pages=100)
        first = frontier.claim([("https://site.test/a", 0), ("https://site.test/b", 0)])
        second = frontier.claim([("https://site.test/a", 1), ("https://site.test/c", 1)])
        assert len(first) == 2
        assert second == [("https://site.test/c", 1)]

    def test_budget_is_hard(self):
        frontier = MemoryFrontier("c2", "https://site.test/", max_pages=3)
        claimed = frontier.claim([(f"https://site.test/{i}", 0) for i in range(10)])
        assert len(claimed) == 3
        assert frontier.claim([("https://site.test/later", 0)]) == []

    def test_counters_and_completion(self):
        frontier = MemoryFrontier("c3", "https://site.test/", max_pages=10)
        frontier.claim([("https://site.test/a", 0)])
        frontier.bump("in_flight")
        assert not frontier.state().is_done
        frontier.bump("fetched")
        frontier.bump("in_flight", -1)
        assert frontier.state().is_done

    def test_targets_are_remembered(self):
        frontier = MemoryFrontier("c4", "https://site.test/", max_pages=10)
        frontier.record_target("https://site.test/a.pdf")
        frontier.record_target("https://site.test/b.pdf")
        assert frontier.targets() == ["https://site.test/a.pdf", "https://site.test/b.pdf"]

    def test_stop_request(self):
        frontier = MemoryFrontier("c5", "https://site.test/", max_pages=10)
        assert not frontier.stop_requested()
        frontier.finish("stopped")
        assert frontier.stop_requested()


PAGE = b"""<!DOCTYPE html><html><head><title>Docs</title></head><body><main>
<p>Reports index page with enough text to parse.</p>
<a href="/docs/q1.pdf">Q1</a>
<a href="/docs/q2.pdf">Q2</a>
<a href="/docs/more">More</a>
<a href="/style.css">css</a>
<a href="https://elsewhere.test/x.pdf">offsite</a>
<a href="/sponsor" rel="nofollow">sponsor</a>
<a href="/docs/q1.pdf">Q1 again</a>
</main></body></html>"""


def parse(url: str, body: bytes) -> ExtractionItem:
    item = ExtractionItem(url=url, raw_bytes=body, content_type="text/html")
    TypeRouter.route(item)
    return registry.dispatch(item)


class TestHarvest:
    def test_splits_links_into_follow_and_collect(self):
        item = parse("https://site.test/docs/", PAGE)
        harvest = harvest_links(item, scope(), depth=0)
        assert sorted(harvest.collect) == [
            "https://site.test/docs/q1.pdf", "https://site.test/docs/q2.pdf"
        ]
        assert harvest.follow == ["https://site.test/docs/more"]

    def test_duplicate_links_are_collapsed(self):
        """Q1 appears twice on the page."""
        item = parse("https://site.test/docs/", PAGE)
        harvest = harvest_links(item, scope(), depth=0)
        assert len(harvest.collect) == len(set(harvest.collect))

    def test_skips_are_counted_with_reasons(self):
        item = parse("https://site.test/docs/", PAGE)
        harvest = harvest_links(item, scope(), depth=0)
        assert harvest.summary()["skipped"] >= 3
        assert harvest.skipped

    def test_nofollow_link_is_not_followed(self):
        item = parse("https://site.test/docs/", PAGE)
        harvest = harvest_links(item, scope(), depth=0)
        assert "https://site.test/sponsor" not in harvest.follow

    def test_page_level_meta_nofollow_applies_to_all_links(self):
        body = b"""<html><head><meta name="robots" content="nofollow"></head><body><main>
        <p>A page that asks not to be walked, with enough text to parse.</p>
        <a href="/a">A</a><a href="/b">B</a></main></body></html>"""
        item = parse("https://site.test/x", body)
        assert harvest_links(item, scope(), depth=0).follow == []

    def test_a_page_with_no_links_harvests_nothing(self):
        item = parse("https://site.test/x", b"<html><body><main><p>Just words here.</p></main></body></html>")
        assert harvest_links(item, scope(), depth=0).total == 0


class TestTermination:
    """The properties that stop a crawl becoming an outage."""

    def test_a_self_linking_page_does_not_loop(self):
        body = b"""<html><body><main><p>A page that links to itself repeatedly.</p>
        <a href="/loop">loop</a><a href="/loop">loop</a></main></body></html>"""
        frontier = MemoryFrontier("t1", "https://site.test/", max_pages=100)
        crawl_scope = CrawlScope.build("https://site.test/", max_depth=5)

        queued = 0
        for _ in range(10):
            item = parse("https://site.test/loop", body)
            harvest = harvest_links(item, crawl_scope, depth=0)
            claimed = frontier.claim([(url, 1) for url in harvest.follow])
            queued += len(claimed)
        assert queued == 1, "the frontier must accept /loop exactly once"

    def test_mutually_linking_pages_terminate(self):
        frontier = MemoryFrontier("t2", "https://site.test/", max_pages=100)
        crawl_scope = CrawlScope.build("https://site.test/", max_depth=10)
        pages = {
            "https://site.test/a": b"<html><body><main><p>Page A with words.</p><a href='/b'>b</a></main></body></html>",
            "https://site.test/b": b"<html><body><main><p>Page B with words.</p><a href='/a'>a</a></main></body></html>",
        }
        pending = frontier.claim([("https://site.test/a", 0)])
        visited = 0
        while pending and visited < 50:
            url, depth = pending.pop()
            visited += 1
            item = parse(url, pages[url])
            harvest = harvest_links(item, crawl_scope, depth=depth)
            pending.extend(frontier.claim([(u, depth + 1) for u in harvest.follow]))
        assert visited == 2

    def test_depth_limit_terminates_an_infinite_chain(self):
        crawl_scope = CrawlScope.build("https://site.test/", max_depth=2)
        assert crawl_scope.classify("https://site.test/x", depth=0).verdict is LinkVerdict.FOLLOW
        assert crawl_scope.classify("https://site.test/x", depth=1).verdict is LinkVerdict.FOLLOW
        assert crawl_scope.classify("https://site.test/x", depth=2).verdict is LinkVerdict.SKIP


class TestExtractionFailureDoesNotEndTheCrawl:
    @respx.mock
    def test_links_survive_a_failed_extraction(self, fake_backend):
        """One 429 on the seed page must not silently end the whole crawl."""
        from errors import TransientExtractError
        from pipeline.extract.cascade import ExtractionCascade
        from pipeline.runner import Pipeline

        respx.get("https://site.test/docs/").mock(
            return_value=httpx.Response(200, content=PAGE, headers={"content-type": "text/html"})
        )
        backend = fake_backend({}, fail=TransientExtractError("rate limited"))
        pipeline = Pipeline(cascade=ExtractionCascade(backend=backend))

        item = pipeline.run_item(
            ExtractionItem(url="https://site.test/docs/"),
            "Extract the title.",
            {"title": "string"},
            allowed_tiers={3},
        )
        assert not item.ok, "extraction did fail"
        assert item.parsed_tree["links"], "but the page still parsed"

        harvest = harvest_links(item, scope(), depth=0)
        assert harvest.collect, "and its links are still usable"


class TestPlanDescription:
    def test_file_hunt(self):
        assert "collect .pdf from site.test" in describe_plan(scope())

    def test_every_page(self):
        assert "every page" in describe_plan(CrawlScope.build("https://site.test/"))


class TestScopeRoundTrip:
    def test_survives_serialisation(self):
        """The scope crosses the Celery queue as JSON and is rebuilt per task."""
        original = scope(include_patterns=[r"/reports/"], exclude_patterns=[r"/old/"], max_depth=4)
        rebuilt = CrawlScope.from_dict(original.to_dict())
        assert rebuilt.to_dict() == original.to_dict()
        assert rebuilt.classify("https://site.test/reports/a.pdf", depth=0).verdict is LinkVerdict.COLLECT


class TestRunnerIntegration:
    @respx.mock
    def test_pages_are_walked_without_being_extracted(self, exploding_backend):
        """extract_when is what lets a crawl read a nav page without paying for it."""
        from pipeline.extract.cascade import ExtractionCascade
        from pipeline.runner import Pipeline

        respx.get("https://site.test/docs/").mock(
            return_value=httpx.Response(200, content=PAGE, headers={"content-type": "text/html"})
        )
        crawl_scope = scope()
        pipeline = Pipeline(cascade=ExtractionCascade(backend=exploding_backend))
        item = pipeline.run_item(
            ExtractionItem(url="https://site.test/docs/"),
            "Extract the title.",
            {"title": "string"},
            extract_when=lambda parsed: crawl_scope.should_extract(parsed.kind, parsed.url),
        )
        assert item.ok
        assert item.extracted_data is None
        assert item.metadata["extraction_skipped"]
        assert item.parsed_tree["links"], "links are still harvested for the crawl"
