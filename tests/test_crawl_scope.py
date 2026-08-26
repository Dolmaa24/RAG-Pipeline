"""What a crawl follows, collects and refuses.

The regression these were written for: a crawl naming no file types collected
HTML and nothing else. ``collect_extensions=[]`` is documented as "extract every
in-scope page", and every document on the site fell between the two rules that
were supposed to implement it -- the target check declined because there was no
filter to match, and the page check refused anything that was not a page.

Measured on one real page linking 54 syllabus PDFs: 14 navigation pages
collected, zero PDFs.
"""

from __future__ import annotations

from pipeline.discover.scope import CrawlScope, LinkVerdict, ResourceKind

BASE = "https://example.ac.in/papers_and_syllabus"
PDF = "https://example.ac.in/static/doc/CS_Syllabus.pdf"
PAGE = "https://example.ac.in/faqs"


def verdict(scope: CrawlScope, url: str, depth: int = 0) -> LinkVerdict:
    return scope.classify(url, depth=depth).verdict


def everything(**kw) -> CrawlScope:
    """A crawl that names no file types."""
    return CrawlScope(start_url=BASE, max_depth=1, **kw)


def hunting(*extensions: str, **kw) -> CrawlScope:
    """A crawl looking for particular files."""
    return CrawlScope(
        start_url=BASE, max_depth=1, collect_extensions=frozenset(extensions), **kw
    )


def test_a_crawl_naming_no_file_types_collects_the_documents_it_finds():
    assert verdict(everything(), PDF) is LinkVerdict.COLLECT


def test_it_still_follows_pages():
    assert verdict(everything(), PAGE) is LinkVerdict.FOLLOW


def test_data_and_archives_count_as_documents():
    scope = everything()
    for url in (
        "https://example.ac.in/results.csv",
        "https://example.ac.in/feed.json",
        "https://example.ac.in/records.xml",
        "https://example.ac.in/bundle.zip",
        "https://example.ac.in/message.eml",
    ):
        assert verdict(scope, url) is LinkVerdict.COLLECT, url


def test_media_is_not_collected_implicitly():
    """Otherwise "crawl this site" means OCR on every logo and icon."""
    scope = everything()
    for url in (
        "https://example.ac.in/static/logo.png",
        "https://example.ac.in/static/hero.jpg",
        "https://example.ac.in/static/intro.mp4",
        "https://example.ac.in/static/theme.mp3",
    ):
        assert verdict(scope, url) is LinkVerdict.SKIP, url


def test_media_is_collected_when_it_is_asked_for_by_name():
    """The explicit request still works -- that is how you crawl for images."""
    assert verdict(hunting("png"), "https://example.ac.in/static/scan.png") is LinkVerdict.COLLECT


def test_page_furniture_is_never_collected():
    scope = everything()
    for url in ("https://example.ac.in/app.css", "https://example.ac.in/app.js"):
        assert verdict(scope, url) is LinkVerdict.SKIP, url


def test_a_file_hunt_collects_only_what_it_named():
    scope = hunting("pdf")
    assert verdict(scope, PDF) is LinkVerdict.COLLECT
    assert verdict(scope, "https://example.ac.in/results.csv") is LinkVerdict.SKIP


def test_a_file_hunt_still_walks_pages_for_their_links():
    """Pages are the map, not the destination."""
    assert verdict(hunting("pdf"), PAGE) is LinkVerdict.FOLLOW


def test_a_named_target_is_collected_past_the_depth_limit():
    """The whole point of hunting for PDFs is that the one at depth 3 counts."""
    assert verdict(hunting("pdf"), PDF, depth=9) is LinkVerdict.COLLECT


def test_an_implicitly_collected_document_stays_within_the_depth_limit():
    """Naming no types asks for what lies inside the depth already set."""
    scope = everything()
    assert verdict(scope, PDF, depth=0) is LinkVerdict.COLLECT
    assert verdict(scope, PDF, depth=1) is LinkVerdict.SKIP


def test_an_off_site_document_is_refused():
    assert verdict(everything(), "https://elsewhere.test/syllabus.pdf") is LinkVerdict.SKIP


def test_an_excluded_document_is_refused():
    scope = CrawlScope(
        start_url=BASE, max_depth=1, exclude_patterns=(r"/archive/",)
    )
    assert verdict(scope, "https://example.ac.in/archive/old.pdf") is LinkVerdict.SKIP


def test_an_include_pattern_filters_documents_as_well_as_targets():
    scope = CrawlScope(
        start_url=BASE, max_depth=1, include_patterns=(r"Syllabus",)
    )
    assert verdict(scope, PDF) is LinkVerdict.COLLECT
    assert verdict(scope, "https://example.ac.in/static/doc/Notice.pdf") is LinkVerdict.SKIP


def test_the_two_shapes_report_different_reasons():
    """A trace has to say which rule collected a file, not merely that one did."""
    assert "document" in everything().classify(PDF, depth=0).reason
    assert "target file" in hunting("pdf").classify(PDF, depth=0).reason


def test_collects_everything_is_what_distinguishes_them():
    assert everything().collects_everything is True
    assert hunting("pdf").collects_everything is False


def test_documents_are_not_followed_for_links():
    """A PDF is collected and extracted; it is not a page to walk."""
    assert verdict(everything(), PDF) is not LinkVerdict.FOLLOW
    assert ResourceKind.DOCUMENT is not ResourceKind.HTML
