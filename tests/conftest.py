"""Shared fixtures.

Every test here is **offline**. Network calls go through ``respx``, robots.txt
comes from an injected fetcher, and clocks are either real-but-tiny or injected.
A test suite that needs the internet is a test suite that fails on a train, and
one that needs Ollama running is one nobody runs before committing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402
from models import ExtractionItem  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch):
    """Predictable settings, and never a real network gate in a unit test."""
    # No test may touch a real database. `CloudDatabase(uri=None)` falls back to
    # config.MONGO_URI, so the only way to guarantee an unconfigured database is
    # to blank the setting — otherwise a developer with MONGO_URI in .env runs a
    # different suite from one without it.
    monkeypatch.setattr(config, "MONGO_URI", None, raising=False)
    monkeypatch.setattr(config, "RESPECT_ROBOTS", False, raising=False)
    monkeypatch.setattr(config, "MAX_RETRIES", 1, raising=False)
    monkeypatch.setattr(config, "USE_BROWSER_FALLBACK", False, raising=False)
    monkeypatch.setattr(config, "HTTP_CACHE_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "EXTRACTION_CACHE_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "DEDUPE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "ALLOW_PRIVATE_ADDRESSES", True, raising=False)
    yield


@pytest.fixture(autouse=True)
def _reset_shared_state():
    """Process-wide singletons must not leak between tests."""
    from pipeline.extract import cascade as cascade_module
    from pipeline.extract.llm import reset as reset_llm
    from pipeline.fetch import breaker, http_cache, rate_limiter

    breaker.reset()
    rate_limiter.reset()
    http_cache.clear()
    cascade_module.reset()
    reset_llm()
    yield
    breaker.reset()
    rate_limiter.reset()


@pytest.fixture
def item_factory():
    def make(url: str = "https://example.test/page", **fields) -> ExtractionItem:
        item = ExtractionItem(url=url)
        for name, value in fields.items():
            setattr(item, name, value)
        return item

    return make


@pytest.fixture
def product_html() -> bytes:
    """A page with JSON-LD, OpenGraph, a table and ordinary prose."""
    return b"""<!DOCTYPE html>
<html lang="en"><head>
<title>Blue Widget | ACME</title>
<meta name="description" content="A very blue widget.">
<meta property="og:image" content="/img/widget.png">
<meta property="og:site_name" content="ACME Store">
<script type="application/ld+json">
{"@context":"https://schema.org","@graph":[
 {"@type":"BreadcrumbList","itemListElement":[]},
 {"@type":"Organization","name":"ACME Store"},
 {"@type":"Product","name":"Blue Widget","sku":"BW-42",
  "brand":{"@type":"Brand","name":"ACME"},
  "description":"A very blue widget.","image":"/img/widget.png",
  "aggregateRating":{"@type":"AggregateRating","ratingValue":"4.6","reviewCount":"128"},
  "offers":{"@type":"Offer","price":"19.99","priceCurrency":"USD",
            "availability":"https://schema.org/InStock"}}]}
</script></head>
<body>
<nav>Home Shop Contact</nav>
<article class="product">
  <h1 class="product-title">Blue Widget</h1>
  <span class="price">$19.99</span>
  <p class="stock">In stock</p>
  <p>It is blue, and it is a widget. Blue widgets are widgets that are blue.</p>
  <table><tr><th>Spec</th><th>Value</th></tr>
         <tr><td>Colour</td><td>Blue</td></tr>
         <tr><td>Weight</td><td>2kg</td></tr></table>
</article>
<footer>&copy; ACME</footer></body></html>"""


@pytest.fixture
def plain_html() -> bytes:
    """A page with no structured data at all, so tier 1 must fall through."""
    return b"""<!DOCTYPE html><html><head><title>Notes</title></head><body>
<main><h1>Field notes</h1><p>The heron stood still for eleven minutes.</p></main>
</body></html>"""


class FakeBackend:
    """An LLM backend that returns a canned answer and counts its calls."""

    name = "fake"
    model = "fake-1"

    def __init__(self, payload: dict | None = None, *, fail: Exception | None = None) -> None:
        self.payload = payload or {}
        self.fail = fail
        self.calls: list[dict] = []

    def available(self) -> bool:
        return True

    def complete_json(self, *, prompt, content, schema_hint, json_schema):
        from pipeline.extract.llm.base import LLMResponse

        self.calls.append({"prompt": prompt, "content": content, "schema": schema_hint})
        if self.fail is not None:
            raise self.fail
        return LLMResponse(data=dict(self.payload), backend=self.name, model=self.model)


class ExplodingBackend:
    """Fails the test if the model tier is reached at all."""

    name = "exploding"
    model = "none"

    def available(self) -> bool:
        return True

    def complete_json(self, **_kwargs):
        raise AssertionError("the model tier was reached but should not have been")


@pytest.fixture
def fake_backend():
    return FakeBackend


@pytest.fixture
def exploding_backend():
    return ExplodingBackend()
