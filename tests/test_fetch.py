"""The fetcher and its gates. Every request is stubbed with respx."""

from __future__ import annotations

import httpx
import pytest
import respx

from config import config
from models import ExtractionItem, Stage
from pipeline.fetch.blocks import detect_block
from pipeline.fetch.breaker import BreakerState, CircuitBreaker
from pipeline.fetch.cache import CachedResponse, HttpCache, MemoryCache
from pipeline.fetch.client import ResilientFetcher
from pipeline.fetch.ratelimit import TokenBucket, parse_retry_after

HTML = b"<!DOCTYPE html><html><body><p>hello</p></body></html>"


@pytest.fixture
def fetcher():
    return ResilientFetcher()


class TestBasicFetch:
    @respx.mock
    def test_success(self, fetcher):
        respx.get("https://a.test/page").mock(
            return_value=httpx.Response(200, content=HTML, headers={"content-type": "text/html"})
        )
        item = fetcher.fetch(ExtractionItem(url="https://a.test/page"))
        assert item.ok
        assert item.status_code == 200
        assert item.raw_bytes == HTML
        assert item.content_hash  # set at fetch time, not later

    @respx.mock
    def test_404_is_not_retried(self, fetcher):
        route = respx.get("https://a.test/gone").mock(return_value=httpx.Response(404))
        item = fetcher.fetch(ExtractionItem(url="https://a.test/gone"))
        assert not item.ok
        assert item.failed_at_stage is Stage.FETCH
        assert route.call_count == 1

    @respx.mock
    def test_503_is_retried(self, fetcher, monkeypatch):
        monkeypatch.setattr(config, "MAX_RETRIES", 2, raising=False)
        route = respx.get("https://a.test/flaky").mock(
            side_effect=[httpx.Response(503), httpx.Response(200, content=HTML)]
        )
        item = fetcher.fetch(ExtractionItem(url="https://a.test/flaky"))
        assert item.ok
        assert route.call_count == 2

    @respx.mock
    def test_body_over_the_cap_is_abandoned(self, fetcher, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONTENT_BYTES", 1024, raising=False)
        respx.get("https://a.test/big").mock(
            return_value=httpx.Response(200, content=b"x" * 4096)
        )
        item = fetcher.fetch(ExtractionItem(url="https://a.test/big"))
        assert not item.ok
        assert "over the" in item.error

    @respx.mock
    def test_declared_content_length_over_the_cap_is_refused(self, fetcher, monkeypatch):
        monkeypatch.setattr(config, "MAX_CONTENT_BYTES", 1024, raising=False)
        respx.get("https://a.test/huge").mock(
            return_value=httpx.Response(200, content=b"x" * 10, headers={"content-length": "999999"})
        )
        item = fetcher.fetch(ExtractionItem(url="https://a.test/huge"))
        assert not item.ok


class TestRedirects:
    @respx.mock
    def test_chain_is_followed_and_recorded(self, fetcher):
        respx.get("https://a.test/1").mock(
            return_value=httpx.Response(302, headers={"location": "https://a.test/2"})
        )
        respx.get("https://a.test/2").mock(return_value=httpx.Response(200, content=HTML))

        item = fetcher.fetch(ExtractionItem(url="https://a.test/1"))
        assert item.ok
        assert item.final_url == "https://a.test/2"
        assert item.redirect_chain == ["https://a.test/1"]

    @respx.mock
    def test_every_hop_is_policy_checked(self, monkeypatch):
        """A 302 onto a disallowed path must not be invisible."""
        monkeypatch.setattr(config, "RESPECT_ROBOTS", True, raising=False)

        from pipeline.compliance.policy import FetchPolicy
        from pipeline.compliance.robots import RobotsGate

        robots = b"User-agent: *\nDisallow: /private/\n"
        gate = RobotsGate("*", fetcher=lambda _url: (200, robots))
        fetcher = ResilientFetcher(policy=FetchPolicy(gate=gate))

        respx.get("https://a.test/public").mock(
            return_value=httpx.Response(302, headers={"location": "https://a.test/private/x"})
        )
        private = respx.get("https://a.test/private/x").mock(
            return_value=httpx.Response(200, content=HTML)
        )

        item = fetcher.fetch(ExtractionItem(url="https://a.test/public"))
        assert not item.ok
        assert item.error_type == "RobotsDisallowed"
        assert private.call_count == 0, "the disallowed hop must never be requested"

    @respx.mock
    def test_redirect_loop_is_bounded(self, fetcher, monkeypatch):
        monkeypatch.setattr(config, "MAX_REDIRECTS", 3, raising=False)
        respx.get("https://a.test/loop").mock(
            return_value=httpx.Response(302, headers={"location": "https://a.test/loop"})
        )
        item = fetcher.fetch(ExtractionItem(url="https://a.test/loop"))
        assert not item.ok
        assert "redirects" in item.error

    @respx.mock
    def test_redirect_without_location_fails(self, fetcher):
        respx.get("https://a.test/bad").mock(return_value=httpx.Response(301))
        item = fetcher.fetch(ExtractionItem(url="https://a.test/bad"))
        assert not item.ok


class TestBlockDetection:
    """A challenge page answers with HTTP 200 — that is the whole problem."""

    def test_cloudflare_interstitial_at_200(self):
        body = b"<html><head><title>Just a moment...</title></head><body>cf_chl_opt</body></html>"
        signal = detect_block(200, {}, body)
        assert signal is not None and "cloudflare" in signal.name

    def test_vendor_header_is_conclusive(self):
        assert detect_block(200, {"cf-mitigated": "challenge"}, b"") is not None
        assert detect_block(200, {"x-datadome": "protected"}, b"") is not None

    def test_ordinary_page_mentioning_access_denied_is_not_a_block(self):
        body = b"<html><body><h1>Our access denied policy</h1><p>We explain 403s.</p></body></html>"
        assert detect_block(200, {}, body) is None

    def test_marker_plus_refusal_status_is_a_block(self):
        assert detect_block(403, {}, b"<html>you have been blocked</html>") is not None

    def test_large_document_is_not_a_challenge(self):
        body = b"you have been blocked" + b"x" * (300 * 1024)
        assert detect_block(403, {}, body) is None

    @respx.mock
    def test_fetcher_raises_and_trips_the_breaker_permanently(self, fetcher):
        respx.get("https://walled.test/x").mock(
            return_value=httpx.Response(
                200, content=b"<title>Just a moment...</title>cf_chl_opt"
            )
        )
        item = fetcher.fetch(ExtractionItem(url="https://walled.test/x"))
        assert not item.ok
        assert item.error_type == "BlockedError"
        assert not item.transient, "a refusal is not a transient failure"
        assert "walled.test" in fetcher.breaker.blocked_hosts


class TestCircuitBreaker:
    def test_opens_after_the_threshold(self):
        breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        for _ in range(3):
            breaker.record_failure("https://a.test/x")
        assert breaker.state_of("https://a.test/x") is BreakerState.OPEN
        assert not breaker.allows("https://a.test/y")

    def test_success_resets_the_count(self):
        breaker = CircuitBreaker(failure_threshold=3)
        breaker.record_failure("https://a.test/x")
        breaker.record_failure("https://a.test/x")
        breaker.record_success("https://a.test/x")
        breaker.record_failure("https://a.test/x")
        assert breaker.state_of("https://a.test/x") is BreakerState.CLOSED

    def test_half_open_after_cooldown_then_closes(self):
        breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=0, recovery_successes=1)
        breaker.record_failure("https://a.test/x")
        assert breaker.allows("https://a.test/x")  # probe permitted
        assert breaker.state_of("https://a.test/x") is BreakerState.HALF_OPEN
        breaker.record_success("https://a.test/x")
        assert breaker.state_of("https://a.test/x") is BreakerState.CLOSED

    def test_a_hard_block_never_re_closes(self):
        """A CAPTCHA wall is a decision, not an outage."""
        breaker = CircuitBreaker(cooldown_seconds=0)
        breaker.record_block("https://a.test/x", "cloudflare")
        breaker.record_success("https://a.test/x")
        assert not breaker.allows("https://a.test/x")

    def test_hosts_are_independent(self):
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("https://a.test/x")
        assert breaker.allows("https://b.test/x")


class TestRateLimit:
    def test_bucket_allows_the_burst_then_refuses(self):
        bucket = TokenBucket(rate=1000, burst=3)
        assert [bucket.try_take() for _ in range(4)] == [True, True, True, False]

    def test_bucket_refills_over_time(self):
        bucket = TokenBucket(rate=100, burst=1)
        assert bucket.try_take()
        waited = bucket.take()
        assert 0 < waited < 0.5

    def test_cannot_take_more_than_capacity(self):
        with pytest.raises(ValueError):
            TokenBucket(rate=1, burst=2).take(5)

    def test_rejects_a_non_positive_rate(self):
        with pytest.raises(ValueError):
            TokenBucket(rate=0)

    @pytest.mark.parametrize("value,expected", [("30", 30.0), ("0", 0.0), (None, None), ("x", None)])
    def test_retry_after_seconds(self, value, expected):
        assert parse_retry_after(value) == expected

    def test_retry_after_http_date(self):
        assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == 0.0  # in the past


class TestHttpCache:
    @pytest.fixture(autouse=True)
    def _enable_cache(self, monkeypatch):
        monkeypatch.setattr(config, "HTTP_CACHE_ENABLED", True, raising=False)

    def test_only_stores_responses_with_a_validator(self):
        cache = HttpCache(MemoryCache())
        cache.store("https://a.test/x", 200, {"content-type": "text/html"}, b"body")
        assert cache.lookup("https://a.test/x") is None

        cache.store("https://a.test/y", 200, {"etag": '"abc"'}, b"body")
        assert cache.lookup("https://a.test/y") is not None

    def test_validators_become_conditional_headers(self):
        entry = CachedResponse(
            "https://a.test/x", 200, {"etag": '"abc"', "last-modified": "Mon, 1 Jan 2026 00:00:00 GMT"},
            b"body", 0.0,
        )
        headers = entry.validators()
        assert headers["If-None-Match"] == '"abc"'
        assert "If-Modified-Since" in headers

    def test_key_is_the_canonical_url(self):
        cache = HttpCache(MemoryCache())
        cache.store("https://a.test/x?utm_source=q", 200, {"etag": '"1"'}, b"body")
        assert cache.lookup("https://a.test/x") is not None

    @respx.mock
    def test_304_serves_the_cached_body(self, monkeypatch):
        monkeypatch.setattr(config, "HTTP_CACHE_ENABLED", True, raising=False)
        cache = HttpCache(MemoryCache())
        fetcher = ResilientFetcher(cache=cache)

        respx.get("https://a.test/doc").mock(
            return_value=httpx.Response(200, content=HTML, headers={"etag": '"v1"'})
        )
        first = fetcher.fetch(ExtractionItem(url="https://a.test/doc"))
        assert first.ok

        respx.get("https://a.test/doc").mock(return_value=httpx.Response(304))
        second = fetcher.fetch(ExtractionItem(url="https://a.test/doc"))
        assert second.ok
        assert second.raw_bytes == HTML
        assert cache.stats()["revalidated"] == 1


class TestBrowserFallbackHeuristic:
    """Rendering in Chromium costs ~300 MB and hundreds of ms. Spend it rarely."""

    def _result(self, body: bytes, status: int = 200, content_type: str = "text/html"):
        from pipeline.fetch.client import FetchResult

        return FetchResult(status, {"content-type": content_type}, body, "https://a.test/x", [])

    def test_a_small_but_complete_page_is_not_re_rendered(self):
        body = (
            b"<!DOCTYPE html><html><head><title>About</title></head><body><main>"
            b"<h1>About us</h1><p>We publish quarterly figures for the group and "
            b"its subsidiaries, updated every three months.</p></main></body></html>"
        )
        assert len(body) < 1024, "the old heuristic would have rendered this"
        assert not ResilientFetcher._should_try_browser(self._result(body))

    def test_a_javascript_app_shell_is_rendered(self):
        body = b'<!DOCTYPE html><html><body><div id="root"></div><script src="/a.js"></script></body></html>'
        assert ResilientFetcher._should_try_browser(self._result(body))

    def test_an_empty_body_is_rendered(self):
        assert ResilientFetcher._should_try_browser(self._result(b""))

    def test_a_large_body_with_no_text_is_rendered(self):
        body = b"<html><body>" + b"<div class='x'></div>" * 400 + b"</body></html>"
        assert ResilientFetcher._should_try_browser(self._result(body))

    def test_a_full_article_is_not_rendered(self):
        body = b"<html><body><article>" + b"Real prose here. " * 200 + b"</article></body></html>"
        assert not ResilientFetcher._should_try_browser(self._result(body))

    def test_non_html_is_never_rendered(self):
        assert not ResilientFetcher._should_try_browser(
            self._result(b"%PDF-1.7", content_type="application/pdf")
        )

    def test_a_blocked_status_still_triggers_the_browser(self):
        assert ResilientFetcher._should_try_browser(self._result(b"nope", status=403))


class TestSsrfGuard:
    def test_private_address_is_refused_before_connecting(self, monkeypatch):
        monkeypatch.setattr(config, "ALLOW_PRIVATE_ADDRESSES", False, raising=False)
        item = ResilientFetcher().fetch(ExtractionItem(url="http://127.0.0.1:9/x"))
        assert not item.ok
        assert item.error_type == "HostNotAllowed"

    def test_non_http_scheme_is_refused(self):
        item = ResilientFetcher().fetch(ExtractionItem(url="file:///etc/passwd"))
        assert not item.ok
