"""robots.txt handling, per RFC 9309 §2.3.1, and the operator's own policy."""

from __future__ import annotations

import pytest

from config import config
from errors import HostNotAllowed, RobotsDisallowed
from pipeline.compliance.policy import FetchPolicy
from pipeline.compliance.robots import RobotsGate

ROBOTS = b"""
User-agent: *
Disallow: /private/
Crawl-delay: 2

User-agent: TestBot
Disallow: /nope/
Crawl-delay: 5
"""


def gate_returning(status, body=b"", **kwargs) -> RobotsGate:
    return RobotsGate("TestBot", fetcher=lambda _url: (status, body), **kwargs)


@pytest.fixture(autouse=True)
def _enable_robots(monkeypatch):
    monkeypatch.setattr(config, "RESPECT_ROBOTS", True, raising=False)


class TestStatusHandling:
    def test_2xx_rules_are_applied(self):
        gate = gate_returning(200, ROBOTS)
        assert gate.allowed("https://a.test/public/page")
        assert not gate.allowed("https://a.test/nope/page")

    @pytest.mark.parametrize("status", [401, 403])
    def test_protected_robots_means_whole_site_disallowed(self, status):
        """A site that hides its policy is not inviting you in."""
        gate = gate_returning(status)
        assert not gate.allowed("https://a.test/anything")
        assert "disallowed" in gate.verdict("https://a.test/x").reason

    @pytest.mark.parametrize("status", [404, 410, 400])
    def test_other_4xx_means_no_rules_exist(self, status):
        assert gate_returning(status).allowed("https://a.test/anything")

    def test_5xx_is_permissive_by_default(self):
        """One flaky 502 should not silently halt a legitimate crawl."""
        assert gate_returning(500).allowed("https://a.test/x")

    def test_5xx_can_be_made_strict(self):
        gate = gate_returning(503, on_unavailable="deny")
        assert not gate.allowed("https://a.test/x")

    def test_network_failure_is_treated_as_unavailable(self):
        def boom(_url):
            raise OSError("connection reset")

        gate = RobotsGate("TestBot", fetcher=boom)
        assert gate.allowed("https://a.test/x")  # permissive default

    def test_rejects_a_bad_on_unavailable_value(self):
        with pytest.raises(ValueError):
            RobotsGate("TestBot", on_unavailable="maybe")


class TestDirectives:
    def test_most_specific_user_agent_wins(self):
        gate = gate_returning(200, ROBOTS)
        assert gate.crawl_delay("https://a.test/x") == 5.0

    def test_request_rate_with_a_unit_suffix_is_understood(self):
        """urllib.robotparser silently drops `Request-rate: 1/10s`."""
        body = b"User-agent: TestBot\nRequest-rate: 1/10s\nDisallow:\n"
        assert gate_returning(200, body).crawl_delay("https://a.test/x") == 10.0

    def test_sitemaps_are_exposed(self):
        body = b"Sitemap: https://a.test/sitemap.xml\nUser-agent: *\nDisallow:\n"
        assert gate_returning(200, body).sitemaps("https://a.test/x") == (
            "https://a.test/sitemap.xml",
        )

    def test_disabling_robots_allows_everything(self, monkeypatch):
        monkeypatch.setattr(config, "RESPECT_ROBOTS", False, raising=False)
        assert gate_returning(403).allowed("https://a.test/x")


class TestCaching:
    def test_robots_is_fetched_once_per_origin(self):
        calls = []

        def counting(url):
            calls.append(url)
            return 200, ROBOTS

        gate = RobotsGate("TestBot", fetcher=counting)
        for _ in range(5):
            gate.allowed("https://a.test/page")
        assert len(calls) == 1

    def test_different_origins_are_separate(self):
        calls = []
        gate = RobotsGate("TestBot", fetcher=lambda url: (calls.append(url), (200, ROBOTS))[1])
        gate.allowed("https://a.test/x")
        gate.allowed("https://b.test/x")
        assert len(calls) == 2


class TestFetchPolicy:
    def test_non_http_scheme_is_refused(self):
        policy = FetchPolicy(gate=gate_returning(404))
        assert not policy.check("file:///etc/passwd").allowed

    def test_private_address_is_refused(self, monkeypatch):
        monkeypatch.setattr(config, "ALLOW_PRIVATE_ADDRESSES", False, raising=False)
        policy = FetchPolicy(gate=gate_returning(404))
        verdict = policy.check("http://169.254.169.254/latest/meta-data/")
        assert not verdict.allowed
        assert "private address" in verdict.reason

    def test_denylist(self, monkeypatch):
        monkeypatch.setattr(config, "HOST_DENYLIST", "blocked.test", raising=False)
        policy = FetchPolicy(gate=gate_returning(404))
        assert not policy.check("https://blocked.test/x").allowed
        assert policy.check("https://other.test/x").allowed

    def test_allowlist_excludes_everything_else(self, monkeypatch):
        monkeypatch.setattr(config, "HOST_ALLOWLIST", "allowed.test", raising=False)
        policy = FetchPolicy(gate=gate_returning(404))
        assert policy.check("https://allowed.test/x").allowed
        assert not policy.check("https://other.test/x").allowed

    def test_excessive_crawl_delay_is_refused(self, monkeypatch):
        monkeypatch.setattr(config, "MAX_CRAWL_DELAY", 10.0, raising=False)
        body = b"User-agent: *\nCrawl-delay: 300\nDisallow:\n"
        policy = FetchPolicy(gate=RobotsGate("*", fetcher=lambda _u: (200, body)))
        verdict = policy.check("https://slow.test/x")
        assert not verdict.allowed
        assert "Crawl-delay" in verdict.reason

    def test_enforce_raises_the_right_error(self, monkeypatch):
        policy = FetchPolicy(gate=gate_returning(403))
        with pytest.raises(RobotsDisallowed):
            policy.enforce("https://a.test/x")

        monkeypatch.setattr(config, "HOST_DENYLIST", "b.test", raising=False)
        policy = FetchPolicy(gate=gate_returning(404))
        with pytest.raises(HostNotAllowed):
            policy.enforce("https://b.test/x")
