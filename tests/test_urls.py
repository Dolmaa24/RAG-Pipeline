from __future__ import annotations

import pytest

from urls import (
    canonicalize,
    host_of,
    is_private_address,
    registrable_host,
    resolve,
    same_site,
    url_filename,
)


class TestCanonicalize:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("HTTPS://Example.COM/Path", "https://example.com/Path"),
            ("https://example.com:443/a", "https://example.com/a"),
            ("http://example.com:80/a", "http://example.com/a"),
            ("https://example.com", "https://example.com/"),
            ("https://example.com/index.html", "https://example.com/"),
            ("https://example.com/dir/index.php", "https://example.com/dir/"),
            ("https://example.com/a#section", "https://example.com/a"),
            ("https://example.com/a?b=2&a=1", "https://example.com/a?a=1&b=2"),
        ],
    )
    def test_normalizes(self, raw, expected):
        assert canonicalize(raw) == expected

    def test_strips_tracking_parameters(self):
        url = "https://example.com/p?utm_source=x&fbclid=y&id=7&gclid=z"
        assert canonicalize(url) == "https://example.com/p?id=7"

    def test_keeps_meaningful_parameters(self):
        assert canonicalize("https://example.com/s?q=heron") == "https://example.com/s?q=heron"

    def test_path_case_is_preserved(self):
        """Plenty of servers serve different documents for different path case."""
        assert canonicalize("https://example.com/CaseSensitive") != canonicalize(
            "https://example.com/casesensitive"
        )

    def test_trailing_slash_is_preserved(self):
        assert canonicalize("https://example.com/a/") == "https://example.com/a/"
        assert canonicalize("https://example.com/a") == "https://example.com/a"

    def test_empty_input(self):
        assert canonicalize("") == ""

    def test_two_links_to_one_document_agree(self):
        from_email = "https://www.Example.com/story?utm_campaign=news&id=3#top"
        from_nav = "https://www.example.com:443/story?id=3"
        assert canonicalize(from_email) == canonicalize(from_nav)


class TestHosts:
    def test_host_of_strips_port_and_userinfo(self):
        assert host_of("https://user:pw@Example.com:8443/a") == "example.com"

    def test_host_of_handles_ipv6(self):
        assert host_of("http://[2001:db8::1]:8080/a") == "2001:db8::1"

    def test_registrable_strips_www(self):
        assert registrable_host("https://www.bbc.co.uk/news") == "bbc.co.uk"
        assert registrable_host("news.bbc.co.uk") == "news.bbc.co.uk"

    def test_same_site(self):
        assert same_site("https://www.a.com/x", "https://a.com/y")
        assert not same_site("https://a.com", "https://b.com")


class TestPrivateAddresses:
    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "localhost", "10.0.0.5", "192.168.1.1", "172.16.0.1",
         "169.254.169.254", "metadata.google.internal", "0.0.0.0", "::1"],
    )
    def test_blocks_private(self, host):
        assert is_private_address(host)

    @pytest.mark.parametrize("host", ["example.com", "8.8.8.8", "93.184.216.34"])
    def test_allows_public(self, host):
        assert not is_private_address(host)

    def test_empty_is_treated_as_private(self):
        """Fail closed: an unparseable host must not be assumed safe."""
        assert is_private_address("")


class TestResolve:
    def test_relative(self):
        assert resolve("https://a.com/dir/page", "../img.png") == "https://a.com/img.png"

    def test_absolute_passthrough(self):
        assert resolve("https://a.com/x", "https://b.com/y") == "https://b.com/y"

    def test_protocol_relative(self):
        assert resolve("https://a.com/x", "//cdn.com/y") == "https://cdn.com/y"


def test_url_filename():
    assert url_filename("https://a.com/dir/report.pdf") == "report.pdf"
    assert url_filename("https://a.com/dir/") == "dir"
    assert url_filename("https://a.com/") == ""
