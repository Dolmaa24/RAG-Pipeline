"""The shared TLS context, and who is allowed to relax it.

The consequential test here is the one about ``evil-iitm.ac.in``. An allowlist
matched with a plain suffix check hands the exception to anyone who registers a
domain ending in the trusted one, which is the whole reason the match is written
out rather than expressed as ``endswith(entry)``.
"""

from __future__ import annotations

import ssl

import pytest

from config import config
from nettls import (
    LEGACY_RENEGOTIATION_MARKER,
    OP_LEGACY_SERVER_CONNECT,
    client_context,
    explain,
    host_of,
    legacy_allowed_for,
)


@pytest.fixture(autouse=True)
def strict_by_default(monkeypatch):
    """Neither escape hatch is open unless a test opens it."""
    monkeypatch.setattr(config, "TLS_ALLOW_LEGACY_RENEGOTIATION", False)
    monkeypatch.setattr(config, "TLS_LEGACY_HOSTS", "")


@pytest.mark.parametrize(
    "given, expected",
    [
        ("https://gate2027.iitm.ac.in/exam_papers", "gate2027.iitm.ac.in"),
        ("http://EXAMPLE.com:8080/a/b", "example.com"),
        ("gate2027.iitm.ac.in", "gate2027.iitm.ac.in"),
        ("IITM.ac.in", "iitm.ac.in"),
        ("iitm.ac.in.", "iitm.ac.in"),
    ],
)
def test_the_host_is_found_in_a_url_or_taken_as_one(given, expected):
    assert host_of(given) == expected


def test_a_listed_host_is_allowed(monkeypatch):
    monkeypatch.setattr(config, "TLS_LEGACY_HOSTS", "iitm.ac.in")
    assert legacy_allowed_for("https://iitm.ac.in/x")


def test_a_bare_domain_covers_its_subdomains(monkeypatch):
    monkeypatch.setattr(config, "TLS_LEGACY_HOSTS", "iitm.ac.in")
    assert legacy_allowed_for("https://gate2027.iitm.ac.in/exam_papers_and_syllabus")


def test_a_domain_that_merely_ends_the_same_way_is_not_covered(monkeypatch):
    """The point of the dot. Anyone can register a name ending in a trusted one."""
    monkeypatch.setattr(config, "TLS_LEGACY_HOSTS", "iitm.ac.in")
    assert not legacy_allowed_for("https://evil-iitm.ac.in/")
    assert not legacy_allowed_for("https://notiitm.ac.in/")


def test_an_unlisted_host_is_not_allowed(monkeypatch):
    monkeypatch.setattr(config, "TLS_LEGACY_HOSTS", "iitm.ac.in")
    assert not legacy_allowed_for("https://example.com/")


def test_the_list_tolerates_spacing_case_and_leading_dots(monkeypatch):
    monkeypatch.setattr(config, "TLS_LEGACY_HOSTS", " .IITM.ac.in , example.org ")
    assert legacy_allowed_for("https://gate2027.iitm.ac.in/")
    assert legacy_allowed_for("https://example.org/")


def test_a_caller_that_names_no_host_gets_the_strict_context(monkeypatch):
    """The safe direction: not saying where you are going earns no exception."""
    monkeypatch.setattr(config, "TLS_LEGACY_HOSTS", "iitm.ac.in")
    assert not legacy_allowed_for(None)


def test_the_global_flag_still_covers_everything(monkeypatch):
    monkeypatch.setattr(config, "TLS_ALLOW_LEGACY_RENEGOTIATION", True)
    assert legacy_allowed_for("https://anything.example/")
    assert legacy_allowed_for(None)


def test_the_strict_context_does_not_carry_the_legacy_option():
    assert not client_context("https://example.com/").options & OP_LEGACY_SERVER_CONNECT


def test_the_permitted_context_carries_it(monkeypatch):
    monkeypatch.setattr(config, "TLS_LEGACY_HOSTS", "iitm.ac.in")
    context = client_context("https://gate2027.iitm.ac.in/")
    assert context.options & OP_LEGACY_SERVER_CONNECT


def test_verification_is_never_what_is_relaxed(monkeypatch):
    """Relaxing renegotiation must not quietly relax who we will talk to."""
    monkeypatch.setattr(config, "TLS_LEGACY_HOSTS", "iitm.ac.in")
    for context in (
        client_context("https://example.com/"),
        client_context("https://gate2027.iitm.ac.in/"),
    ):
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True


def test_two_hosts_on_the_same_footing_share_one_context(monkeypatch):
    """Building a context is not cheap, and the cache is keyed on the decision."""
    monkeypatch.setattr(config, "TLS_LEGACY_HOSTS", "iitm.ac.in,example.org")
    assert client_context("https://a.iitm.ac.in/") is client_context("https://example.org/")


def _renegotiation_error() -> Exception:
    return ssl.SSLError(f"[SSL: {LEGACY_RENEGOTIATION_MARKER}] unsafe legacy renegotiation disabled")


def test_the_remedy_names_the_host_it_applies_to():
    text = explain(_renegotiation_error(), "https://gate2027.iitm.ac.in/exam_papers")
    assert "TLS_LEGACY_HOSTS=gate2027.iitm.ac.in" in text
    # The narrow fix is offered before the global one.
    assert text.index("TLS_LEGACY_HOSTS") < text.index("TLS_ALLOW_LEGACY_RENEGOTIATION")


def test_the_message_says_verification_is_not_the_thing_being_relaxed():
    text = explain(_renegotiation_error(), "https://gate2027.iitm.ac.in/")
    assert "Certificate verification is unaffected" in text


def test_no_remedy_is_offered_once_the_host_is_already_allowed(monkeypatch):
    """A failure here is some other handshake problem, and the old advice misleads."""
    monkeypatch.setattr(config, "TLS_LEGACY_HOSTS", "iitm.ac.in")
    text = explain(_renegotiation_error(), "https://gate2027.iitm.ac.in/")
    assert "TLS_LEGACY_HOSTS" not in text


def test_other_transport_errors_are_passed_through_unexplained():
    text = explain(ssl.SSLCertVerificationError("certificate has expired"), "https://example.com/")
    assert "certificate has expired" in text
    assert "TLS_LEGACY_HOSTS" not in text
