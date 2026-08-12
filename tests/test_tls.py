"""The TLS context every outbound client shares.

The bug this covers: a site whose TLS stack predates RFC 5746 fails the
handshake under OpenSSL 3 while loading fine in curl. Worse, the *robots.txt*
fetch failed the same way, so the gate reported "unavailable" and the crawl
proceeded under the permissive default — an assumption presented as permission.
"""

from __future__ import annotations

import ssl

import pytest

import nettls
from config import config
from nettls import OP_LEGACY_SERVER_CONNECT, client_context, explain


@pytest.fixture(autouse=True)
def _clear_context_cache():
    """The context is cached, so a flag flip must not hit a stale entry."""
    nettls._context.cache_clear()
    yield
    nettls._context.cache_clear()


class TestContext:
    def test_certificates_are_verified_either_way(self, monkeypatch):
        """The legacy flag must never become an accidental `verify=False`."""
        for allow in (False, True):
            monkeypatch.setattr(config, "TLS_ALLOW_LEGACY_RENEGOTIATION", allow, raising=False)
            nettls._context.cache_clear()
            context = client_context()
            assert context.verify_mode is ssl.CERT_REQUIRED
            assert context.check_hostname is True

    def test_legacy_renegotiation_is_off_by_default(self, monkeypatch):
        monkeypatch.setattr(config, "TLS_ALLOW_LEGACY_RENEGOTIATION", False, raising=False)
        assert not client_context().options & OP_LEGACY_SERVER_CONNECT

    def test_the_flag_turns_legacy_renegotiation_on(self, monkeypatch):
        monkeypatch.setattr(config, "TLS_ALLOW_LEGACY_RENEGOTIATION", True, raising=False)
        assert client_context().options & OP_LEGACY_SERVER_CONNECT

    def test_the_context_is_shared_rather_than_rebuilt_per_request(self, monkeypatch):
        """A fresh context per fetch is slow and defeats session resumption."""
        monkeypatch.setattr(config, "TLS_ALLOW_LEGACY_RENEGOTIATION", False, raising=False)
        assert client_context() is client_context()


class TestExplain:
    def test_the_legacy_failure_names_the_setting_that_fixes_it(self, monkeypatch):
        monkeypatch.setattr(config, "TLS_ALLOW_LEGACY_RENEGOTIATION", False, raising=False)
        message = explain(
            ConnectionError("[SSL: UNSAFE_LEGACY_RENEGOTIATION_DISABLED] unsafe legacy renegotiation")
        )
        assert "TLS_ALLOW_LEGACY_RENEGOTIATION" in message
        assert "RFC 5746" in message

    def test_no_advice_once_the_flag_is_already_on(self, monkeypatch):
        """Suggesting a setting that is already set sends people in circles."""
        monkeypatch.setattr(config, "TLS_ALLOW_LEGACY_RENEGOTIATION", True, raising=False)
        message = explain(ConnectionError("[SSL: UNSAFE_LEGACY_RENEGOTIATION_DISABLED] nope"))
        assert "TLS_ALLOW_LEGACY_RENEGOTIATION" not in message

    def test_other_errors_pass_through_unchanged(self):
        message = explain(ConnectionError("connection refused"))
        assert message == "ConnectionError: connection refused"
        assert "RFC 5746" not in message
