"""One shared TLS context for every outbound request.

Three places in this pipeline open sockets to the wider internet — the fetcher,
the robots.txt reader and the type probe — and all three must agree about what
a valid handshake looks like. If they disagree you get the worst possible bug:
robots.txt reports "unavailable, allowing by configuration" because *its*
handshake failed, and the crawl then proceeds on an assumption nobody made.

The one knob here is legacy renegotiation. OpenSSL 3 refuses to talk to servers
that never implemented RFC 5746, and httpx uses OpenSSL, so such a site fails
with `UNSAFE_LEGACY_RENEGOTIATION_DISABLED` — while curl on macOS, which links
LibreSSL, fetches it happily. That gap is confusing enough to be worth naming
in the error message rather than leaving as a raw SSL string.

That knob is granted **per host**. One out-of-date university site is a reason
to lower the bar for that site, not for every host the crawler will ever reach,
and the global flag left on after the one crawl that needed it is exactly how
that happens. ``TLS_ALLOW_LEGACY_RENEGOTIATION`` still exists and still applies
everywhere, for when a whole deployment sits behind such servers.

Relaxing renegotiation is *not* relaxing verification. The certificate chain,
its expiry and the hostname are checked identically either way; what changes is
only whether a peer missing the RFC 5746 extension is refused at the handshake.
"""

from __future__ import annotations

import ssl
from functools import lru_cache
from urllib.parse import urlsplit

import certifi

from config import config

#: Not exposed by the `ssl` module. Tells OpenSSL to permit a handshake with a
#: peer that does not advertise RFC 5746 support.
OP_LEGACY_SERVER_CONNECT = 0x4

#: The substring OpenSSL puts in the exception for exactly this failure.
LEGACY_RENEGOTIATION_MARKER = "UNSAFE_LEGACY_RENEGOTIATION_DISABLED"


@lru_cache(maxsize=4)
def _context(allow_legacy: bool) -> ssl.SSLContext:
    """Build a verifying context. Cached — building one is not cheap.

    Keyed on the setting rather than reading it inside, so that flipping the
    flag in a test produces a different context instead of a stale hit.
    """
    context = ssl.create_default_context(cafile=certifi.where())
    if allow_legacy:
        context.options |= OP_LEGACY_SERVER_CONNECT
    return context


def host_of(url_or_host: str) -> str:
    """The hostname, whether given a URL or already a bare host."""
    if "//" in url_or_host:
        return (urlsplit(url_or_host).hostname or "").lower()
    return url_or_host.strip().lower().rstrip(".")


def legacy_allowed_for(url_or_host: str | None) -> bool:
    """Whether this host may use legacy renegotiation.

    Matching is exact or a parent-domain suffix, so ``iitm.ac.in`` admits
    ``gate2027.iitm.ac.in``. The dot in the suffix test is what keeps it from
    also admitting ``evil-iitm.ac.in``, which shares the ending but not the
    domain — the whole reason to write the check rather than use ``in``.
    """
    if config.TLS_ALLOW_LEGACY_RENEGOTIATION:
        return True
    allowed = config.tls_legacy_hosts
    if not allowed or not url_or_host:
        return False

    host = host_of(url_or_host)
    return any(host == entry or host.endswith(f".{entry}") for entry in allowed)


def client_context(url_or_host: str | None = None) -> ssl.SSLContext:
    """The TLS context every outbound client should be constructed with.

    Called without a host it returns the strict context unless the global flag
    is set, which is the safe direction: a caller that has not said where it is
    going does not get the exception.
    """
    return _context(legacy_allowed_for(url_or_host))


def explain(exc: BaseException, url_or_host: str | None = None) -> str:
    """Render a transport error, expanding the one that needs explaining.

    Certificate and hostname failures are already legible. This one is not:
    nothing in "unsafe legacy renegotiation disabled" tells you there is a
    setting that fixes it, or that it is the *server* that is out of date.
    """
    text = f"{type(exc).__name__}: {exc}"
    if LEGACY_RENEGOTIATION_MARKER in str(exc) and not legacy_allowed_for(url_or_host):
        host = host_of(url_or_host) if url_or_host else ""
        # Name the per-host setting first and with the host already filled in.
        # The global flag is the bigger hammer, so it is offered second.
        remedy = (
            f" Add it to TLS_LEGACY_HOSTS (e.g. TLS_LEGACY_HOSTS={host}) to allow"
            " this host only, or set TLS_ALLOW_LEGACY_RENEGOTIATION=true for every"
            " host."
            if host
            else " Set TLS_ALLOW_LEGACY_RENEGOTIATION=true to connect anyway."
        )
        text += (
            " — the server does not support RFC 5746 secure renegotiation, which"
            " OpenSSL refuses by default. Certificate verification is unaffected"
            " either way; only the handshake is."
            + remedy
        )
    return text


__all__ = [
    "LEGACY_RENEGOTIATION_MARKER",
    "OP_LEGACY_SERVER_CONNECT",
    "client_context",
    "explain",
    "host_of",
    "legacy_allowed_for",
]
