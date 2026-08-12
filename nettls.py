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
"""

from __future__ import annotations

import ssl
from functools import lru_cache

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


def client_context() -> ssl.SSLContext:
    """The TLS context every outbound client should be constructed with."""
    return _context(bool(config.TLS_ALLOW_LEGACY_RENEGOTIATION))


def explain(exc: BaseException) -> str:
    """Render a transport error, expanding the one that needs explaining.

    Certificate and hostname failures are already legible. This one is not:
    nothing in "unsafe legacy renegotiation disabled" tells you there is a
    setting that fixes it, or that it is the *server* that is out of date.
    """
    text = f"{type(exc).__name__}: {exc}"
    if LEGACY_RENEGOTIATION_MARKER in str(exc) and not config.TLS_ALLOW_LEGACY_RENEGOTIATION:
        text += (
            " — the server does not support RFC 5746 secure renegotiation, which"
            " OpenSSL refuses by default. Set TLS_ALLOW_LEGACY_RENEGOTIATION=true"
            " to connect anyway (this weakens transport security; do it only for"
            " sites you trust)."
        )
    return text


__all__ = [
    "LEGACY_RENEGOTIATION_MARKER",
    "OP_LEGACY_SERVER_CONNECT",
    "client_context",
    "explain",
]
