"""Recognising anti-bot walls.

The purpose of this module is *recognition*, not evasion.

It exists because a Cloudflare interstitial returns **HTTP 200**. Without this
check the pipeline treats "Checking your browser before accessing…" as a
successful fetch and spends a 300-second Ollama call extracting product fields
from a challenge page — the single most expensive no-op in the system.

When a challenge answers instead of the application, that is a stated refusal
to serve automated clients. Detecting it precisely matters because the correct
response is the opposite of the response to a transient error: a 503 you retry,
a challenge you stop and route around. So a detected block raises
:class:`~errors.BlockedError`, marks the host permanently open on the circuit
breaker, and prints the operator guidance below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

#: Headers set by the protection layer itself. Conclusive whatever the status.
_HEADER_SIGNALS: tuple[tuple[str, Optional[str], str], ...] = (
    ("cf-mitigated", "challenge", "cloudflare challenge"),
    ("x-datadome", None, "datadome"),
    ("x-iinfo", None, "imperva/incapsula"),
    ("x-sucuri-id", None, "sucuri"),
    ("x-akamai-bot-manager", None, "akamai bot manager"),
    ("server", "awselb/2.0-challenge", "aws waf challenge"),
)

#: Body markers. Only believed alongside a refusal status *or* an interstitial
#: title, because "captcha" appears legitimately on plenty of ordinary pages —
#: a security blog post is not a wall.
_BODY_SIGNALS: tuple[tuple[str, str], ...] = (
    ("cf-browser-verification", "cloudflare browser check"),
    ("cf_chl_opt", "cloudflare challenge"),
    ("just a moment...", "cloudflare interstitial"),
    ("checking your browser before accessing", "browser check interstitial"),
    ("attention required! | cloudflare", "cloudflare block page"),
    ("_incapsula_resource", "imperva/incapsula"),
    ("request unsuccessful. incapsula incident", "imperva/incapsula"),
    ("/_px/", "perimeterx"),
    ("px-captcha", "perimeterx captcha"),
    ("g-recaptcha", "recaptcha challenge"),
    ("h-captcha", "hcaptcha challenge"),
    ("please enable js and disable any ad blocker", "js challenge"),
    ("enable javascript and cookies to continue", "js challenge"),
    ("you have been blocked", "generic block page"),
    ("unusual traffic from your computer network", "traffic analysis block"),
    ("are you a robot", "bot challenge"),
    ("verifying you are human", "human verification"),
    ("user validation required", "captcha interstitial"),
    ("captcha_resp_txt", "captcha interstitial"),
)

#: Markers strong enough to believe at HTTP 200, because no ordinary page has
#: them. This is what catches the Cloudflare-200 case.
_STANDALONE_SIGNALS: frozenset[str] = frozenset(
    {
        "cf-browser-verification",
        "cf_chl_opt",
        "just a moment...",
        "checking your browser before accessing",
        "_incapsula_resource",
        "px-captcha",
        "verifying you are human",
        # Appliance-style challenges that answer at 200 on *every* path, this
        # project's own /robots.txt included. A wall serving robots.txt is the
        # dangerous shape: the parser reads a challenge page, finds no rules,
        # and reports the site as permitting everything. Both markers are
        # specific enough to believe alone -- "captcha" on its own is not, and
        # is deliberately absent, because ordinary pages discuss captchas.
        "user validation required",
        "captcha_resp_txt",
    }
)

_REFUSAL_STATUSES = frozenset({401, 403, 405, 406, 409, 418, 429, 503})

#: A challenge page is small. A 4 MB document that happens to mention
#: "access denied" is a document about access denial, not a wall.
_MAX_CHALLENGE_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class BlockSignal:
    """Evidence that an anti-bot control answered."""

    name: str
    #: "header" | "body" — header evidence is conclusive on its own.
    source: str
    confidence: float

    def __str__(self) -> str:
        return f"{self.name} (via {self.source})"


def detect_block(
    status: int,
    headers: Mapping[str, str],
    body: bytes,
    *,
    text: Optional[str] = None,
) -> Optional[BlockSignal]:
    """Return a :class:`BlockSignal` if this response is a wall, else ``None``.

    Conservative by construction. A false positive stops a legitimate job; a
    false negative merely produces a normal retry. So the evidence must be
    either a protection vendor's own header, or a marker in a short body that
    is paired with a refusal status or is standalone-conclusive.
    """
    lowered_headers = {k.lower(): (v or "") for k, v in headers.items()}
    for header, expected, name in _HEADER_SIGNALS:
        value = lowered_headers.get(header)
        if value is None:
            continue
        if expected is None or expected in value.lower():
            return BlockSignal(name, "header", 1.0)

    if len(body) > _MAX_CHALLENGE_BYTES:
        return None

    # Decoding is deferred until the cheap checks have failed, so ordinary
    # pages never pay for it.
    haystack = (text if text is not None else body.decode("utf-8", errors="ignore")).lower()
    refusal = status in _REFUSAL_STATUSES

    for marker, name in _BODY_SIGNALS:
        if marker not in haystack:
            continue
        if refusal or marker in _STANDALONE_SIGNALS:
            return BlockSignal(name, "body", 1.0 if marker in _STANDALONE_SIGNALS else 0.9)

    return None


def block_guidance(host: str) -> str:
    """The operator-facing next steps when a host has said no.

    Surfaced rather than buried, because the decision it describes is a project
    decision, not a code change.
    """
    return (
        f"{host} is refusing automated access.\n"
        "Supported ways forward, in order of preference:\n"
        "  1. Official API or paid data product — usually cheaper than the\n"
        "     engineering time already spent getting this far.\n"
        "  2. Bulk export or open-data portal (data.gov, EU Open Data, national\n"
        "     equivalents; regulatory filings such as SEC EDGAR).\n"
        "  3. A licensed data vendor.\n"
        "  4. Ask. Email the site owner — access is granted far more often than\n"
        "     people expect.\n"
        "  5. A mirror: Common Crawl, the Wayback Machine CDX API, Wikidata,\n"
        "     OpenStreetMap, or an aggregator that already licensed the data.\n"
        "  6. If none of those exist, this dataset is not available to you.\n"
        "Do not respond by rotating addresses, spoofing fingerprints, or solving\n"
        "the challenge: those defeat a control whose purpose is to refuse you."
    )


__all__ = ["BlockSignal", "block_guidance", "detect_block"]
