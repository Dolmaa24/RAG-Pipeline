"""URL canonicalisation and host helpers.

Without canonicalisation ``?utm_source=twitter`` makes a page a different
record from the same page reached from an email, and the unique index in Mongo
happily stores both. Canonicalising first is what makes the index mean
"one row per document" rather than "one row per link someone happened to click".
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

#: Parameters that identify a campaign or a session, never a document. Removing
#: them is safe; removing anything else risks changing which page you get.
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_source_platform", "utm_creative_format", "utm_marketing_tactic",
        "gclid", "gclsrc", "dclid", "gbraid", "wbraid", "fbclid", "msclkid",
        "twclid", "igshid", "ttclid", "yclid", "rb_clickid", "s_kwcid",
        "mc_cid", "mc_eid", "_hsenc", "_hsmi", "hsCtaTracking",
        "vero_conv", "vero_id", "oly_anon_id", "oly_enc_id",
        "ref_src", "ref_url", "spm", "scm", "share_id",
        "mkt_tok", "trk", "trkCampaign", "sc_campaign", "sc_channel",
        "pk_campaign", "pk_kwd", "piwik_campaign", "matomo_campaign",
        "__s", "wickedid", "epik", "cmpid", "campaign_id",
    }
)

_DEFAULT_PORTS = {"http": "80", "https": "443", "ftp": "21", "ws": "80", "wss": "443"}
_DEFAULT_INDEX = re.compile(r"/(index|default)\.(html?|php|aspx?|jsp)$", re.IGNORECASE)

#: Hosts that resolve to the machine running the pipeline or to a cloud
#: metadata endpoint. Fetching these on behalf of a user-supplied URL is SSRF.
_METADATA_HOSTS = frozenset({"metadata.google.internal", "metadata.goog", "instance-data"})


def host_of(url: str) -> str:
    """Lower-cased hostname with the port and any userinfo removed."""
    try:
        netloc = urlsplit(url).netloc
    except ValueError:
        return ""
    host = netloc.rsplit("@", 1)[-1]
    if host.startswith("["):  # IPv6 literal
        return host.partition("]")[0].lstrip("[").lower()
    return host.split(":")[0].lower()


def registrable_host(url_or_host: str) -> str:
    """Host with a leading ``www.`` stripped.

    Not a public-suffix lookup — that needs a bundled PSL and is overkill for
    grouping rate limits and selector specs, where ``www.`` is the only prefix
    that reliably means "same site".
    """
    host = url_or_host if "://" not in url_or_host else host_of(url_or_host)
    host = host.split("@")[-1].split(":")[0].lower()
    return host[4:] if host.startswith("www.") else host


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), "", "", ""))


def is_http_url(url: str) -> bool:
    return urlsplit(url).scheme.lower() in ("http", "https")


def is_private_address(host: str) -> bool:
    """True for loopback, link-local, private, and cloud-metadata hosts.

    Only catches literal addresses — a name that *resolves* to 169.254.169.254
    is caught at connect time by the fetcher's policy check, not here.
    """
    if not host:
        return True
    if host.lower() in _METADATA_HOSTS or host.lower().endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host.lower() == "localhost"
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def canonicalize(
    url: str,
    *,
    strip_tracking: bool = True,
    strip_fragment: bool = True,
    sort_query: bool = True,
) -> str:
    """Return a stable, comparable form of ``url``.

    Applies only transformations that cannot change which resource is served:
    lower-casing the scheme and host, dropping the default port, removing an
    empty query or fragment, collapsing ``/index.html`` to ``/``, and dropping
    known tracking parameters. Path case and trailing slashes are **kept** —
    on plenty of servers those do change the response.
    """
    url = (url or "").strip()
    if not url:
        return ""

    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    netloc = parts.netloc

    userinfo, _, hostport = netloc.rpartition("@")
    host, _, port = hostport.partition(":") if not hostport.startswith("[") else (hostport, "", "")
    host = host.lower().rstrip(".")
    if port and port == _DEFAULT_PORTS.get(scheme):
        port = ""
    netloc = host + (f":{port}" if port else "")
    if userinfo:
        netloc = f"{userinfo}@{netloc}"

    path = parts.path or "/"
    # Normalize backslashes to forward slashes before canonicalizing
    path = unquote(path).replace("\\", "/")
    # Re-encode so %2f and %2F, and encoded-but-safe characters, agree.
    path = quote(path, safe="/:@!$&'()*+,;=~-._")
    path = _DEFAULT_INDEX.sub("/", path)
    if not path.startswith("/"):
        path = "/" + path

    query = parts.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        if strip_tracking:
            pairs = [(k, v) for k, v in pairs if k.lower() not in TRACKING_PARAMS]
        if sort_query:
            pairs.sort()
        query = urlencode(pairs, doseq=True)

    fragment = "" if strip_fragment else parts.fragment
    return urlunsplit((scheme, netloc, path, query, fragment))


def resolve(base: str, href: str) -> str:
    """Absolute URL for ``href`` seen on the page at ``base``."""
    try:
        return urljoin(base, (href or "").strip())
    except ValueError:
        return ""


def same_site(a: str, b: str) -> bool:
    return registrable_host(a) == registrable_host(b)


def url_filename(url: str) -> str:
    """Last path segment, or "" when the URL is a bare directory."""
    return urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]


__all__ = [
    "TRACKING_PARAMS",
    "canonicalize",
    "host_of",
    "is_http_url",
    "is_private_address",
    "origin_of",
    "registrable_host",
    "resolve",
    "same_site",
    "url_filename",
]
