"""robots.txt fetching, caching, and interpretation.

Status handling follows RFC 9309 §2.3.1, which is stricter than most crawlers
bother to be:

===============  =========================================================
Status           Interpretation
===============  =========================================================
2xx              Parse and apply the rules.
401, 403         Access to the rules themselves is denied, so assume the
                 whole site is disallowed. A site that hides its policy is
                 not inviting you in.
Other 4xx        No rules exist, so everything is allowed.
5xx / network    Undetermined. RFC 9309 says treat as a full disallow;
                 common practice is to proceed. Configurable via
                 ROBOTS_ON_UNAVAILABLE, defaulting to the permissive
                 reading, because a flaky 502 on robots.txt should not
                 silently halt a legitimate crawl.
===============  =========================================================

The gate is consulted on **every redirect hop**, not just the submitted URL.
A redirect from an allowed path onto a disallowed one is the ordinary way a
compliant-looking crawler ends up somewhere it was told not to go.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.robotparser import RobotFileParser

import httpx

from config import config
from nettls import client_context, explain
from observability import get_logger
from urls import origin_of

log = get_logger("compliance.robots")

#: (status, body). ``status`` is None when the request never completed.
RobotsFetcher = Callable[[str], "tuple[Optional[int], bytes]"]

#: ``Request-rate: 1/10s`` — the conventional spelling carries a unit suffix,
#: but urllib.robotparser only accepts two bare integers and silently drops
#: anything else. Rewrite the line to seconds rather than reimplement a parser.
_REQUEST_RATE_RE = re.compile(
    r"^(\s*request-rate\s*:\s*)(\d+)\s*/\s*(\d+)\s*([smhd])\s*$", re.IGNORECASE
)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

#: robots.txt is a text file. Google stops reading at 500 KiB; anything larger
#: is a misconfiguration or a trap.
_MAX_ROBOTS_BYTES = 512 * 1024


def _normalize_request_rate(line: str) -> str:
    match = _REQUEST_RATE_RE.match(line)
    if not match:
        return line
    prefix, requests, period, unit = match.groups()
    return f"{prefix}{requests}/{int(period) * _UNIT_SECONDS[unit.lower()]}"


@dataclass(frozen=True, slots=True)
class RobotsVerdict:
    allowed: bool
    reason: str
    crawl_delay: Optional[float] = None

    def __bool__(self) -> bool:
        return self.allowed


@dataclass(slots=True)
class _Entry:
    parser: Optional[RobotFileParser]
    status: Optional[int]
    fetched_at: float
    #: Set when the fetch failed or was denied, so the verdict is not
    #: re-derived from an empty parser (which would allow everything).
    blanket: Optional[bool] = None
    sitemaps: tuple[str, ...] = field(default_factory=tuple)


class RobotsGate:
    """Per-origin robots.txt cache with a configurable failure mode.

    One instance is shared across the process. robots.txt is fetched at most
    once per origin per TTL, under a per-origin lock, so a burst of concurrent
    workers starting at once cannot stampede a site's robots.txt.
    """

    def __init__(
        self,
        user_agent: Optional[str] = None,
        *,
        fetcher: Optional[RobotsFetcher] = None,
        on_unavailable: Optional[str] = None,
        timeout: Optional[float] = None,
        cache_ttl: Optional[float] = None,
    ) -> None:
        self.user_agent = user_agent or config.robots_agent
        self.on_unavailable = on_unavailable or config.ROBOTS_ON_UNAVAILABLE
        if self.on_unavailable not in {"allow", "deny"}:
            raise ValueError("on_unavailable must be 'allow' or 'deny'")
        self.timeout = timeout if timeout is not None else config.ROBOTS_TIMEOUT
        self.cache_ttl = cache_ttl if cache_ttl is not None else config.ROBOTS_CACHE_TTL
        self._fetch = fetcher or self._default_fetcher
        self._cache: dict[str, _Entry] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def verdict(self, url: str) -> RobotsVerdict:
        if not config.RESPECT_ROBOTS:
            return RobotsVerdict(True, "robots checking disabled by configuration")

        entry = self._entry_for(url)

        if entry.blanket is not None:
            return RobotsVerdict(entry.blanket, self._blanket_reason(entry), None)

        assert entry.parser is not None  # blanket is None only when parsed
        allowed = entry.parser.can_fetch(self.user_agent, url)
        return RobotsVerdict(
            allowed,
            "allowed by robots.txt" if allowed else "a Disallow rule matched",
            self._crawl_delay(entry),
        )

    def allowed(self, url: str) -> bool:
        return self.verdict(url).allowed

    def crawl_delay(self, url: str, default: float = 0.0) -> float:
        if not config.RESPECT_ROBOTS:
            return default
        delay = self._crawl_delay(self._entry_for(url))
        return delay if delay is not None else default

    def sitemaps(self, url: str) -> tuple[str, ...]:
        """Sitemap URLs advertised in robots.txt.

        Worth calling first on any new site: one request can hand you the URL
        inventory that would otherwise take a week of crawling to discover.
        """
        return self._entry_for(url).sitemaps

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _entry_for(self, url: str) -> _Entry:
        origin = origin_of(url)
        if not origin:
            raise ValueError(f"robots check needs an absolute URL, got {url!r}")
        now = time.monotonic()

        with self._lock:
            cached = self._cache.get(origin)
            if cached is not None and now - cached.fetched_at < self.cache_ttl:
                return cached
            origin_lock = self._locks.setdefault(origin, threading.Lock())

        # One fetch per origin: whoever gets the lock fetches, the rest wait
        # and then find the fresh entry in the cache.
        with origin_lock:
            with self._lock:
                cached = self._cache.get(origin)
                if cached is not None and time.monotonic() - cached.fetched_at < self.cache_ttl:
                    return cached

            entry = self._load(origin)

            with self._lock:
                self._cache[origin] = entry
            return entry

    def _load(self, origin: str) -> _Entry:
        robots_url = f"{origin}/robots.txt"
        now = time.monotonic()

        try:
            status, body = self._fetch(robots_url)
        except Exception as exc:
            log.warning("robots.fetch_failed", origin=origin, error=repr(exc))
            status, body = None, b""

        if status is not None and 200 <= status < 300:
            parser = RobotFileParser()
            parser.set_url(robots_url)
            text = body.decode("utf-8", errors="replace")
            parser.parse([_normalize_request_rate(line) for line in text.splitlines()])
            sitemaps = tuple(parser.site_maps() or ())
            log.debug("robots.loaded", origin=origin, sitemaps=len(sitemaps))
            return _Entry(parser, status, now, blanket=None, sitemaps=sitemaps)

        if status in (401, 403):
            log.info("robots.access_denied", origin=origin, status=status)
            return _Entry(None, status, now, blanket=False)

        if status is not None and 400 <= status < 500:
            return _Entry(None, status, now, blanket=True)

        permissive = self.on_unavailable == "allow"
        log.warning("robots.unavailable", origin=origin, status=status, allowing=permissive)
        return _Entry(None, status, now, blanket=permissive)

    def _blanket_reason(self, entry: _Entry) -> str:
        if entry.status in (401, 403):
            return f"robots.txt returned {entry.status}; treating the site as disallowed"
        if entry.status is not None and 400 <= entry.status < 500:
            return f"no robots.txt (HTTP {entry.status})"
        verb = "allowing" if entry.blanket else "denying"
        return f"robots.txt unavailable (status={entry.status}); {verb} by configuration"

    def _crawl_delay(self, entry: _Entry) -> Optional[float]:
        if entry.parser is None:
            return None

        raw = entry.parser.crawl_delay(self.user_agent)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass

        rate = entry.parser.request_rate(self.user_agent)
        if rate is not None and rate.requests > 0:
            return float(rate.seconds) / float(rate.requests)
        return None

    def _default_fetcher(self, robots_url: str) -> tuple[Optional[int], bytes]:
        """Fetch robots.txt with a plain client.

        Deliberately not routed through the pipeline's own fetcher: that one
        consults *this* gate before every request, and a robots fetch that
        needs a robots check cannot terminate.
        """
        headers = {"User-Agent": config.user_agent, "Accept": "text/plain,*/*;q=0.5"}
        try:
            # Same TLS context as the fetcher. If these two disagree, robots.txt
            # can fail its handshake while the page succeeds, and the crawl then
            # runs under "unavailable, allowing by configuration" — a permission
            # nobody granted.
            with httpx.Client(
                timeout=self.timeout, follow_redirects=True, verify=client_context()
            ) as client:
                with client.stream("GET", robots_url, headers=headers) as response:
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) >= _MAX_ROBOTS_BYTES:
                            break
                    return response.status_code, bytes(body[:_MAX_ROBOTS_BYTES])
        except httpx.HTTPError as exc:
            log.warning("robots.fetch_failed", url=robots_url, error=explain(exc))
            return None, b""


#: Process-wide gate. Sharing it is the point — the cache is what keeps a
#: 200-URL job to one robots.txt request per site.
robots_gate = RobotsGate()

__all__ = ["RobotsFetcher", "RobotsGate", "RobotsVerdict", "robots_gate"]
