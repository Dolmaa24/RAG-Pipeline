"""What the pipeline is allowed to request at all.

Three separate questions, checked before a socket is opened:

1. **Is this a fetchable scheme?** ``file://`` and ``gopher://`` are not, and a
   URL submitted over an HTTP API must not be able to read local files.
2. **Does this address belong to the operator?** A user-supplied URL that
   resolves to 127.0.0.1 or 169.254.169.254 turns the worker into a proxy for
   whatever is on the private network. That is SSRF, and it is the reason this
   check exists on every hop rather than just the first.
3. **Is this host in scope?** An explicit allowlist for jobs that should never
   wander, a denylist for hosts already known to refuse.

robots.txt is the *site's* answer and lives in :mod:`.robots`; this module is
the *operator's* answer.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

from config import config
from errors import HostNotAllowed, RobotsDisallowed, SchemeNotAllowed
from observability import get_logger
from urls import host_of, is_private_address, registrable_host

from .robots import RobotsGate, RobotsVerdict, robots_gate

log = get_logger("compliance.policy")


@dataclass(frozen=True, slots=True)
class PolicyVerdict:
    allowed: bool
    reason: str
    crawl_delay: Optional[float] = None

    def __bool__(self) -> bool:
        return self.allowed


class FetchPolicy:
    """Operator-side access rules, plus the robots.txt gate.

    :meth:`check` is called for the submitted URL *and* for the target of every
    redirect. That is the whole point: a 302 from an allowed path onto a
    disallowed one is invisible to any crawler that checks robots.txt once.
    """

    def __init__(self, gate: Optional[RobotsGate] = None) -> None:
        self.gate = gate or robots_gate

    def check(self, url: str, *, hop: int = 0) -> PolicyVerdict:
        """Full verdict for one URL. Raises nothing; see :meth:`enforce`."""
        parts = urlsplit(url)
        scheme = parts.scheme.lower()

        if scheme not in config.allowed_schemes:
            return PolicyVerdict(False, f"scheme {scheme!r} is not fetchable")

        host = host_of(url)
        if not host:
            return PolicyVerdict(False, "URL has no host")

        if not config.ALLOW_PRIVATE_ADDRESSES:
            private, detail = self._resolves_private(host)
            if private:
                return PolicyVerdict(False, f"refusing to fetch a private address ({detail})")

        registrable = registrable_host(host)
        denylist = config.host_denylist
        if denylist and (host in denylist or registrable in denylist):
            return PolicyVerdict(False, "host is on the denylist")

        allowlist = config.host_allowlist
        if allowlist and host not in allowlist and registrable not in allowlist:
            return PolicyVerdict(False, "host is not on the allowlist")

        verdict: RobotsVerdict = self.gate.verdict(url)
        if not verdict.allowed:
            log.info("policy.robots_denied", url=url, hop=hop, reason=verdict.reason)
            return PolicyVerdict(False, verdict.reason, verdict.crawl_delay)

        delay = verdict.crawl_delay
        if delay is not None and delay > config.MAX_CRAWL_DELAY:
            # A site asking for a 5-minute gap between pages is asking you not
            # to crawl it. Say so, rather than sleeping for five minutes.
            return PolicyVerdict(
                False,
                f"Crawl-delay of {delay:.0f}s exceeds the {config.MAX_CRAWL_DELAY:.0f}s "
                "we are willing to honour; treat this site as off-limits",
                delay,
            )

        return PolicyVerdict(True, verdict.reason, delay)

    def enforce(self, url: str, *, hop: int = 0) -> PolicyVerdict:
        """Like :meth:`check`, but raises the matching compliance error."""
        verdict = self.check(url, hop=hop)
        if verdict.allowed:
            return verdict

        scheme = urlsplit(url).scheme.lower()
        if scheme not in config.allowed_schemes:
            raise SchemeNotAllowed(url, scheme)
        if "robots" in verdict.reason or "Disallow" in verdict.reason or "Crawl-delay" in verdict.reason:
            raise RobotsDisallowed(url, verdict.reason)
        raise HostNotAllowed(host_of(url) or url, verdict.reason)

    @staticmethod
    def _resolves_private(host: str) -> tuple[bool, str]:
        """Block literal private addresses and names that resolve to them.

        DNS is resolved here and then again by the HTTP client, so a hostile
        server could in principle answer differently the second time
        (a DNS-rebinding attack). Closing that hole properly means pinning the
        resolved address into the connection, which httpx does not expose;
        this check stops the ordinary case, which is what SSRF attempts
        actually look like.
        """
        if is_private_address(host):
            return True, host
        try:
            infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror:
            return False, ""  # unresolvable: let the fetch fail normally
        for info in infos:
            address = info[4][0]
            if is_private_address(address):
                return True, f"{host} resolves to {address}"
        return False, ""


policy = FetchPolicy()

__all__ = ["FetchPolicy", "PolicyVerdict", "policy"]
