"""Self-imposed rate limiting, per host.

Two principles:

*Limits are per host, not global.* Sixteen parallel requests spread over
sixteen domains is polite. Sixteen against one small site is a load test they
did not ask for — and with the io queue running ``-c 16`` that is exactly what
you would otherwise get.

*Rate-limit yourself before the server has to.* The token bucket is the floor;
:meth:`HostRateLimiter.observe_headers` then reads the server's own
``RateLimit-*`` headers and slows down *before* the 429, which is the
difference between a job that finishes and one that gets the IP banned.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Iterator, Mapping, Optional

from config import config
from observability import get_logger
from urls import registrable_host

log = get_logger("fetch.ratelimit")


class TokenBucket:
    """Sustained ``rate``/second with a burst allowance.

    Uses :func:`time.monotonic`, so an NTP correction or a daylight-saving
    change cannot make the limiter think it has a year's worth of tokens.
    """

    __slots__ = ("_lock", "_tokens", "_updated", "capacity", "rate")

    def __init__(self, rate: float, burst: Optional[float] = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.rate = rate
        self.capacity = float(burst if burst is not None else max(rate, 1.0))
        self._tokens = self.capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._updated = now

    def try_take(self, amount: float = 1.0) -> bool:
        """Take tokens if available. Never blocks."""
        with self._lock:
            self._refill(time.monotonic())
            if self._tokens >= amount:
                self._tokens -= amount
                return True
            return False

    def take(self, amount: float = 1.0, *, timeout: Optional[float] = None) -> float:
        """Block until ``amount`` tokens are available. Returns seconds waited.

        Sleeps outside the lock, so a worker waiting on tokens does not block
        another worker that could have proceeded.
        """
        if amount > self.capacity:
            raise ValueError(f"cannot take {amount} tokens from a bucket of {self.capacity}")

        deadline = None if timeout is None else time.monotonic() + timeout
        waited = 0.0
        while True:
            with self._lock:
                self._refill(time.monotonic())
                if self._tokens >= amount:
                    self._tokens -= amount
                    return waited
                shortfall = (amount - self._tokens) / self.rate

            if deadline is not None and time.monotonic() + shortfall > deadline:
                raise TimeoutError(f"waited more than {timeout}s for rate-limit tokens")
            time.sleep(shortfall)
            waited += shortfall

    @property
    def available(self) -> float:
        with self._lock:
            self._refill(time.monotonic())
            return self._tokens


@dataclass
class _HostState:
    bucket: TokenBucket
    semaphore: threading.Semaphore
    last_request: float = 0.0
    #: From robots.txt Crawl-delay, or a server-signalled slowdown.
    min_gap: float = 0.0
    #: Monotonic time before which no request may be sent (Retry-After).
    blocked_until: float = 0.0


class HostRateLimiter:
    """Per-host token buckets, concurrency caps, and honoured backoffs."""

    def __init__(
        self,
        *,
        requests_per_second: Optional[float] = None,
        burst: Optional[int] = None,
        min_host_delay: Optional[float] = None,
        max_concurrency_per_host: Optional[int] = None,
    ) -> None:
        self.rate = requests_per_second if requests_per_second is not None else config.REQUESTS_PER_SECOND
        self.burst = burst if burst is not None else config.RATE_BURST
        self.min_host_delay = min_host_delay if min_host_delay is not None else config.MIN_HOST_DELAY
        self.max_concurrency_per_host = (
            max_concurrency_per_host
            if max_concurrency_per_host is not None
            else config.MAX_CONCURRENCY_PER_HOST
        )
        self._hosts: dict[str, _HostState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Acquisition
    # ------------------------------------------------------------------ #

    @contextmanager
    def slot(self, url: str) -> Iterator[float]:
        """Hold a request slot for this URL's host for the body of the block.

        Order matters. The concurrency semaphore is taken first so at most N
        workers are ever *waiting* on the bucket for one host, then the bucket,
        then the minimum-gap sleep. Reversing the first two lets unbounded
        workers pile up holding tokens.
        """
        host = registrable_host(url) or url
        state = self._state(host)
        waited = 0.0

        state.semaphore.acquire()
        try:
            waited += self._wait_for_block(state, host)
            waited += state.bucket.take()
            waited += self._wait_for_gap(state)
            yield waited
        finally:
            state.last_request = time.monotonic()
            state.semaphore.release()

    def _state(self, host: str) -> _HostState:
        with self._lock:
            state = self._hosts.get(host)
            if state is None:
                state = _HostState(
                    bucket=TokenBucket(self.rate, self.burst),
                    semaphore=threading.Semaphore(self.max_concurrency_per_host),
                    min_gap=self.min_host_delay,
                )
                self._hosts[host] = state
            return state

    @staticmethod
    def _wait_for_block(state: _HostState, host: str) -> float:
        remaining = state.blocked_until - time.monotonic()
        if remaining <= 0:
            return 0.0
        log.info("ratelimit.paused", host=host, seconds=round(remaining, 1))
        time.sleep(remaining)
        return remaining

    @staticmethod
    def _wait_for_gap(state: _HostState) -> float:
        if state.min_gap <= 0:
            return 0.0
        elapsed = time.monotonic() - state.last_request
        if elapsed >= state.min_gap:
            return 0.0
        time.sleep(state.min_gap - elapsed)
        return state.min_gap - elapsed

    # ------------------------------------------------------------------ #
    # Feedback from the server
    # ------------------------------------------------------------------ #

    def apply_crawl_delay(self, url: str, delay: Optional[float]) -> None:
        """Adopt a robots.txt ``Crawl-delay`` — but only ever to go slower."""
        if not delay or delay <= 0:
            return
        host = registrable_host(url) or url
        state = self._state(host)
        if delay > state.min_gap:
            state.min_gap = delay
            log.info("ratelimit.crawl_delay_applied", host=host, seconds=delay)

    def penalize(self, url: str, seconds: float) -> None:
        """Pause a host — the response to ``Retry-After``."""
        if seconds <= 0:
            return
        host = registrable_host(url) or url
        state = self._state(host)
        state.blocked_until = max(state.blocked_until, time.monotonic() + seconds)
        log.warning("ratelimit.penalty", host=host, seconds=round(seconds, 1))

    def observe_response(self, url: str, status: int, headers: Mapping[str, str]) -> None:
        """React to one response: honour Retry-After, pre-empt the next 429."""
        if status in (429, 503):
            retry_after = parse_retry_after(headers.get("retry-after"))
            if retry_after:
                self.penalize(url, min(retry_after, 3600.0))
        self.observe_headers(url, headers)

    def observe_headers(self, url: str, headers: Mapping[str, str]) -> None:
        """Slow down when the server says the budget is nearly spent.

        Recognises the IETF draft spelling (``RateLimit-Remaining``) and the
        widespread ``X-RateLimit-*`` one. Under 10% of quota, the remaining
        requests are spread across the reset window instead of being spent
        immediately and hitting the 429.
        """
        remaining = _header_float(headers, "ratelimit-remaining", "x-ratelimit-remaining")
        limit = _header_float(headers, "ratelimit-limit", "x-ratelimit-limit")
        reset = _header_float(headers, "ratelimit-reset", "x-ratelimit-reset")

        if remaining is None or limit is None or limit <= 0 or remaining > limit * 0.1:
            return

        host = registrable_host(url) or url
        state = self._state(host)
        if not reset or reset <= 0:
            return
        # `reset` is seconds-until-reset in the draft; large values are a unix
        # timestamp in the older convention.
        window = reset - time.time() if reset > 1e9 else reset
        if window <= 0:
            return
        gap = min(window / max(remaining, 1.0), 60.0)
        if gap > state.min_gap:
            state.min_gap = gap
            log.warning(
                "ratelimit.throttling_early",
                host=host,
                remaining=remaining,
                limit=limit,
                new_gap=round(gap, 2),
            )

    def stats(self) -> dict[str, dict[str, float]]:
        with self._lock:
            now = time.monotonic()
            return {
                host: {
                    "tokens_available": round(state.bucket.available, 2),
                    "min_gap": round(state.min_gap, 2),
                    "blocked_for": max(0.0, round(state.blocked_until - now, 1)),
                }
                for host, state in self._hosts.items()
            }

    def reset(self) -> None:
        with self._lock:
            self._hosts.clear()


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    """``Retry-After`` is either delta-seconds or an HTTP date. Accept both."""
    if not value:
        return None
    raw = value.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    import datetime as _dt

    now = _dt.datetime.now(_dt.timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    return max(0.0, (when - now).total_seconds())


def _header_float(headers: Mapping[str, str], *names: str) -> Optional[float]:
    lowered = {key.lower(): value for key, value in headers.items()}
    for name in names:
        raw = lowered.get(name)
        if raw is None:
            continue
        try:
            return float(str(raw).strip())
        except (TypeError, ValueError):
            continue
    return None


#: Process-wide limiter. Per-process, not per-worker-thread: 16 io threads in
#: one worker share these buckets, which is the entire point.
rate_limiter = HostRateLimiter()

__all__ = ["HostRateLimiter", "TokenBucket", "parse_retry_after", "rate_limiter"]
