"""Rate-limit state that every worker process shares.

The limiter next door is built on ``threading`` primitives, which coordinate
threads inside one interpreter and nothing at all between processes. The io
queue runs a thread pool, so there it is correct. The cpu queue runs
``--pool=prefork``, and there each child got its own token bucket and its own
semaphore: a configured 0.5 requests per second became 0.5 *per child*, and a
per-host concurrency cap of 1 became one per child. Measured against a site
that had already challenged us once, both workers were refused independently --
the log shows ``breaker.blocked_permanently`` twice, because neither knew the
other had already been told no.

So the state that decides politeness lives in Redis, which every worker already
talks to.

**The clock comes from Redis, not from the caller.** ``time.monotonic()`` is
meaningless across processes -- each interpreter picks its own origin -- and
wall clocks disagree by however much NTP last corrected. A Lua script can call
``TIME`` and get one authoritative reading, so every process refills the same
bucket against the same clock.

**Concurrency is a lease, not a counter.** A worker killed mid-request cannot
decrement anything, and a plain counter would drift upward until the host was
permanently unreachable. Leases carry an expiry and are swept on the next
acquire, so a dead worker's slot returns on its own.

Redis being unavailable is not an error here. The caller falls back to its
local primitives, which is what this pipeline did before and is still correct
for a single process.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from config import config
from observability import get_logger

log = get_logger("fetch.shared")

#: How long an idle host's bucket is kept. Long enough that a paced crawl never
#: loses its place, short enough that a one-off host does not linger for ever.
_BUCKET_TTL_MS = 10 * 60 * 1000

#: An unreleased concurrency lease expires after this. It bounds how long one
#: crashed worker can hold a slot, so it must exceed the slowest legitimate
#: request -- a 64 MiB download on a slow host -- without being so long that a
#: crash stalls a crawl for minutes.
_LEASE_SECONDS = 180.0

#: Refill the bucket, take a token if one is there, and report the wait if not.
#: Nothing is deducted on a shortfall: the caller sleeps and asks again, which
#: is how the in-process bucket behaves and keeps the two interchangeable.
_TAKE = """
local key      = KEYS[1]
local rate     = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local amount   = tonumber(ARGV[3])
local ttl      = tonumber(ARGV[4])

local t   = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000

local stored  = redis.call('HMGET', key, 'tokens', 'updated')
local tokens  = tonumber(stored[1])
local updated = tonumber(stored[2])
if tokens == nil or updated == nil then
  tokens  = capacity
  updated = now
end

local elapsed = now - updated
if elapsed > 0 then
  tokens = math.min(capacity, tokens + elapsed * rate)
end

local wait = 0.0
if tokens >= amount then
  tokens = tokens - amount
else
  wait = (amount - tokens) / rate
end

redis.call('HSET', key, 'tokens', tokens, 'updated', now)
redis.call('PEXPIRE', key, ttl)
return tostring(wait)
"""

#: Sweep expired leases, then take one if the host is under its cap.
_ACQUIRE = """
local key   = KEYS[1]
local limit = tonumber(ARGV[1])
local lease = ARGV[2]
local ttl   = tonumber(ARGV[3])

local t   = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000

redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
if redis.call('ZCARD', key) < limit then
  redis.call('ZADD', key, now + ttl, lease)
  redis.call('EXPIRE', key, math.ceil(ttl) + 60)
  return 1
end
return 0
"""


class SharedLimiterState:
    """Buckets, leases and penalties, keyed by host, held in Redis."""

    def __init__(self, client) -> None:
        self._redis = client
        self._take = client.register_script(_TAKE)
        self._acquire = client.register_script(_ACQUIRE)

    # -- keys ---------------------------------------------------------- #

    @staticmethod
    def _bucket_key(host: str) -> str:
        return f"ratelimit:bucket:{host}"

    @staticmethod
    def _lease_key(host: str) -> str:
        return f"ratelimit:leases:{host}"

    @staticmethod
    def _penalty_key(host: str) -> str:
        return f"ratelimit:penalty:{host}"

    # -- the bucket ---------------------------------------------------- #

    def take(self, host: str, *, rate: float, capacity: float, amount: float = 1.0) -> float:
        """Seconds to wait before this request may go out. 0 means now."""
        raw = self._take(
            keys=[self._bucket_key(host)],
            args=[rate, capacity, amount, _BUCKET_TTL_MS],
        )
        return float(raw)

    # -- concurrency --------------------------------------------------- #

    def try_acquire(self, host: str, limit: int) -> Optional[str]:
        """A lease id if the host is under its cap, else None."""
        lease = uuid.uuid4().hex
        taken = self._acquire(
            keys=[self._lease_key(host)], args=[limit, lease, _LEASE_SECONDS]
        )
        return lease if taken else None

    def release(self, host: str, lease: str) -> None:
        self._redis.zrem(self._lease_key(host), lease)

    # -- penalties ----------------------------------------------------- #

    def penalize(self, host: str, seconds: float) -> None:
        """Pause a host for every worker, not just this one.

        Extends rather than overwrites: a longer ``Retry-After`` arriving while
        a shorter pause is running must not shorten it.
        """
        key = self._penalty_key(host)
        remaining = self.penalty_remaining(host)
        if seconds > remaining:
            self._redis.set(key, "1", px=int(seconds * 1000))

    def penalty_remaining(self, host: str) -> float:
        """Seconds left on this host's pause. The key's own TTL is the clock."""
        ttl = self._redis.pttl(self._penalty_key(host))
        return ttl / 1000.0 if ttl and ttl > 0 else 0.0

    # -- housekeeping -------------------------------------------------- #

    def clear(self, host: Optional[str] = None) -> None:
        pattern = f"ratelimit:*:{host}" if host else "ratelimit:*"
        keys = list(self._redis.scan_iter(pattern, count=500))
        if keys:
            self._redis.delete(*keys)


def connect() -> Optional[SharedLimiterState]:
    """Shared state, or ``None`` when Redis cannot answer.

    A pipeline that cannot reach Redis has larger problems than rate-limit
    accuracy -- there would be no queue either -- so this degrades to the
    in-process limiter rather than refusing to fetch. The warning is what makes
    a quieter-than-configured crawl explicable later.
    """
    if not getattr(config, "RATELIMIT_SHARED", True):
        return None
    try:
        import redis

        client = redis.Redis.from_url(config.REDIS_URL, socket_timeout=2.0)
        client.ping()
        return SharedLimiterState(client)
    except Exception as exc:
        log.warning(
            "ratelimit.shared_unavailable",
            error=repr(exc),
            effect="per-process limits; a prefork pool will exceed the configured rate",
        )
        return None


__all__ = ["SharedLimiterState", "connect"]
