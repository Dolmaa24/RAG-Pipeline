"""The frontier: what has been seen, what is left, and when to stop.

A crawl fanned out across workers has no single process holding the seen-set,
so the seen-set has to live where every worker can reach it. Redis is already
the broker, so it is already there.

Two invariants, and the crawl is unbounded without either:

**A URL is claimed exactly once.** ``SADD`` returns whether the member was new,
atomically, so two workers discovering the same link at the same moment cannot
both enqueue it. Without this a site with a shared navigation bar re-enqueues
every page from every page.

**The budget is hard.** Every claim increments a counter and is refused past
``max_pages``. The increment-then-check can overshoot by at most the number of
workers claiming concurrently, which is bounded and small; the alternative is a
Lua script for an exactness nobody needs.

The in-memory implementation behind the same interface is what lets the crawler
run in tests and in ``main.py`` with no Redis at all.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Protocol

from config import config
from observability import get_logger

log = get_logger("discover.frontier")

#: Crawl bookkeeping outlives the crawl by a day so a finished run can still be
#: inspected, then expires itself rather than accumulating forever.
CRAWL_TTL_SECONDS = 86_400


@dataclass
class CrawlState:
    crawl_id: str
    start_url: str
    status: str = "running"          # running | finished | stopped | failed
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    claimed: int = 0                 # URLs accepted into the frontier
    fetched: int = 0                 # pages actually fetched
    collected: int = 0               # targets extracted
    failed: int = 0
    skipped: int = 0
    in_flight: int = 0
    budget: int = 0
    scope: dict = field(default_factory=dict)

    @property
    def is_done(self) -> bool:
        return self.status != "running" or (self.in_flight <= 0 and self.claimed > 0)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["elapsed_seconds"] = round(
            (self.finished_at or time.time()) - self.started_at, 1
        )
        payload["done"] = self.is_done
        return payload


class Frontier(Protocol):
    def claim(self, urls: Iterable[tuple[str, int]]) -> list[tuple[str, int]]: ...
    def state(self) -> CrawlState: ...
    def bump(self, field_name: str, amount: int = 1) -> None: ...
    def finish(self, status: str = "finished") -> None: ...


class MemoryFrontier:
    """Single-process frontier. Used by tests and the synchronous runner."""

    def __init__(self, crawl_id: str, start_url: str, max_pages: int, scope: Optional[dict] = None):
        self._seen: set[str] = set()
        self._targets: list[str] = []
        self._lock = threading.Lock()
        self._state = CrawlState(
            crawl_id=crawl_id, start_url=start_url, budget=max_pages, scope=scope or {}
        )

    def claim(self, urls: Iterable[tuple[str, int]]) -> list[tuple[str, int]]:
        accepted: list[tuple[str, int]] = []
        with self._lock:
            for url, depth in urls:
                if not url or url in self._seen:
                    continue
                if self._state.claimed >= self._state.budget:
                    break
                self._seen.add(url)
                self._state.claimed += 1
                accepted.append((url, depth))
        return accepted

    def state(self) -> CrawlState:
        with self._lock:
            return CrawlState(**asdict(self._state))

    def bump(self, field_name: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self._state, field_name, getattr(self._state, field_name, 0) + amount)

    def finish(self, status: str = "finished") -> None:
        with self._lock:
            self._state.status = status
            self._state.finished_at = time.time()

    def record_target(self, url: str) -> None:
        with self._lock:
            self._targets.append(url)

    def targets(self, limit: int = 1000) -> list[str]:
        with self._lock:
            return list(self._targets[:limit])

    def stop_requested(self) -> bool:
        with self._lock:
            return self._state.status in ("stopped", "finished")


class RedisFrontier:
    """Frontier shared by every worker in the fleet."""

    def __init__(
        self,
        crawl_id: str,
        start_url: str = "",
        max_pages: int = 0,
        scope: Optional[dict] = None,
        *,
        client=None,
    ) -> None:
        self.crawl_id = crawl_id
        self._client = client or self._connect()
        self._seen_key = f"crawl:{crawl_id}:seen"
        self._state_key = f"crawl:{crawl_id}:state"
        self._targets_key = f"crawl:{crawl_id}:targets"

        if start_url:
            self._initialise(start_url, max_pages, scope or {})

    @staticmethod
    def _connect():
        import redis

        return redis.Redis.from_url(config.REDIS_URL, decode_responses=True)

    def _initialise(self, start_url: str, max_pages: int, scope: dict) -> None:
        pipe = self._client.pipeline()
        pipe.hset(
            self._state_key,
            mapping={
                "crawl_id": self.crawl_id,
                "start_url": start_url,
                "status": "running",
                "started_at": time.time(),
                "claimed": 0, "fetched": 0, "collected": 0,
                "failed": 0, "skipped": 0, "in_flight": 0,
                "budget": max_pages,
                "scope": json.dumps(scope, default=str),
            },
        )
        pipe.expire(self._state_key, CRAWL_TTL_SECONDS)
        pipe.expire(self._seen_key, CRAWL_TTL_SECONDS)
        pipe.execute()

    def claim(self, urls: Iterable[tuple[str, int]]) -> list[tuple[str, int]]:
        """Accept only URLs never seen before, and only within budget."""
        candidates = [(url, depth) for url, depth in urls if url]
        if not candidates:
            return []

        try:
            budget = int(self._client.hget(self._state_key, "budget") or 0)
            claimed = int(self._client.hget(self._state_key, "claimed") or 0)
        except Exception as exc:
            log.warning("frontier.state_read_failed", error=repr(exc))
            return []

        remaining = max(0, budget - claimed)
        if remaining <= 0:
            return []

        # SADD is atomic and reports novelty, which is the entire dedupe
        # mechanism: two workers seeing the same link cannot both win.
        pipe = self._client.pipeline()
        for url, _ in candidates:
            pipe.sadd(self._seen_key, url)
        try:
            results = pipe.execute()
        except Exception as exc:
            log.warning("frontier.claim_failed", error=repr(exc))
            return []

        accepted: list[tuple[str, int]] = []
        for (url, depth), is_new in zip(candidates, results):
            if not is_new:
                continue
            if len(accepted) >= remaining:
                # Over budget: unsee it so a later, smaller crawl is not
                # blocked by a URL this one never actually visited.
                self._client.srem(self._seen_key, url)
                continue
            accepted.append((url, depth))

        if accepted:
            self._client.hincrby(self._state_key, "claimed", len(accepted))
            self._client.expire(self._seen_key, CRAWL_TTL_SECONDS)
        return accepted

    def state(self) -> CrawlState:
        try:
            raw = self._client.hgetall(self._state_key)
        except Exception as exc:
            log.warning("frontier.state_failed", error=repr(exc))
            raw = {}
        if not raw:
            return CrawlState(crawl_id=self.crawl_id, start_url="", status="unknown")

        def number(name: str, default: float = 0) -> float:
            try:
                return float(raw.get(name, default) or default)
            except (TypeError, ValueError):
                return default

        return CrawlState(
            crawl_id=raw.get("crawl_id", self.crawl_id),
            start_url=raw.get("start_url", ""),
            status=raw.get("status", "running"),
            started_at=number("started_at", time.time()),
            finished_at=number("finished_at") or None,
            claimed=int(number("claimed")),
            fetched=int(number("fetched")),
            collected=int(number("collected")),
            failed=int(number("failed")),
            skipped=int(number("skipped")),
            in_flight=int(number("in_flight")),
            budget=int(number("budget")),
            scope=json.loads(raw.get("scope") or "{}"),
        )

    def bump(self, field_name: str, amount: int = 1) -> None:
        try:
            self._client.hincrby(self._state_key, field_name, amount)
            self._client.expire(self._state_key, CRAWL_TTL_SECONDS)
        except Exception as exc:
            log.debug("frontier.bump_failed", field=field_name, error=repr(exc))

    def finish(self, status: str = "finished") -> None:
        try:
            self._client.hset(
                self._state_key, mapping={"status": status, "finished_at": time.time()}
            )
        except Exception as exc:
            log.debug("frontier.finish_failed", error=repr(exc))

    def record_target(self, url: str) -> None:
        """Remember a file the crawl found. For "find every PDF", this is the answer."""
        try:
            self._client.rpush(self._targets_key, url)
            self._client.expire(self._targets_key, CRAWL_TTL_SECONDS)
        except Exception as exc:
            log.debug("frontier.target_failed", error=repr(exc))

    def targets(self, limit: int = 1000) -> list[str]:
        try:
            return self._client.lrange(self._targets_key, 0, limit - 1) or []
        except Exception:
            return []

    def stop_requested(self) -> bool:
        """True once someone has asked the crawl to stop.

        Checked before each page rather than relying on revoking hundreds of
        already-queued tasks, which Celery does not do reliably.
        """
        try:
            return self._client.hget(self._state_key, "status") in ("stopped", "finished")
        except Exception:
            return False


def get_frontier(
    crawl_id: str,
    *,
    start_url: str = "",
    max_pages: int = 0,
    scope: Optional[dict] = None,
) -> Frontier:
    """A Redis frontier when Redis is reachable, an in-memory one otherwise."""
    try:
        frontier = RedisFrontier(crawl_id, start_url, max_pages, scope)
        frontier._client.ping()
        return frontier
    except Exception as exc:
        log.warning("frontier.redis_unavailable", error=repr(exc))
        return MemoryFrontier(crawl_id, start_url, max_pages, scope)


__all__ = [
    "CRAWL_TTL_SECONDS",
    "CrawlState",
    "Frontier",
    "MemoryFrontier",
    "RedisFrontier",
    "get_frontier",
]
