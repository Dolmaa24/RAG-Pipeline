"""Conditional-request HTTP cache.

The cheapest fetch is the one the server answers with ``304 Not Modified``:
no body, no parse, no extraction. On a re-crawl that is the difference between
a job that costs minutes and one that costs milliseconds.

Two stores behind one interface:

* :class:`MemoryCache` — per-process, no dependencies, used when Mongo is not
  configured or in tests.
* :class:`MongoCache` — shared across every worker, which is what makes a
  16-thread io queue benefit from a validator another worker stored a minute
  ago.

Only ``ETag`` and ``Last-Modified`` are stored, plus the body. This is not a
full RFC 9111 implementation: it revalidates rather than trusting freshness
lifetimes, so a stale answer is never served from the cache without the origin
agreeing to it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol

from config import config
from observability import get_logger
from urls import canonicalize

log = get_logger("fetch.cache")


@dataclass(slots=True)
class CachedResponse:
    url: str
    status: int
    headers: dict[str, str]
    body: bytes
    stored_at: float

    @property
    def etag(self) -> Optional[str]:
        return self.headers.get("etag")

    @property
    def last_modified(self) -> Optional[str]:
        return self.headers.get("last-modified")

    def validators(self) -> dict[str, str]:
        """The conditional headers to send on the next request for this URL."""
        headers: dict[str, str] = {}
        if self.etag:
            headers["If-None-Match"] = self.etag
        if self.last_modified:
            headers["If-Modified-Since"] = self.last_modified
        return headers

    @property
    def age_seconds(self) -> float:
        return time.time() - self.stored_at


class CacheBackend(Protocol):
    def get(self, key: str) -> Optional[CachedResponse]: ...
    def put(self, key: str, response: CachedResponse) -> None: ...
    def clear(self) -> None: ...


class MemoryCache:
    """Bounded in-process cache. Evicts the oldest entry when full."""

    def __init__(self, max_entries: int = 512) -> None:
        self.max_entries = max_entries
        self._entries: dict[str, CachedResponse] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[CachedResponse]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.age_seconds > config.HTTP_CACHE_TTL_SECONDS:
                del self._entries[key]
                return None
            return entry

    def put(self, key: str, response: CachedResponse) -> None:
        with self._lock:
            if len(self._entries) >= self.max_entries and key not in self._entries:
                oldest = min(self._entries, key=lambda k: self._entries[k].stored_at)
                del self._entries[oldest]
            self._entries[key] = response

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class MongoCache:
    """Shared cache in a TTL-indexed Mongo collection.

    Bodies over ~12 MB are not stored: the 16 MB BSON document limit makes that
    a failed write rather than a slow one, and a cache that occasionally misses
    is fine while a cache that throws is not.
    """

    MAX_BODY_BYTES = 12 * 1024 * 1024

    def __init__(self, database) -> None:  # database: CloudDatabase
        self._db = database

    def get(self, key: str) -> Optional[CachedResponse]:
        try:
            collection = self._db.http_cache_collection()
            doc = collection.find_one({"_id": key})
        except Exception as exc:
            log.debug("cache.read_failed", error=repr(exc))
            return None
        if not doc:
            return None
        return CachedResponse(
            url=doc.get("url", ""),
            status=doc.get("status", 200),
            headers=doc.get("headers", {}),
            body=bytes(doc.get("body", b"")),
            stored_at=doc.get("stored_at", 0.0),
        )

    def put(self, key: str, response: CachedResponse) -> None:
        if len(response.body) > self.MAX_BODY_BYTES:
            return
        try:
            from datetime import datetime, timedelta, timezone

            collection = self._db.http_cache_collection()
            collection.replace_one(
                {"_id": key},
                {
                    "_id": key,
                    "url": response.url,
                    "status": response.status,
                    "headers": response.headers,
                    "body": response.body,
                    "stored_at": response.stored_at,
                    "expires_at": datetime.now(timezone.utc)
                    + timedelta(seconds=config.HTTP_CACHE_TTL_SECONDS),
                },
                upsert=True,
            )
        except Exception as exc:
            log.debug("cache.write_failed", error=repr(exc))

    def clear(self) -> None:
        try:
            self._db.http_cache_collection().delete_many({})
        except Exception:  # pragma: no cover - best effort
            pass


class HttpCache:
    """Front end over a backend, with the revalidation rules in one place."""

    #: Only these are worth keeping. A response with no validator cannot be
    #: revalidated, so storing it would mean serving it blind.
    _KEEP_HEADERS = frozenset(
        {"etag", "last-modified", "content-type", "content-encoding", "content-language", "vary"}
    )

    def __init__(self, backend: Optional[CacheBackend] = None) -> None:
        self.backend: CacheBackend = backend or MemoryCache()
        self.hits = 0
        self.misses = 0
        self.revalidated = 0

    @staticmethod
    def key_for(url: str) -> str:
        return canonicalize(url)

    def lookup(self, url: str) -> Optional[CachedResponse]:
        if not config.HTTP_CACHE_ENABLED:
            return None
        entry = self.backend.get(self.key_for(url))
        if entry is None:
            self.misses += 1
        return entry

    def store(self, url: str, status: int, headers: Mapping[str, str], body: bytes) -> None:
        """Store a 200 that carries a validator. Anything else is skipped."""
        if not config.HTTP_CACHE_ENABLED or status != 200:
            return
        lowered = {k.lower(): v for k, v in headers.items()}
        if "etag" not in lowered and "last-modified" not in lowered:
            return
        if "no-store" in lowered.get("cache-control", "").lower():
            return
        kept = {k: v for k, v in lowered.items() if k in self._KEEP_HEADERS}
        self.backend.put(
            self.key_for(url),
            CachedResponse(url=url, status=status, headers=kept, body=body, stored_at=time.time()),
        )

    def on_not_modified(self, url: str, entry: CachedResponse) -> CachedResponse:
        """Refresh the stored timestamp after a 304 and return the cached body."""
        self.hits += 1
        self.revalidated += 1
        entry.stored_at = time.time()
        self.backend.put(self.key_for(url), entry)
        log.info("cache.revalidated", url=url, bytes=len(entry.body))
        return entry

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "revalidated": self.revalidated}

    def clear(self) -> None:
        self.backend.clear()


http_cache = HttpCache()

__all__ = ["CacheBackend", "CachedResponse", "HttpCache", "MemoryCache", "MongoCache", "http_cache"]
