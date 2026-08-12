"""Tier 0: the page has not changed, so neither has the answer.

The key is ``sha256(content_hash + schema_hash + prompt_hash)``. All three
matter:

* **content hash** — a changed page must not serve a stale answer;
* **schema hash** — asking for different fields is a different question;
* **prompt hash** — so does asking it differently.

Get any of them wrong and the cache is either useless or actively lying. Get
them right and a re-crawl of an unchanged corpus costs a Mongo lookup per page
instead of a model call per page, which is the difference between a nightly job
that finishes and one that does not.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from config import config
from observability import get_logger, metrics

log = get_logger("extract.cache")


def cache_key(content_hash: str, schema_hash: str, prompt_hash: str) -> str:
    digest = hashlib.sha256(f"{content_hash}|{schema_hash}|{prompt_hash}".encode("utf-8"))
    return digest.hexdigest()


class ExtractionCache:
    """Content-addressed store of finished extractions."""

    def __init__(self, database=None, max_memory_entries: int = 2048) -> None:
        self._db = database
        self._memory: dict[str, dict] = {}
        self._max_memory = max_memory_entries
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[dict]:
        if not config.EXTRACTION_CACHE_ENABLED:
            return None

        with self._lock:
            entry = self._memory.get(key)
        if entry is not None:
            self.hits += 1
            metrics.incr("extract.cache_hit")
            return entry

        if self._db is not None:
            try:
                doc = self._db.extraction_cache_collection().find_one({"_id": key})
            except Exception as exc:
                log.debug("extract_cache.read_failed", error=repr(exc))
                doc = None
            if doc:
                payload = {
                    "data": doc.get("data"),
                    "method": doc.get("method"),
                    "tier": doc.get("tier"),
                    "confidence": doc.get("confidence", 0.0),
                    "source_url": doc.get("url"),
                }
                self._remember(key, payload)
                self.hits += 1
                metrics.incr("extract.cache_hit")
                return payload

        self.misses += 1
        metrics.incr("extract.cache_miss")
        return None

    def put(
        self,
        key: str,
        data: dict,
        *,
        method: str,
        tier: Optional[int],
        confidence: float,
        url: str = "",
        content_hash: str = "",
        schema_hash: str = "",
    ) -> None:
        if not config.EXTRACTION_CACHE_ENABLED or not data:
            return

        payload = {
            "data": data,
            "method": method,
            "tier": tier,
            "confidence": confidence,
            "source_url": url,
        }
        self._remember(key, payload)

        if self._db is None:
            return
        try:
            self._db.extraction_cache_collection().replace_one(
                {"_id": key},
                {
                    "_id": key,
                    "data": data,
                    "method": method,
                    "tier": tier,
                    "confidence": confidence,
                    "url": url,
                    "content_hash": content_hash,
                    "schema_hash": schema_hash,
                    "created_at": datetime.now(timezone.utc),
                    "expires_at": datetime.now(timezone.utc)
                    + timedelta(days=config.EXTRACTION_CACHE_TTL_DAYS),
                },
                upsert=True,
            )
        except Exception as exc:
            log.debug("extract_cache.write_failed", error=repr(exc))

    def _remember(self, key: str, payload: dict) -> None:
        with self._lock:
            if len(self._memory) >= self._max_memory and key not in self._memory:
                # Plain FIFO eviction: an LRU needs bookkeeping on every read,
                # and the access pattern here (one pass over a crawl) makes the
                # two behave almost identically.
                self._memory.pop(next(iter(self._memory)))
            self._memory[key] = payload

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "memory_entries": len(self._memory),
        }

    def clear(self) -> None:
        with self._lock:
            self._memory.clear()


__all__ = ["ExtractionCache", "cache_key"]
