"""Duplicate detection, exact and near.

Two different problems:

**Exact duplicates** are the same bytes seen twice. The content hash catches
them, and the unique Mongo index turns them into upserts for free.

**Near duplicates** are the harder and more common case: the same article on
three syndication partners, the same product page with a different ad slot, the
same document reachable at ``/page`` and ``/page?ref=nav``. Byte hashes see four
distinct documents. A simhash sees one.

Simhash works by hashing shingles of the text into a 64-bit fingerprint where
*similar documents produce similar fingerprints* — the number of differing bits
(the Hamming distance) tracks how different the documents are. A threshold of 3
bits is the widely used cut-off, and holds up here.

One property to know before tuning that threshold: distance tracks the
*fraction* of shingles that changed, not the absolute amount of new text. On a
page-length document an appended advertisement moves the fingerprint by a
handful of bits; on a two-sentence stub the same advertisement is a third of the
document and moves it far outside any sane threshold. Near-duplicate detection
on very short texts is therefore noisy by construction, and the default is
calibrated for real pages.

The index is banded: a 64-bit fingerprint is split into four 16-bit bands, and
two fingerprints within 3 bits must share at least one band exactly. So instead
of comparing against every stored fingerprint, only the handful sharing a band
are compared — which is what makes this usable past a few thousand documents.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from typing import Iterable, Optional

from config import config
from observability import get_logger

log = get_logger("trust.dedupe")

_TOKEN = re.compile(r"\w+", re.UNICODE)
#: Words per shingle. Three is long enough that common phrases do not collide,
#: short enough that a paragraph edit does not change every shingle.
SHINGLE_SIZE = 3
BITS = 64
BAND_BITS = 16
BANDS = BITS // BAND_BITS


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def shingles(text: str, size: int = SHINGLE_SIZE) -> list[str]:
    words = tokenize(text)
    if len(words) < size:
        return [" ".join(words)] if words else []
    return [" ".join(words[index : index + size]) for index in range(len(words) - size + 1)]


def simhash(text: str, *, size: int = SHINGLE_SIZE) -> int:
    """64-bit similarity fingerprint of ``text``."""
    features = shingles(text, size)
    if not features:
        return 0

    weights = [0] * BITS
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(BITS):
            weights[bit] += 1 if (value >> bit) & 1 else -1

    fingerprint = 0
    for bit in range(BITS):
        if weights[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def bands_of(fingerprint: int) -> list[int]:
    """The four 16-bit bands of a fingerprint, tagged by position.

    Tagging matters: band 0 being ``0xABCD`` is a different fact from band 2
    being ``0xABCD``, and merging them would produce spurious candidates.
    """
    mask = (1 << BAND_BITS) - 1
    return [
        (index << BAND_BITS) | ((fingerprint >> (index * BAND_BITS)) & mask)
        for index in range(BANDS)
    ]


@dataclass(slots=True)
class DuplicateVerdict:
    is_duplicate: bool
    kind: str = "none"  # "exact" | "near" | "none"
    distance: Optional[int] = None
    matched_url: Optional[str] = None
    matched_hash: Optional[str] = None

    def __bool__(self) -> bool:
        return self.is_duplicate


@dataclass
class Deduplicator:
    """Banded simhash index with an optional Mongo-backed store.

    In-process, the index is a dict of band → fingerprints. With a database it
    also consults the ``fingerprints`` collection, so duplicates are caught
    across workers and across runs, not only within one process.
    """

    database: object = None
    max_distance: int = field(default_factory=lambda: config.SIMHASH_MAX_DISTANCE)
    _exact: dict[str, str] = field(default_factory=dict, init=False)
    _by_band: dict[int, list[tuple[int, str]]] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def check(self, text: str, content_hash: str, url: str) -> DuplicateVerdict:
        """Is this document one we already have?"""
        if not config.DEDUPE_ENABLED:
            return DuplicateVerdict(False)

        with self._lock:
            seen_at = self._exact.get(content_hash)
        if seen_at is not None and seen_at != url:
            return DuplicateVerdict(True, "exact", 0, seen_at, content_hash)

        if seen_at is None and self.database is not None:
            match = self._db_exact(content_hash, url)
            if match is not None:
                return DuplicateVerdict(True, "exact", 0, match, content_hash)

        fingerprint = simhash(text)
        if fingerprint == 0:
            return DuplicateVerdict(False)

        candidate = self._nearest(fingerprint, url)
        if candidate is not None:
            distance, matched_url = candidate
            log.info("dedupe.near_duplicate", url=url, matched=matched_url, distance=distance)
            return DuplicateVerdict(True, "near", distance, matched_url)

        return DuplicateVerdict(False)

    def add(self, text: str, content_hash: str, url: str) -> int:
        """Record a document. Returns its fingerprint."""
        fingerprint = simhash(text)
        with self._lock:
            self._exact[content_hash] = url
            for band in bands_of(fingerprint):
                bucket = self._by_band.setdefault(band, [])
                # Buckets are bounded: a pathological band (an empty-ish page
                # repeated ten thousand times) must not turn lookups linear.
                if len(bucket) < 256:
                    bucket.append((fingerprint, url))

        if self.database is not None:
            self._db_add(fingerprint, content_hash, url)
        return fingerprint

    def _nearest(self, fingerprint: int, url: str) -> Optional[tuple[int, str]]:
        candidates: dict[str, int] = {}

        with self._lock:
            for band in bands_of(fingerprint):
                for stored, stored_url in self._by_band.get(band, ()):
                    if stored_url == url:
                        continue
                    candidates[stored_url] = min(
                        candidates.get(stored_url, BITS), hamming(fingerprint, stored)
                    )

        if self.database is not None:
            for stored, stored_url in self._db_candidates(fingerprint):
                if stored_url == url:
                    continue
                candidates[stored_url] = min(
                    candidates.get(stored_url, BITS), hamming(fingerprint, stored)
                )

        if not candidates:
            return None
        best_url = min(candidates, key=lambda key: candidates[key])
        best_distance = candidates[best_url]
        return (best_distance, best_url) if best_distance <= self.max_distance else None

    def _db_exact(self, content_hash: str, url: str) -> Optional[str]:
        try:
            doc = self.database.fingerprints_collection().find_one(
                {"content_hash": content_hash, "url": {"$ne": url}}, {"url": 1}
            )
        except Exception:
            return None
        return doc["url"] if doc else None

    def _db_candidates(self, fingerprint: int) -> Iterable[tuple[int, str]]:
        try:
            cursor = (
                self.database.fingerprints_collection()
                .find({"bands": {"$in": bands_of(fingerprint)}}, {"simhash": 1, "url": 1})
                .limit(200)
            )
            return [(int(doc["simhash"]), doc["url"]) for doc in cursor]
        except Exception:
            return []

    def _db_add(self, fingerprint: int, content_hash: str, url: str) -> None:
        try:
            self.database.fingerprints_collection().update_one(
                {"url": url, "content_hash": content_hash},
                {
                    "$set": {
                        # Mongo has no unsigned 64-bit integer, so the
                        # fingerprint is stored as a string and compared in
                        # Python. The bands are what the query actually uses.
                        "simhash": str(fingerprint),
                        "bands": bands_of(fingerprint),
                        "url": url,
                        "content_hash": content_hash,
                    }
                },
                upsert=True,
            )
        except Exception as exc:
            log.debug("dedupe.store_failed", error=repr(exc))

    def clear(self) -> None:
        with self._lock:
            self._exact.clear()
            self._by_band.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"documents": len(self._exact), "bands": len(self._by_band)}


__all__ = [
    "BITS",
    "Deduplicator",
    "DuplicateVerdict",
    "bands_of",
    "hamming",
    "shingles",
    "simhash",
    "tokenize",
]
