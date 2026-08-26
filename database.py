"""MongoDB persistence, plus a local sink so a database outage is not data loss.

Design decisions worth stating:

**Idempotency comes from the index, not from application logic.** The unique
index on ``(canonical_url, content_hash, schema_hash)`` means resubmitting a URL
upserts instead of duplicating. Nothing has to check first, so nothing can race.

**Raw bytes go to GridFS.** Being able to re-parse without re-fetching is worth
the space: changing a prompt or a schema then costs a pass over stored bytes
rather than a full re-crawl, and re-crawling is the expensive, rate-limited,
occasionally-rude part.

**Storage failures never lose a finished extraction.** Every write falls back to
a JSONL file on disk. An Atlas outage should cost you a query interface for a
few hours, not the run.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from config import config
from observability import get_logger

log = get_logger("database")

OUTPUT_DIR = Path(__file__).parent / "output"

#: One file per record type. They were briefly pooled into one, and the result
#: was that a failed job sat in `extractions.jsonl` looking like an extraction
#: with every field null — technically distinguishable by a `kind` marker, but
#: the first thing anyone does with that file is read it as a list of results.
#: A file whose name is a promise about its contents is worth three files.
FALLBACK_PATH = OUTPUT_DIR / "extractions.jsonl"
DEAD_LETTER_PATH = OUTPUT_DIR / "dead_letter.jsonl"
RUNS_PATH = OUTPUT_DIR / "runs.jsonl"


class MongoNotConfigured(RuntimeError):
    """Raised when MONGO_URI is missing or still holds a placeholder."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CloudDatabase:
    """Thin MongoDB wrapper that degrades to local files when unavailable."""

    def __init__(self, uri: Optional[str] = None) -> None:
        self.uri = uri if uri is not None else config.MONGO_URI
        self._client = None
        self._indexed = False
        # Reentrant on purpose. `ensure_indexes()` holds this lock and then
        # calls `_database()`, which takes it again — with a plain Lock that is
        # a deadlock, and one that only appears once MONGO_URI is set, because
        # `ensure_indexes()` returns at its guard when it is not. That made it
        # invisible for as long as the pipeline ran on the JSONL fallback.
        self._lock = threading.RLock()
        self._warned = False

    @property
    def is_configured(self) -> bool:
        return bool(self.uri) and "<" not in (self.uri or "")

    def _database(self):
        if not self.uri:
            raise MongoNotConfigured(
                "MONGO_URI is not set. Copy .env.example to .env and fill in a "
                "MongoDB connection string, or run without one — results still "
                f"land in {FALLBACK_PATH}."
            )
        if "<" in self.uri or ">" in self.uri:
            raise MongoNotConfigured(
                "MONGO_URI still contains placeholder angle brackets (e.g. <password>). "
                "Replace them with the real, URL-encoded password."
            )

        with self._lock:
            if self._client is None:
                from pymongo import MongoClient

                options: dict[str, Any] = {
                    "serverSelectionTimeoutMS": 5000,
                    "appname": "universal-extractor",
                    "retryWrites": True,
                }
                # Only hand pymongo a CA bundle when the connection is actually
                # TLS. Passing `tlsCAFile` *enables* TLS as a side effect, so
                # setting it unconditionally makes the driver attempt an SSL
                # handshake against a plain local mongod — which fails with an
                # opaque "SSL handshake failed / UNEXPECTED_EOF" rather than
                # anything that points at the cause.
                if self._uses_tls():
                    import certifi

                    # Avoids the macOS "certificate verify failed" against Atlas.
                    options["tlsCAFile"] = certifi.where()

                self._client = MongoClient(self.uri, **options)
            return self._client[config.MONGO_DB_NAME]

    def _uses_tls(self) -> bool:
        """True when the connection string implies TLS.

        ``mongodb+srv://`` turns TLS on by default; a plain ``mongodb://`` does
        not unless the URI asks for it.
        """
        uri = (self.uri or "").lower()
        if uri.startswith("mongodb+srv://"):
            return "tls=false" not in uri and "ssl=false" not in uri
        return "tls=true" in uri or "ssl=true" in uri

    def get_collection(self):
        self.ensure_indexes()
        return self._database()[config.MONGO_COLLECTION]

    def specs_collection(self):
        return self._database()[config.MONGO_SPECS_COLLECTION]

    def extraction_cache_collection(self):
        return self._database()[config.MONGO_CACHE_COLLECTION]

    def http_cache_collection(self):
        return self._database()[config.MONGO_HTTP_CACHE_COLLECTION]

    def runs_collection(self):
        return self._database()[config.MONGO_RUNS_COLLECTION]

    def dead_letter_collection(self):
        return self._database()[config.MONGO_DEADLETTER_COLLECTION]

    def fingerprints_collection(self):
        return self._database()[config.MONGO_FINGERPRINTS_COLLECTION]

    def raw_bucket(self):
        from gridfs import GridFS

        return GridFS(self._database(), collection=config.GRIDFS_BUCKET)

    def ensure_indexes(self) -> None:
        """Create every index once per process. Safe to call on every write."""
        if self._indexed or not self.is_configured:
            return
        with self._lock:
            if self._indexed:
                return
            try:
                from pymongo import ASCENDING, DESCENDING

                database = self._database()

                extractions = database[config.MONGO_COLLECTION]
                # The idempotency guarantee: one row per (document, question).
                extractions.create_index(
                    [("canonical_url", ASCENDING), ("content_hash", ASCENDING),
                     ("schema_hash", ASCENDING)],
                    unique=True,
                    name="uniq_url_content_schema",
                )
                extractions.create_index([("created_at", DESCENDING)], name="created_at_desc")
                extractions.create_index([("provenance.method", ASCENDING)], name="by_method")
                extractions.create_index([("domain", ASCENDING), ("created_at", DESCENDING)],
                                         name="by_domain")
                extractions.create_index([("simhash", ASCENDING)], name="by_simhash", sparse=True)

                # TTL indexes: caches and raw bytes expire themselves.
                database[config.MONGO_CACHE_COLLECTION].create_index(
                    [("expires_at", ASCENDING)], expireAfterSeconds=0, name="ttl"
                )
                database[config.MONGO_HTTP_CACHE_COLLECTION].create_index(
                    [("expires_at", ASCENDING)], expireAfterSeconds=0, name="ttl"
                )
                database[f"{config.GRIDFS_BUCKET}.files"].create_index(
                    [("metadata.expires_at", ASCENDING)], expireAfterSeconds=0, name="ttl"
                )

                database[config.MONGO_SPECS_COLLECTION].create_index(
                    [("domain", ASCENDING), ("schema_hash", ASCENDING)], name="by_domain_schema"
                )
                database[config.MONGO_RUNS_COLLECTION].create_index(
                    [("started_at", DESCENDING)], name="started_at_desc"
                )
                database[config.MONGO_DEADLETTER_COLLECTION].create_index(
                    [("failed_at", DESCENDING)], name="failed_at_desc"
                )
                self._indexed = True
                log.info("database.indexes_ready", db=config.MONGO_DB_NAME)
            except MongoNotConfigured:
                raise
            except Exception as exc:
                # An index that cannot be created is worth knowing about, but it
                # must not stop the pipeline writing rows.
                log.warning("database.index_failed", error=repr(exc))
                self._indexed = True

    def save_item(self, item, *, run_id: Optional[str] = None) -> Optional[str]:
        """Persist one finished :class:`~models.ExtractionItem` with provenance.

        Returns the document id, or None when the write went to the fallback
        file instead. A completed extraction is never discarded because the
        database is unreachable.
        """
        from urls import registrable_host

        provenance = item.provenance(
            schema_hash=item.metadata.get("schema_hash"),
            prompt_hash=item.metadata.get("prompt_hash"),
            llm_backend=item.metadata.get("llm_backend"),
            llm_model=item.metadata.get("llm_model"),
            selector_spec_id=item.metadata.get("selector_spec"),
            pipeline_version=config.PIPELINE_VERSION,
        )

        record: dict[str, Any] = {
            "url": item.url,
            "canonical_url": item.canonical_url or item.url,
            "domain": registrable_host(item.url),
            "type": item.kind.value,
            "content_hash": item.content_hash,
            "schema_hash": item.metadata.get("schema_hash"),
            "metadata": item.metadata,
            "extracted_data": item.normalized_data or item.extracted_data,
            "provenance": provenance.model_dump(mode="json"),
            "validation_failures": item.validation_failures,
            "warnings": item.warnings,
            "run_id": run_id,
            "created_at": _utcnow(),
        }
        if item.metadata.get("simhash"):
            record["simhash"] = item.metadata["simhash"]

        return self._upsert(record)

    def _upsert(self, record: dict) -> Optional[str]:
        key = {
            "canonical_url": record["canonical_url"],
            "content_hash": record["content_hash"],
            "schema_hash": record["schema_hash"],
        }
        try:
            collection = self.get_collection()
            result = collection.update_one(
                key,
                {"$set": record, "$setOnInsert": {"first_seen": _utcnow()}},
                upsert=True,
            )
            doc_id = str(result.upserted_id) if result.upserted_id else None
            if doc_id is None:
                existing = collection.find_one(key, {"_id": 1})
                doc_id = str(existing["_id"]) if existing else None
            log.info(
                "database.saved",
                url=record["url"],
                doc_id=doc_id,
                inserted=bool(result.upserted_id),
                method=record["provenance"].get("method"),
            )
            return doc_id
        except MongoNotConfigured as exc:
            if not self._warned:
                log.warning("database.not_configured", detail=str(exc))
                self._warned = True
            self._write_fallback(record)
            return None
        except Exception as exc:
            log.error("database.write_failed", url=record.get("url"), error=repr(exc))
            self._write_fallback(record)
            return None

    @staticmethod
    def _write_fallback(record: dict, path: Optional[Path] = None) -> None:
        """Append to a local JSONL file. The sink of last resort.

        ``path`` decides which file, and the caller is expected to pick the one
        matching the record type: successful extractions never share a file
        with failures, because anything reading the results should not have to
        filter them out first.

        Defaulted to ``None`` rather than to the constant, so the module global
        is read at call time — a default argument is bound once at definition,
        which would make the destination unoverridable.
        """
        target = path or FALLBACK_PATH
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover - disk full, read-only fs
            log.error("database.fallback_failed", path=str(target), error=repr(exc))

    # Original signature, kept so existing callers keep working.
    def save_extraction(
        self,
        url: str,
        content_type: str,
        data: dict,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        from urls import canonicalize, registrable_host

        record = {
            "url": url,
            "canonical_url": canonicalize(url),
            "domain": registrable_host(url),
            "type": content_type,
            "content_hash": (metadata or {}).get("content_hash"),
            "schema_hash": (metadata or {}).get("schema_hash"),
            "metadata": metadata or {},
            "extracted_data": data,
            "provenance": {"method": "legacy", "source_url": url},
            "created_at": _utcnow(),
        }
        return self._upsert(record)

    def store_raw(self, item) -> Optional[str]:
        """Keep the fetched bytes so the document can be re-parsed later."""
        if not config.RAW_STORE_ENABLED or not item.raw_bytes:
            return None
        if not self.is_configured:
            return None
        try:
            bucket = self.raw_bucket()
            existing = bucket.find_one({"metadata.content_hash": item.content_hash})
            if existing is not None:
                return str(existing._id)
            file_id = bucket.put(
                item.raw_bytes,
                filename=item.canonical_url or item.url,
                contentType=item.content_type or "application/octet-stream",
                metadata={
                    "url": item.url,
                    "content_hash": item.content_hash,
                    "kind": item.kind.value,
                    "fetched_at": _utcnow(),
                    "expires_at": _utcnow() + timedelta(days=config.RAW_STORE_TTL_DAYS),
                },
            )
            return str(file_id)
        except Exception as exc:
            log.debug("database.raw_store_failed", error=repr(exc))
            return None

    def load_raw(self, content_hash: str) -> Optional[bytes]:
        try:
            handle = self.raw_bucket().find_one({"metadata.content_hash": content_hash})
            return handle.read() if handle is not None else None
        except Exception:
            return None

    def save_run(self, report) -> Optional[str]:
        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        payload["saved_at"] = _utcnow()
        try:
            result = self.runs_collection().insert_one(payload)
            return str(result.inserted_id)
        except Exception as exc:
            log.debug("database.run_save_failed", error=repr(exc))
            self._write_fallback({"kind": "run_report", **payload}, RUNS_PATH)
            return None

    def dead_letter(self, url: str, task: str, error: str, payload: Optional[dict] = None) -> None:
        """Record a job that exhausted its retries, so nothing fails silently."""
        record = {
            "url": url,
            "task": task,
            "error": error,
            "payload": payload or {},
            "failed_at": _utcnow(),
        }
        try:
            self.dead_letter_collection().insert_one(record)
            log.warning("database.dead_letter", url=url, task=task)
        except Exception:
            log.warning("database.dead_letter_to_file", url=url, task=task, path=str(DEAD_LETTER_PATH))
            self._write_fallback({"kind": "dead_letter", **record}, DEAD_LETTER_PATH)

    def recent(self, limit: int = 25, *, domain: Optional[str] = None) -> list[dict]:
        query = {"domain": domain} if domain else {}
        try:
            from pymongo import DESCENDING

            cursor = (
                self.get_collection()
                .find(query, {"provenance": 1, "url": 1, "type": 1, "extracted_data": 1,
                              "created_at": 1, "metadata.title": 1})
                .sort("created_at", DESCENDING)
                .limit(limit)
            )
            return [{**doc, "_id": str(doc["_id"])} for doc in cursor]
        except Exception:
            return self._read_fallback(limit)

    @staticmethod
    def _read_fallback(limit: int) -> list[dict]:
        if not FALLBACK_PATH.exists():
            return []
        try:
            lines = FALLBACK_PATH.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        records = []
        for line in reversed(lines[-limit * 4 :]):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Defensive: an older file may still hold dead letters and run
            # reports written before these were split apart. They are not
            # results, so they do not belong in a list of results.
            if record.get("kind") in ("dead_letter", "run_report"):
                continue
            records.append(record)
            if len(records) >= limit:
                break
        return records

    def dead_letters(self, limit: int = 50) -> list[dict]:
        """Jobs that exhausted their retries. Read this when a run looks short."""
        try:
            from pymongo import DESCENDING

            cursor = (
                self.dead_letter_collection()
                .find({})
                .sort("failed_at", DESCENDING)
                .limit(limit)
            )
            return [{**doc, "_id": str(doc["_id"])} for doc in cursor]
        except Exception:
            pass

        if not DEAD_LETTER_PATH.exists():
            return []
        try:
            lines = DEAD_LETTER_PATH.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        records = []
        for line in reversed(lines[-limit * 2 :]):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(records) >= limit:
                break
        return records

    def method_breakdown(self, days: int = 7) -> dict[str, int]:
        """How many records each tier produced. The cascade's report card."""
        try:
            since = _utcnow() - timedelta(days=days)
            pipeline: Iterable[dict] = [
                {"$match": {"created_at": {"$gte": since}}},
                {"$group": {"_id": "$provenance.method", "count": {"$sum": 1}}},
            ]
            return {
                (row["_id"] or "unknown"): row["count"]
                for row in self.get_collection().aggregate(list(pipeline))
            }
        except Exception:
            return {}

    def ping(self) -> bool:
        try:
            self._database().client.admin.command("ping")
            return True
        except Exception:
            return False

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None


__all__ = [
    "DEAD_LETTER_PATH",
    "FALLBACK_PATH",
    "RUNS_PATH",
    "CloudDatabase",
    "MongoNotConfigured",
]
