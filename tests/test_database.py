"""The file sink of last resort: one file per record type."""

from __future__ import annotations

import json

import pytest

import database
from database import CloudDatabase
from models import ExtractionItem, ExtractionMethod, ResourceKind, RunReport


@pytest.fixture
def sink(tmp_path, monkeypatch):
    """Point every fallback file at a temp directory."""
    monkeypatch.setattr(database, "OUTPUT_DIR", tmp_path, raising=False)
    monkeypatch.setattr(database, "FALLBACK_PATH", tmp_path / "extractions.jsonl", raising=False)
    monkeypatch.setattr(database, "DEAD_LETTER_PATH", tmp_path / "dead_letter.jsonl", raising=False)
    monkeypatch.setattr(database, "RUNS_PATH", tmp_path / "runs.jsonl", raising=False)
    # No MONGO_URI, so everything takes the file path.
    return CloudDatabase(uri=None), tmp_path


def read(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestRecordSeparation:
    def test_extractions_dead_letters_and_runs_go_to_different_files(self, sink):
        db, tmp = sink

        item = ExtractionItem(url="https://a.test/ok", raw_bytes=b"body")
        item.compute_content_hash()
        item.kind = ResourceKind.HTML
        item.method = ExtractionMethod.STRUCTURED_DATA
        item.normalized_data = {"title": "A result"}
        db.save_item(item)

        db.dead_letter("https://a.test/bad", "tasks.extract_url", "RuntimeError: boom")
        db.save_run(RunReport())

        extractions = read(tmp / "extractions.jsonl")
        dead = read(tmp / "dead_letter.jsonl")
        runs = read(tmp / "runs.jsonl")

        assert len(extractions) == 1
        assert len(dead) == 1
        assert len(runs) == 1

    def test_the_results_file_holds_only_results(self, sink):
        """Anything reading it should not have to filter failures out first."""
        db, tmp = sink
        db.dead_letter("https://a.test/bad", "tasks.extract_url", "boom")
        db.save_run(RunReport())

        assert read(tmp / "extractions.jsonl") == []

    def test_a_dead_letter_carries_what_is_needed_to_act_on_it(self, sink):
        db, tmp = sink
        db.dead_letter("https://a.test/bad", "tasks.extract_url", "RuntimeError: boom",
                       payload={"task_id": "abc"})
        record = read(tmp / "dead_letter.jsonl")[0]
        assert record["url"] == "https://a.test/bad"
        assert record["task"] == "tasks.extract_url"
        assert "boom" in record["error"]
        assert record["payload"]["task_id"] == "abc"
        assert record["failed_at"]

    def test_extraction_records_keep_their_provenance(self, sink):
        db, tmp = sink
        item = ExtractionItem(url="https://a.test/x", raw_bytes=b"body")
        item.compute_content_hash()
        item.kind = ResourceKind.DOCUMENT
        item.method = ExtractionMethod.LLM
        item.tier = 3
        item.normalized_data = {"title": "T"}
        db.save_item(item)

        record = read(tmp / "extractions.jsonl")[0]
        assert record["provenance"]["method"] == "llm"
        assert record["provenance"]["tier"] == 3
        assert record["extracted_data"] == {"title": "T"}


class TestConnectionOptions:
    """Two bugs that only appeared once MONGO_URI was actually set."""

    @pytest.mark.parametrize(
        "uri,expected",
        [
            ("mongodb://localhost:27017", False),  # noqa: PT014 - distinct cases below
            ("mongodb://host:27017/?tls=true", True),
            ("mongodb://host:27017/?ssl=true", True),
            ("mongodb+srv://u:p@cluster.mongodb.net/", True),
            ("mongodb+srv://u:p@cluster.net/?tls=false", False),
        ],
    )
    def test_tls_is_inferred_from_the_uri(self, uri, expected):
        """`tlsCAFile` *enables* TLS as a side effect.

        Passing it unconditionally made the driver attempt an SSL handshake
        against a plain local mongod, which fails with an opaque
        "SSL handshake failed / UNEXPECTED_EOF" that says nothing about why.
        """
        assert CloudDatabase(uri=uri)._uses_tls() is expected

    def test_the_connection_lock_is_reentrant(self):
        """`ensure_indexes()` holds the lock and then calls `_database()`.

        With a plain `threading.Lock` that is a deadlock — and one invisible
        until MONGO_URI is set, because `ensure_indexes()` returns at its guard
        when the database is not configured.
        """
        db = CloudDatabase(uri="mongodb://localhost:27017")
        assert db._lock.acquire(blocking=False)
        try:
            assert db._lock.acquire(blocking=False), "a non-reentrant lock deadlocks here"
            db._lock.release()
        finally:
            db._lock.release()

    def test_ensure_indexes_is_a_no_op_without_a_uri(self):
        db = CloudDatabase(uri=None)
        db.ensure_indexes()  # must return rather than raise or block
        assert not db.is_configured

    def test_a_placeholder_uri_is_not_treated_as_configured(self):
        assert not CloudDatabase(uri="mongodb+srv://user:<password>@c.net/").is_configured


class TestReaders:
    def test_recent_ignores_legacy_mixed_rows(self, sink):
        """An older file may still hold all three types in one place."""
        db, tmp = sink
        path = tmp / "extractions.jsonl"
        path.write_text(
            json.dumps({"url": "https://a.test/1", "extracted_data": {"a": 1}}) + "\n"
            + json.dumps({"kind": "dead_letter", "url": "https://a.test/2"}) + "\n"
            + json.dumps({"kind": "run_report", "submitted": 3}) + "\n"
        )
        recent = db.recent(limit=10)
        assert len(recent) == 1
        assert recent[0]["url"] == "https://a.test/1"

    def test_dead_letters_are_readable_back(self, sink):
        db, tmp = sink
        for index in range(3):
            db.dead_letter(f"https://a.test/{index}", "tasks.extract_url", "boom")
        found = db.dead_letters(limit=10)
        assert len(found) == 3
        assert all(record["kind"] == "dead_letter" for record in found)

    def test_no_files_yet(self, sink):
        db, _ = sink
        assert db.recent() == []
        assert db.dead_letters() == []
