"""Threads and messages on disk.

SQLite rather than Mongo, and the choice is defended in the module: MONGO_URI
is unset by default, so every write there falls through to a JSONL file. An
extraction record that lands in a file can be re-run; a conversation you cannot
reopen is simply gone.

The cascade test is the one worth reading. SQLite defaults foreign keys *off*,
so ``ON DELETE CASCADE`` is silently inert unless a pragma turns it on per
connection — the failure is not an error, it is orphaned rows accumulating for
ever.
"""

from __future__ import annotations

import sqlite3

import pytest

from config import config
from playground import store


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "pg.db"
    monkeypatch.setattr(config, "PLAYGROUND_DB_PATH", str(path))
    store.reset_cache()
    yield path
    store.reset_cache()


# --- migration --------------------------------------------------------------


def test_a_fresh_file_migrates_to_the_current_version(db):
    assert store.migrate(db) == len(store._MIGRATIONS)


def test_migrating_twice_changes_nothing(db):
    first = store.migrate(db)
    assert store.migrate(db) == first


def test_the_version_is_recorded_in_the_file(db):
    """PRAGMA user_version, so the file itself knows — no separate ledger to
    fall out of step with it."""
    store.migrate(db)
    connection = sqlite3.connect(str(db))
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == len(
            store._MIGRATIONS
        )
    finally:
        connection.close()


def test_an_existing_thread_survives_a_migration(db):
    thread = store.create_thread("keep me")
    store.migrate(db)
    assert store.get_thread(thread.id).title == "keep me"


def test_the_database_is_created_on_demand(db):
    assert not db.exists()
    store.create_thread("first")
    assert db.exists()


# --- threads ----------------------------------------------------------------


def test_a_thread_round_trips(db):
    made = store.create_thread("About the corpus")
    got = store.get_thread(made.id)
    assert got.id == made.id
    assert got.title == "About the corpus"
    assert got.status == "active"


def test_a_thread_id_is_a_uuid(db):
    assert len(store.create_thread().id) == 32


def test_two_threads_do_not_share_an_id(db):
    assert store.create_thread().id != store.create_thread().id


def test_an_unknown_thread_is_an_error_not_none(db):
    """A 404 upstream. Returning None would make every caller check."""
    with pytest.raises(store.ThreadNotFound):
        store.get_thread("nope")


def test_the_title_comes_from_the_first_user_message(db):
    thread = store.create_thread()
    store.append_message(thread.id, "user", "What does the corpus contain?")
    assert store.get_thread(thread.id).title == "What does the corpus contain?"


def test_a_later_message_does_not_rename_the_thread(db):
    thread = store.create_thread()
    store.append_message(thread.id, "user", "First question")
    store.append_message(thread.id, "user", "Second question")
    assert store.get_thread(thread.id).title == "First question"


def test_an_explicit_title_is_not_overwritten(db):
    thread = store.create_thread("Chosen name")
    store.append_message(thread.id, "user", "Something else entirely")
    assert store.get_thread(thread.id).title == "Chosen name"


def test_a_long_title_is_trimmed(db):
    thread = store.create_thread("word " * 60)
    title = store.get_thread(thread.id).title
    assert len(title) <= store.TITLE_CHARS
    assert title.endswith("…")


def test_a_title_is_one_line(db):
    """It is rendered in a narrow list, and a message starting with blank lines
    would otherwise produce an entry nobody can tell from another."""
    assert "\n" not in store.title_from("\n\n\nWhat is this?\n\nAnd that?")


def test_an_empty_title_has_a_name_anyway(db):
    assert store.title_from("   ") == "New conversation"


# --- ordering ---------------------------------------------------------------


def test_threads_are_listed_by_most_recent_activity(db):
    first = store.create_thread("first")
    second = store.create_thread("second")
    store.append_message(first.id, "user", "bumped")

    assert [t.id for t in store.list_threads()][0] == first.id


def test_reading_a_thread_does_not_reorder_the_list(db):
    """Someone browsing history must not shuffle it under themselves."""
    first = store.create_thread("first")
    second = store.create_thread("second")
    before = [t.id for t in store.list_threads()]

    store.get_thread(first.id)
    store.messages(first.id)

    assert [t.id for t in store.list_threads()] == before


def test_appending_moves_a_thread_to_the_top(db):
    a = store.create_thread("a")
    b = store.create_thread("b")
    store.append_message(a.id, "user", "hello")
    assert store.list_threads()[0].id == a.id


def test_listing_pages(db):
    for i in range(5):
        store.create_thread(f"thread {i}")
    assert len(store.list_threads(limit=2)) == 2
    assert len(store.list_threads(limit=2, offset=4)) == 1
    assert store.count_threads() == 5


# --- messages ---------------------------------------------------------------


def test_messages_come_back_oldest_first(db):
    thread = store.create_thread()
    for i in range(4):
        store.append_message(thread.id, "user", f"m{i}")
    assert [m.content for m in store.messages(thread.id)] == ["m0", "m1", "m2", "m3"]


def test_the_message_count_is_reported(db):
    thread = store.create_thread()
    store.append_message(thread.id, "user", "one")
    store.append_message(thread.id, "assistant", "two")
    assert store.get_thread(thread.id).message_count == 2


def test_meta_round_trips_as_json(db):
    thread = store.create_thread()
    store.append_message(
        thread.id, "assistant", "answer", meta={"sources": [{"number": 1}], "path": "answer"}
    )
    stored = store.messages(thread.id)[0]
    assert stored.meta["path"] == "answer"
    assert stored.meta["sources"][0]["number"] == 1


def test_broken_meta_does_not_break_reading(db):
    """A row written by an older version, or by hand. The message still matters
    more than its annotations."""
    thread = store.create_thread()
    store.append_message(thread.id, "assistant", "answer")
    connection = sqlite3.connect(str(db))
    try:
        connection.execute("UPDATE messages SET meta = 'not json'")
        connection.commit()
    finally:
        connection.close()
    assert store.messages(thread.id)[0].meta == {}


def test_an_unknown_role_is_refused(db):
    thread = store.create_thread()
    with pytest.raises(ValueError, match="role must be one of"):
        store.append_message(thread.id, "narrator", "hello")


def test_a_message_to_an_unknown_thread_is_refused(db):
    with pytest.raises(store.ThreadNotFound):
        store.append_message("nope", "user", "hello")


def test_since_returns_only_what_is_newer(db):
    thread = store.create_thread()
    first = store.append_message(thread.id, "user", "one")
    store.append_message(thread.id, "assistant", "two")
    assert [m.content for m in store.messages(thread.id, since=first.id)] == ["two"]


def test_the_last_message_is_the_newest(db):
    thread = store.create_thread()
    store.append_message(thread.id, "user", "one")
    store.append_message(thread.id, "assistant", "two")
    assert store.last_message(thread.id).content == "two"


def test_an_empty_thread_has_no_last_message(db):
    assert store.last_message(store.create_thread().id) is None


# --- deletion ---------------------------------------------------------------


def test_deleting_a_thread_removes_its_messages(db):
    """The cascade. SQLite defaults foreign keys OFF, so without the pragma
    this passes the delete and silently orphans every message."""
    thread = store.create_thread()
    for i in range(3):
        store.append_message(thread.id, "user", f"m{i}")

    assert store.delete_thread(thread.id) == 3

    connection = sqlite3.connect(str(db))
    try:
        left = connection.execute(
            "SELECT COUNT(*) FROM messages WHERE thread_id = ?", (thread.id,)
        ).fetchone()[0]
    finally:
        connection.close()
    assert left == 0


def test_foreign_keys_are_actually_on(db):
    """Asserted directly, because the cascade above is the only symptom and it
    is one someone could 'fix' by deleting messages in application code."""
    store.create_thread()
    from playground.store import _connect

    with _connect(db) as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_deleting_leaves_other_threads_alone(db):
    keep = store.create_thread("keep")
    store.append_message(keep.id, "user", "mine")
    doomed = store.create_thread("go")
    store.append_message(doomed.id, "user", "theirs")

    store.delete_thread(doomed.id)
    assert len(store.messages(keep.id)) == 1


def test_deleting_an_unknown_thread_is_an_error(db):
    with pytest.raises(store.ThreadNotFound):
        store.delete_thread("nope")


# --- summary ----------------------------------------------------------------


def test_a_summary_is_stored_with_how_far_it_reaches(db):
    thread = store.create_thread()
    store.set_summary(thread.id, "They discussed the corpus.", 7)
    got = store.get_thread(thread.id)
    assert got.summary == "They discussed the corpus."
    assert got.summarised_through == 7


def test_a_new_thread_has_no_summary(db):
    got = store.get_thread(store.create_thread().id)
    assert got.summary == "" and got.summarised_through == 0
