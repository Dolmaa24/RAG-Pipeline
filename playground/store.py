"""Threads and messages, in SQLite.

Not Mongo, which every other record in this pipeline uses, and the reason is
specific rather than aesthetic: ``MONGO_URI`` is unset here, so
``CloudDatabase.is_configured`` is False and every write falls through to a
JSONL file. That is a reasonable outcome for an extraction record you can
re-run; it is not one for a conversation, where the whole feature is that you
can open the thread again and find it.

Not SQLAlchemy either. It is installed, but only as a transitive dependency of
langchain and absent from ``requirements.txt``, so reaching for it means
declaring a dependency to hold two tables and about eight queries. ``sqlite3``
is in the standard library and is what a single-machine pipeline should use for
a few thousand rows.

**Migrations are ``PRAGMA user_version``.** A list of steps, applied forward
from whatever version the file is at. That is the smallest thing that is still
a migration path — it survives a new column later without anyone remembering to
run a tool, and it needs nothing installed.

**Foreign keys are turned on per connection.** SQLite defaults them *off*, so
``ON DELETE CASCADE`` is silently inert without the pragma and deleting a
thread would leave its messages behind for ever. It has its own test.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from config import config
from observability import get_logger

log = get_logger("playground.store")

ROLES = ("user", "assistant", "system", "tool")

#: How much of the first message becomes the thread's title.
TITLE_CHARS = 60


class ThreadNotFound(LookupError):
    """No thread with that id. A 404, not a server fault."""


def _now() -> str:
    """ISO-8601 UTC, to the microsecond.

    Not to the second. Threads are ordered by ``updated_at`` and two created
    inside the same second would tie, leaving the order to fall back on a
    random uuid — so a thread you just replied to would sometimes not move to
    the top of the list, and sometimes would.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def database_path() -> Path:
    """Where the conversations live. Relative paths resolve against the
    repository root, so a worker started from anywhere opens the same file."""
    configured = Path(config.PLAYGROUND_DB_PATH).expanduser()
    if configured.is_absolute():
        return configured
    return (Path(__file__).resolve().parents[1] / configured).resolve()


# --- migrations -------------------------------------------------------------


def _v1(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS threads (
            id                 TEXT PRIMARY KEY,
            title              TEXT NOT NULL,
            status             TEXT NOT NULL DEFAULT 'active',
            summary            TEXT NOT NULL DEFAULT '',
            summarised_through INTEGER NOT NULL DEFAULT 0,
            created_at         TEXT NOT NULL,
            updated_at         TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id  TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
            role       TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
            content    TEXT NOT NULL,
            meta       TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS messages_by_thread ON messages(thread_id, id);
        CREATE INDEX IF NOT EXISTS threads_by_updated ON threads(updated_at DESC);
        """
    )


#: Applied in order, from whatever ``user_version`` the file is at to len().
#: Append; never reorder, and never edit one that has shipped.
_MIGRATIONS = [_v1]

_migrated: set[str] = set()
_migrate_lock = threading.Lock()


def migrate(path: Optional[Path] = None) -> int:
    """Bring the file up to the current schema. Returns the version it is at."""
    target = path or database_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(target))
    try:
        # WAL so a reader and a writer coexist. Celery's prefork children each
        # open their own connection, and the default rollback journal makes one
        # of them wait on the other for the whole transaction.
        connection.execute("PRAGMA journal_mode=WAL")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        for step in _MIGRATIONS[version:]:
            step(connection)
            version += 1
            connection.execute(f"PRAGMA user_version = {version}")
        connection.commit()
    finally:
        connection.close()
    return version


@contextmanager
def _connect(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """One connection, migrated once per file per process."""
    target = path or database_path()
    key = str(target)
    if key not in _migrated:
        with _migrate_lock:
            if key not in _migrated:
                migrate(target)
                _migrated.add(key)

    connection = sqlite3.connect(str(target), timeout=10.0)
    connection.row_factory = sqlite3.Row
    # Off by default in SQLite, which makes ON DELETE CASCADE do nothing at all.
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def reset_cache() -> None:
    """Forget which files have been migrated. For tests using tmp_path."""
    _migrated.clear()


# --- rows -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Message:
    id: int
    thread_id: str
    role: str
    content: str
    created_at: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "meta": self.meta,
        }


@dataclass(frozen=True, slots=True)
class Thread:
    id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    summary: str = ""
    summarised_through: int = 0
    message_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "has_summary": bool(self.summary),
        }


def _thread(row: sqlite3.Row, count: int = 0) -> Thread:
    return Thread(
        id=row["id"],
        title=row["title"],
        status=row["status"],
        summary=row["summary"],
        summarised_through=row["summarised_through"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        message_count=count,
    )


def _message(row: sqlite3.Row) -> Message:
    try:
        meta = json.loads(row["meta"] or "{}")
    except json.JSONDecodeError:
        meta = {}
    return Message(
        id=row["id"],
        thread_id=row["thread_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        meta=meta if isinstance(meta, dict) else {},
    )


# --- threads ----------------------------------------------------------------


def title_from(text: str) -> str:
    """A thread's name, from whatever was said first.

    One line, because a title is rendered in a narrow list, and a message
    beginning with three blank lines would otherwise produce a blank entry
    nobody can tell apart from another blank entry.
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "New conversation"
    if len(cleaned) <= TITLE_CHARS:
        return cleaned
    return cleaned[: TITLE_CHARS - 1].rstrip() + "…"


def create_thread(title: str = "", *, path: Optional[Path] = None) -> Thread:
    now = _now()
    thread_id = uuid.uuid4().hex
    with _connect(path) as connection:
        connection.execute(
            "INSERT INTO threads (id, title, status, summary, summarised_through,"
            " created_at, updated_at) VALUES (?, ?, 'active', '', 0, ?, ?)",
            (thread_id, title_from(title), now, now),
        )
    log.info("playground.thread_created", thread=thread_id)
    return Thread(
        id=thread_id, title=title_from(title), status="active",
        created_at=now, updated_at=now,
    )


def get_thread(thread_id: str, *, path: Optional[Path] = None) -> Thread:
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT * FROM threads WHERE id = ?", (thread_id,)
        ).fetchone()
        if row is None:
            raise ThreadNotFound(f"no thread {thread_id!r}")
        count = connection.execute(
            "SELECT COUNT(*) FROM messages WHERE thread_id = ?", (thread_id,)
        ).fetchone()[0]
    return _thread(row, count)


def list_threads(
    *, limit: int = 50, offset: int = 0, path: Optional[Path] = None
) -> list[Thread]:
    """Newest activity first — which is `updated_at`, not `created_at`. A thread
    you replied to a minute ago belongs at the top however old it is."""
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT t.*, COUNT(m.id) AS n FROM threads t"
            " LEFT JOIN messages m ON m.thread_id = t.id"
            " GROUP BY t.id ORDER BY t.updated_at DESC, t.id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    return [_thread(row, row["n"]) for row in rows]


def count_threads(*, path: Optional[Path] = None) -> int:
    with _connect(path) as connection:
        return connection.execute("SELECT COUNT(*) FROM threads").fetchone()[0]


def delete_thread(thread_id: str, *, path: Optional[Path] = None) -> int:
    """Remove a thread and its messages. Returns how many messages went."""
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM messages WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        removed = row[0] if row else 0
        cursor = connection.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        if cursor.rowcount == 0:
            raise ThreadNotFound(f"no thread {thread_id!r}")
    log.info("playground.thread_deleted", thread=thread_id, messages=removed)
    return removed


def set_summary(
    thread_id: str, summary: str, through: int, *, path: Optional[Path] = None
) -> None:
    """Store the rolling summary and how far into the thread it reaches."""
    with _connect(path) as connection:
        connection.execute(
            "UPDATE threads SET summary = ?, summarised_through = ? WHERE id = ?",
            (summary, through, thread_id),
        )


def set_status(thread_id: str, status: str, *, path: Optional[Path] = None) -> None:
    with _connect(path) as connection:
        connection.execute(
            "UPDATE threads SET status = ?, updated_at = ? WHERE id = ?",
            (status, _now(), thread_id),
        )


# --- messages ---------------------------------------------------------------


def append_message(
    thread_id: str,
    role: str,
    content: str,
    *,
    meta: Optional[dict[str, Any]] = None,
    path: Optional[Path] = None,
) -> Message:
    """Add one message, and move the thread to the top of the list.

    The thread's ``updated_at`` moves here and nowhere else. Reading a thread
    must not reorder the list under someone who is only looking at it.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {', '.join(ROLES)}; got {role!r}")

    now = _now()
    with _connect(path) as connection:
        exists = connection.execute(
            "SELECT title FROM threads WHERE id = ?", (thread_id,)
        ).fetchone()
        if exists is None:
            raise ThreadNotFound(f"no thread {thread_id!r}")

        cursor = connection.execute(
            "INSERT INTO messages (thread_id, role, content, meta, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (thread_id, role, content, json.dumps(meta or {}), now),
        )
        # An untitled thread takes its name from the first thing said in it.
        if role == "user" and exists["title"] in ("", "New conversation"):
            connection.execute(
                "UPDATE threads SET title = ? WHERE id = ?",
                (title_from(content), thread_id),
            )
        connection.execute(
            "UPDATE threads SET updated_at = ? WHERE id = ?", (now, thread_id)
        )
        message_id = cursor.lastrowid

    return Message(
        id=message_id, thread_id=thread_id, role=role,
        content=content, created_at=now, meta=meta or {},
    )


def messages(
    thread_id: str, *, since: int = 0, path: Optional[Path] = None
) -> list[Message]:
    """Every message in the thread, oldest first."""
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT * FROM messages WHERE thread_id = ? AND id > ? ORDER BY id",
            (thread_id, since),
        ).fetchall()
    return [_message(row) for row in rows]


def last_message(thread_id: str, *, path: Optional[Path] = None) -> Optional[Message]:
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT * FROM messages WHERE thread_id = ? ORDER BY id DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
    return _message(row) if row else None


__all__ = [
    "Message",
    "Thread",
    "ThreadNotFound",
    "append_message",
    "count_threads",
    "create_thread",
    "database_path",
    "delete_thread",
    "get_thread",
    "last_message",
    "list_threads",
    "messages",
    "migrate",
    "reset_cache",
    "set_status",
    "set_summary",
    "title_from",
]
