"""Releasing the Kuzu lock, deterministically.

Kuzu is single-writer and the lock is process-wide: one read-write handle
blocks every other open, read-only ones included. So a worker that writes the
graph and keeps its handle makes every later search fail with "Could not set
lock on file" while sitting idle.

close() used to drop the Python references and leave the rest to refcounting.
That frees the handle only when the last reference goes, and inside a
long-lived worker one is easy to keep by accident.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from pipeline.graph.store import GraphStore


def opens_elsewhere(path: str) -> bool:
    """Whether a separate process can take the write lock."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import kuzu; kuzu.Database({path!r})"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode == 0


@pytest.fixture
def db_path():
    return str(Path(tempfile.mkdtemp(prefix="kuzu_lock_")) / "db")


def test_an_open_store_holds_the_lock(db_path):
    """The premise. Without this the next test proves nothing."""
    store = GraphStore(db_path)
    try:
        assert not opens_elsewhere(db_path)
    finally:
        store.close()


def test_close_releases_the_lock_while_the_store_object_survives(db_path):
    """The regression: the caller still holds the store, and the lock is gone.

    Merely dropping references cannot be tested this way -- the reference here
    keeps them alive, which is exactly the situation a worker gets into.
    """
    store = GraphStore(db_path)
    store.close()
    assert opens_elsewhere(db_path)


def test_close_is_idempotent(db_path):
    store = GraphStore(db_path)
    store.close()
    store.close()
    assert opens_elsewhere(db_path)


def test_the_context_manager_releases_it_too(db_path):
    with GraphStore(db_path):
        pass
    assert opens_elsewhere(db_path)
