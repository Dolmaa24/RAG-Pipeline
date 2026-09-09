"""Restart a worker when its source changes.

Celery dropped ``--autoreload`` in 4.x and never replaced it, so a worker runs
whatever it imported at startup for as long as it lives. Editing a task and
watching the old one run is the most expensive confusion this project has
produced: it presents as a code bug, and the code is fine.

**Why this exists rather than ``watchmedo``.** watchdog matches its
``--ignore-patterns`` with :meth:`pathlib.PurePath.match`, which matches from
the *right* and does not let ``*`` cross a separator. So ``*/venv/*`` matches
``/proj/venv/foo.py`` and not ``/proj/venv/lib/site-packages/foo.py``, and no
spelling of that pattern -- ``**/venv/**`` included -- excludes a directory's
whole subtree. Measured: every one of five candidate patterns excluded nothing.

That matters here more than it would elsewhere, because this repository writes
Python *as data*. A build generates modules into ``workspace/``, and a watcher
that noticed would restart the worker that was writing them, mid-build, for
ever. So the ignore rule is a check on path components, which says what it
means and can be tested.

Not for production. Nothing imports this; it is spawned by ``run.sh --reload``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

#: Directory names that never contain source worth reloading for. Checked as
#: path *components*, not as globs, because a glob cannot express this.
IGNORED_PARTS = frozenset({
    "venv", ".venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", "output", "lance_data", "kuzu_db", "workspace", ".ruff_cache",
})

#: Where source lives. Watched recursively; everything else is not watched at
#: all, which is cheaper and less surprising than watching all of it and
#: filtering afterwards.
WATCHED = ("pipeline", "playground", "config", "bench")

#: Root-level modules. The repository keeps most of its entry points here, and
#: tasks.py is the file most likely to be edited while a worker is running.
WATCH_ROOT_FILES = True

#: Changes inside this window count as one restart. Saving a file often emits
#: several events, and an editor writing atomically emits a create and a move.
DEBOUNCE_SECONDS = 1.0


def is_source(path: str) -> bool:
    """Whether a change to ``path`` should restart the worker."""
    candidate = Path(path)
    if candidate.suffix != ".py":
        return False
    if candidate.name.startswith("."):
        return False
    try:
        parts = candidate.resolve().relative_to(ROOT).parts
    except (ValueError, OSError):
        parts = candidate.parts
    return not any(part in IGNORED_PARTS for part in parts)


class Runner:
    """The child process, restarted on demand."""

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            # Its own process group, so a restart can signal the worker and
            # every child it forked -- the prefork pool leaves orphans
            # otherwise, and they hold the queue open.
            self.process = subprocess.Popen(self.command, start_new_session=True)

    def stop(self, grace: float = 10.0) -> None:
        with self._lock:
            process = self.process
            self.process = None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            # A worker in the middle of a model call will not stop politely,
            # and waiting for it defeats the point of a reload.
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def restart(self, why: str) -> None:
        print(f"\n[devwatch] {why} — restarting\n", flush=True)
        self.stop()
        self.start()


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: devwatch.py -- <command>", file=sys.stderr)
        return 2

    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print(
            "--reload needs watchdog:  ./venv/bin/pip install watchdog",
            file=sys.stderr,
        )
        return 1

    runner = Runner(argv)
    pending: list[float] = []
    changed: list[str] = []
    lock = threading.Lock()

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event) -> None:
            if event.is_directory:
                return
            for path in (event.src_path, getattr(event, "dest_path", "")):
                if path and is_source(path):
                    with lock:
                        pending.append(time.monotonic())
                        changed.append(Path(path).name)
                    return

    observer = Observer()
    handler = Handler()
    for name in WATCHED:
        directory = ROOT / name
        if directory.is_dir():
            observer.schedule(handler, str(directory), recursive=True)
    if WATCH_ROOT_FILES:
        # Not recursive: the repository root also holds lance_data, kuzu_db and
        # workspace, and watching those is the restart loop this file exists to
        # avoid.
        observer.schedule(handler, str(ROOT), recursive=False)

    observer.start()
    runner.start()

    stopping = threading.Event()

    def shut_down(*_) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, shut_down)
    signal.signal(signal.SIGTERM, shut_down)

    try:
        while not stopping.is_set():
            time.sleep(0.25)
            with lock:
                due = pending and (time.monotonic() - pending[-1]) >= DEBOUNCE_SECONDS
                names = sorted(set(changed))
                if due:
                    pending.clear()
                    changed.clear()
            if due:
                runner.restart(", ".join(names[:4]))
            process = runner.process
            if process is not None and process.poll() is not None:
                # The worker exited on its own — a crash, or Ctrl-C reaching it
                # first. Follow it out rather than restarting a thing that has
                # decided to stop.
                return process.returncode or 0
    finally:
        observer.stop()
        observer.join(timeout=5)
        runner.stop()
    return 0


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    sys.exit(main(arguments))
