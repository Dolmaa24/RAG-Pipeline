"""Which file changes restart a worker, and which must not.

Celery has had no --autoreload since 4.x, so a worker runs whatever it imported
at startup. That is the confusion this repo has paid most for: an edited task,
an unchanged worker, and an error that looks like a code bug.

The negative cases matter more than the positive ones here, and one of them is
close to catastrophic. **This repository writes Python as data** — a build
generates modules into ``workspace/`` — so a watcher that treated those as
source would restart the very worker writing them, mid-build, for ever.

watchdog's own ``--ignore-patterns`` cannot express any of this. It matches
with ``PurePath.match``, which matches from the right and will not let ``*``
cross a separator, so ``*/workspace/*`` matches ``/proj/workspace/a.py`` and
not ``/proj/workspace/b/a.py``. Measured against five candidate spellings,
including ``**/workspace/**``: none excluded anything. Hence a component check
rather than a glob, which says what it means and can be tested — this file.
"""

from __future__ import annotations

import pytest

import devwatch


@pytest.mark.parametrize(
    "path",
    [
        "pipeline/agents/loop.py",
        "pipeline/skills/match.py",
        "playground/store.py",
        "tasks.py",
        "app.py",
        "config.py",
    ],
)
def test_editing_source_restarts_the_worker(path):
    assert devwatch.is_source(path)


@pytest.mark.parametrize(
    "path, why",
    [
        ("workspace/baristo/orders.py", "a build writes this while running"),
        ("workspace/b/pkg/sub/deep.py", "and nests it arbitrarily deep"),
        ("venv/lib/python3.13/site-packages/foo.py", "installed packages never change"),
        ("pipeline/__pycache__/loop.py", "a cache, not a source file"),
        (".git/hooks/pre-commit.py", "git's business"),
        ("lance_data/chunks.lance/data.py", "the vector store's directory"),
        ("kuzu_db/x.py", "the graph store's directory"),
        ("output/dump.py", "run artefacts"),
        (".pytest_cache/v/x.py", "a cache"),
    ],
)
def test_these_never_restart_the_worker(path, why):
    assert not devwatch.is_source(path), why


@pytest.mark.parametrize(
    "path",
    [
        "skills/health/SKILL.md",
        "README.md",
        "requirements.txt",
        "output/extractions.jsonl",
        "lance_data/chunks.lance/data.lance",
        "playground.db",
    ],
)
def test_non_python_files_never_restart_the_worker(path):
    """Skills are the interesting case: they already hot-reload on their own,
    because the loader caches on mtime and notices an edited file. Restarting a
    worker for one would be a slower way to get the same result."""
    assert not devwatch.is_source(path)


def test_a_build_writing_a_module_does_not_look_like_source():
    """The whole reason this is a component check and not a glob.

    Written as the real thing rather than a string: a build creates the
    directory and the file, and both must be invisible to the watcher.
    """
    from pathlib import Path

    generated = devwatch.ROOT / "workspace" / "probe" / "orders.py"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text("x = 1\n")
    try:
        assert not devwatch.is_source(str(generated))
        assert not devwatch.is_source(str(generated.resolve()))
    finally:
        generated.unlink()
        generated.parent.rmdir()


def test_the_watched_directories_all_exist():
    """A typo here is silent: the watch is simply never scheduled."""
    present = [name for name in devwatch.WATCHED if (devwatch.ROOT / name).is_dir()]
    assert "pipeline" in present
    assert "playground" in present


def test_the_ignore_list_covers_every_directory_the_pipeline_writes_to():
    """Each of these is written to while a job runs. Any one of them missing is
    a restart loop under load rather than a wrong answer, which is worse: it
    only shows up when the system is busy."""
    from config import config

    for path in (config.BUILD_WORKSPACE, config.KUZU_DB_PATH, "lance_data", "output"):
        name = str(path).strip("./").split("/")[0]
        assert name in devwatch.IGNORED_PARTS, name
