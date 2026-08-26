"""Seeding a traversal without paying for a model that is not needed.

`graph_neighbors` was measured at 13.1 seconds inside an investigation, and the
guess in the plan was that Kuzu opened per call. It did — for 0.05s. The cost
was one line in :meth:`EntityIndex.seeds`, which embeds the query to find
entities by meaning, and embedding one short string loads BGE the first time:

    graph_exists()          0.00s
    EntityIndex()           1.08s
    .seeds()               10.60s
    GraphStore open         0.05s
    .neighbours()           0.03s

Two things follow. A caller that already knows an entity's real name — which is
the usual case, because the name came out of a previous traversal — should not
need the model at all. And the semantic path is a cold start rather than a
per-call cost, since the embedder is a process-wide singleton, so exactly one
run per worker pays it and it is always the first one someone is watching.
"""

from __future__ import annotations

import pytest

import celery_app as celery_module
from config import config
from pipeline.graph.entities import EntityIndex


class FakeTable:
    """Just enough LanceDB to answer both paths."""

    def __init__(self, names):
        self.names = names
        self.where_clauses: list[str] = []
        self.vector_searches = 0

    def search(self, vector=None, vector_column_name=None):
        if vector is not None:
            self.vector_searches += 1
            return _Query(self, [{"name": n} for n in self.names])
        return _Query(self, None)


class _Query:
    def __init__(self, table, rows):
        self._table = table
        self._rows = rows
        self._where = None

    def where(self, clause):
        self._table.where_clauses.append(clause)
        self._where = clause
        return self

    def select(self, _columns):
        return self

    def limit(self, _n):
        return self

    def to_list(self):
        if self._rows is not None:
            return self._rows
        # Undo SQL quoting the way a database would, doubled quotes included —
        # a fake that only handles unquoted names would pass while the real
        # filter was broken on any name containing an apostrophe.
        literal = self._where.split("= ", 1)[1].strip()
        wanted = literal[1:-1].replace("''", "'")
        return [{"name": n} for n in self._table.names if n.lower() == wanted]


class ExplodingEmbedder:
    """Touching this is the failure the fast path exists to prevent."""

    def embed_query(self, text):
        raise AssertionError("the embedder was loaded for an exact name")


@pytest.fixture
def index():
    idx = EntityIndex.__new__(EntityIndex)  # no lancedb.connect
    idx._embedder = ExplodingEmbedder()
    idx._table = FakeTable(["Acme Corporation", "Beta Industries", "ACME"])
    idx._lock = None
    return idx


def test_an_exact_name_never_reaches_the_embedder(index):
    assert index.seeds("Acme Corporation") == ["Acme Corporation"]
    assert index._table.vector_searches == 0


def test_matching_ignores_case(index):
    assert index.seeds("acme corporation") == ["Acme Corporation"]
    assert index.seeds("  ACME  ") == ["ACME"]
    assert index._table.vector_searches == 0


def test_an_exact_match_seeds_only_that_entity(index):
    """Not only faster — more precise.

    Asked for "Acme Corporation", the vector search returned
    ['ACME', 'Acme Corporation', 'Beta Industries'], seeding a traversal from a
    company that merely appears nearby.
    """
    assert index.seeds("Acme Corporation") == ["Acme Corporation"]


def test_a_name_with_a_quote_does_not_break_the_filter(index):
    # The clause is built by string interpolation, so the escaping matters —
    # both for correctness and because a name is a value from the model.
    index._table.names.append("O'Brien Holdings")
    assert index.seeds("O'Brien Holdings") == ["O'Brien Holdings"]
    assert "'o''brien holdings'" in index._table.where_clauses[-1]


def test_a_partial_name_still_goes_to_the_model(index):
    # The fast path is an addition, not a replacement: "the guy who ran Apple's
    # car project" has to keep working.
    index._embedder = _Embedder()
    assert index.seeds("the company that bought a sensor maker")
    assert index._table.vector_searches == 1


def test_an_empty_query_asks_nothing(index):
    assert index.seeds("   ") == []
    assert index._table.vector_searches == 0


def test_a_failing_exact_lookup_falls_through_rather_than_failing(index):
    class Broken(FakeTable):
        def search(self, vector=None, vector_column_name=None):
            if vector is None:
                raise RuntimeError("scalar index missing")
            return super().search(vector, vector_column_name)

    index._table = Broken(["Acme Corporation"])
    index._embedder = _Embedder()

    assert index.seeds("Acme Corporation") == ["Acme Corporation"]
    assert index._table.vector_searches == 1


class _Embedder:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


def test_the_agents_worker_warms_and_others_do_not(monkeypatch):
    warmed: list[bool] = []
    monkeypatch.setattr(celery_module, "_warm_agents_worker", lambda: warmed.append(True))

    class Sender:
        hostname = "test@host"

        class app:
            class amqp:
                queues: dict = {}

    def ready(queues):
        Sender.app.amqp.queues = {n: type("Q", (), {"name": n})() for n in queues}
        celery_module._on_worker_ready(sender=Sender)

    ready([config.IO_QUEUE])
    ready([config.CPU_QUEUE])
    assert warmed == []

    ready([config.AGENTS_QUEUE])
    assert warmed == [True]


def test_warming_is_off_when_the_setting_is(monkeypatch):
    monkeypatch.setattr(celery_module.config, "AGENT_WARM_EMBEDDER", False)
    started: list[str] = []
    monkeypatch.setattr(
        celery_module.threading, "Thread",
        lambda **kwargs: started.append(kwargs.get("name")) or _NoThread(),
    )

    celery_module._warm_agents_worker()
    assert started == []


def test_warming_happens_off_the_ready_path(monkeypatch):
    """A daemon thread, so a ten-second load delays no task and blocks no
    shutdown. Preloading in worker_process_init is what billiard's four-second
    handshake kills; this runs after the worker has already reported itself."""
    created: dict = {}

    class Recorder:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def start(self):
            created["started"] = True

    monkeypatch.setattr(celery_module.config, "AGENT_WARM_EMBEDDER", True)
    monkeypatch.setattr(celery_module.threading, "Thread", Recorder)

    celery_module._warm_agents_worker()

    assert created["daemon"] is True
    assert created["started"] is True


def test_a_failed_warm_up_does_not_raise(monkeypatch):
    # The worst case must be the behaviour we already had: loaded on demand.
    monkeypatch.setattr(celery_module.config, "AGENT_WARM_EMBEDDER", True)
    captured = {}

    class Immediate:
        def __init__(self, target=None, **kwargs):
            captured["target"] = target

        def start(self):
            captured["target"]()

    monkeypatch.setattr(celery_module.threading, "Thread", Immediate)
    import pipeline.embed.dense as dense

    monkeypatch.setattr(
        dense, "get_dense_embedder", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model"))
    )

    celery_module._warm_agents_worker()  # must not raise


class _NoThread:
    def start(self):
        raise AssertionError("a thread was started while warming was disabled")


def test_the_agents_worker_stays_on_the_threads_pool():
    """Warming only lasts because this worker never recycles.

    ``max_tasks_per_child`` is implemented by the prefork pool alone, and the
    threads pool ignores it — so the embedder loaded at startup is held for the
    worker's life. On prefork it would break twice: each forked child needs its
    own copy, and loading one is exactly what billiard's four-second UP
    handshake kills. That makes the pool choice load-bearing rather than a
    preference, so it is pinned here.
    """
    import inspect
    from pathlib import Path

    from celery.concurrency.base import BasePool
    from celery.concurrency.thread import TaskPool

    assert "max_tasks_per_child" not in inspect.getsource(BasePool.__init__)
    assert "max_tasks_per_child" not in inspect.getsource(inspect.getmodule(TaskPool))

    launcher = Path(__file__).resolve().parent.parent / "run.sh"
    agents_command = next(
        line for line in launcher.read_text().splitlines()
        if "--queues=agents" in line
    )
    assert "--pool=threads" in agents_command
