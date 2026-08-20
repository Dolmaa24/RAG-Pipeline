"""Celery configuration — two queues, because one was the bottleneck.

With a single queue and ``-c 1``, one twenty-minute transcription blocks two
hundred quick page fetches behind it. The work has two completely different
shapes, so it gets two workers:

* **io** — fetching, HEAD checks, downloads. Network-bound, so the threads are
  almost always waiting; ``--pool=threads -c 16`` is right and costs almost
  nothing in memory.
* **cpu** — Whisper, the local LLM, headless Chromium, OCR. Each one wants a
  whole core or the GPU, so ``--pool=prefork -c 2`` keeps them from fighting.

The macOS trap this file exists to defuse: **Celery's prefork pool forks after
the Objective-C runtime has initialised, and the child then crashes or hangs**
the first time it touches a framework — which on this machine means Vision,
CoreML, or anything MLX. ``OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES`` must be set
*before* the fork, so it is set here at import rather than left to the shell,
where forgetting it costs an afternoon of confusing hangs.
"""

from __future__ import annotations

import os
import resource

# Must be set before the prefork pool forks. Importing this module is the
# earliest reliable point in both the worker and the API process.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
# Tokenizers' Rust parallelism also does not survive a fork.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from celery import Celery  # noqa: E402
from celery.signals import (  # noqa: E402
    task_failure,
    worker_process_init,
    worker_process_shutdown,
    worker_ready,
)

from config import config  # noqa: E402
from observability import configure_logging, get_logger  # noqa: E402

log = get_logger("celery")


def _raise_file_limit(target: int = 4096) -> None:
    """Sixteen concurrent fetchers plus Chromium exceed the default 256 on macOS."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(target, hard), hard))
    except (ValueError, OSError) as exc:  # pragma: no cover - platform dependent
        log.debug("celery.rlimit_failed", error=repr(exc))


_raise_file_limit()

celery_app = Celery(
    "universal_extractor",
    broker=config.REDIS_URL,
    backend=config.REDIS_URL,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_extended=True,
    # Results are polled by the client and the dashboard; a day is plenty and
    # keeps Redis from growing without bound.
    result_expires=86400,

    # --- routing ---------------------------------------------------------- #
    task_default_queue=config.IO_QUEUE,
    task_routes={
        "tasks.fetch_*": {"queue": config.IO_QUEUE},
        "tasks.discover_*": {"queue": config.IO_QUEUE},
        # Crawling is almost pure network waiting, so it belongs on io. The
        # files a crawl finds are routed to cpu explicitly at enqueue time.
        "tasks.crawl_*": {"queue": config.IO_QUEUE},
        "tasks.extract_url": {"queue": config.IO_QUEUE},
        "tasks.process_web_scrape": {"queue": config.IO_QUEUE},
        "tasks.extract_batch": {"queue": config.IO_QUEUE},
        "tasks.transcribe_*": {"queue": config.CPU_QUEUE},
        "tasks.extract_media": {"queue": config.CPU_QUEUE},
        "tasks.process_media_scrape": {"queue": config.CPU_QUEUE},
        "tasks.capture_livestream": {"queue": config.CPU_QUEUE},
        "tasks.render_*": {"queue": config.CPU_QUEUE},
        # Embedding is a transformer forward pass. On the io pool it would sit
        # in a thread that sixteen fetches are waiting behind.
        "tasks.index_*": {"queue": config.CPU_QUEUE},
        # Retrieval embeds the query and may load a cross-encoder. Same reason.
        "tasks.search": {"queue": config.CPU_QUEUE},
    },

    # --- limits ----------------------------------------------------------- #
    # A blocked fetch, a long transcript and a slow local model add up; cap it
    # so a wedged task cannot occupy a worker forever. The live-stream task
    # overrides these with its own, longer, limits.
    task_soft_time_limit=1800,
    task_time_limit=2100,
    # Whisper and Chromium both leak. Recycling bounds it — but not so often
    # that the per-worker model preload stops paying for itself.
    worker_max_tasks_per_child=25,
    worker_prefetch_multiplier=1,  # long tasks must not be hoarded by one worker
    task_acks_late=True,           # a killed worker's task is redelivered
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": 3600},
)


@worker_ready.connect
def _on_worker_ready(sender=None, **_kwargs) -> None:
    configure_logging(config.LOG_LEVEL, config.LOG_FORMAT)
    queues = sorted(queue.name for queue in (sender.app.amqp.queues.values() if sender else []))
    log.info("worker.ready", hostname=getattr(sender, "hostname", "?"), queues=queues)


@worker_process_init.connect
def _on_process_init(**_kwargs) -> None:
    """Load models once per process rather than once per task.

    Constructing a Whisper model costs seconds and hundreds of MB. Doing it
    inside every task, as the original pipeline did, meant paying that on every
    single job. Only the cpu worker preloads: the io worker never transcribes
    and should not carry the memory.
    """
    configure_logging(config.LOG_LEVEL, config.LOG_FORMAT)
    if not config.PRELOAD_MODELS:
        return
    if os.environ.get("CELERY_WORKER_QUEUE", config.CPU_QUEUE) != config.CPU_QUEUE:
        return
    try:
        from pipeline.transcribe import preload

        preload()
    except Exception as exc:  # a worker must start even without audio support
        log.warning("worker.preload_failed", error=repr(exc))

    # The embedding model is deliberately *not* preloaded here. Constructing it
    # takes about six seconds, and billiard gives a forked child four to report
    # UP before it kills it and starts another — which turns a preload into an
    # endless loop of children that are killed while still loading. It is loaded
    # on the first indexing task instead, and then held for the process's life,
    # which `worker_max_tasks_per_child` amortises over 25 tasks.


@worker_process_shutdown.connect
def _on_process_shutdown(**_kwargs) -> None:
    """Close the shared browser so Chromium processes do not outlive the worker."""
    try:
        from pipeline.fetch.browser import shutdown

        shutdown()
    except Exception:  # pragma: no cover - teardown best effort
        pass


@task_failure.connect
def _on_task_failure(sender=None, task_id=None, exception=None, args=None, **_kwargs) -> None:
    """Record exhausted tasks so nothing fails silently.

    ``task_failure`` fires after retries are exhausted, which is exactly when a
    job needs to become visible rather than disappear into a log line.
    """
    try:
        from database import CloudDatabase

        url = args[0] if args else "?"
        CloudDatabase().dead_letter(
            url=str(url),
            task=getattr(sender, "name", "unknown"),
            error=f"{type(exception).__name__}: {exception}",
            payload={"task_id": task_id},
        )
    except Exception:  # pragma: no cover - the failure path must not fail
        pass


__all__ = ["celery_app"]
