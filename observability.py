"""Structured logging and in-process metrics.

``logger.info("fetch.ok", url=..., status=200, ms=31)`` gives an event *name*
and typed fields instead of an f-string. The name is stable enough to count and
alert on; the fields are queryable. With ``LOG_FORMAT=json`` each line is a JSON
object ready for any log shipper; with ``LOG_FORMAT=text`` the same call renders
as a readable ``fetch.ok url=... status=200 ms=31`` for local work.

A correlation id (the Celery task id, usually) is held in a
:class:`~contextvars.ContextVar`, so every line emitted while handling one job
carries it without any function having to thread it through its signature.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

_job_id: ContextVar[Optional[str]] = ContextVar("job_id", default=None)
_job_url: ContextVar[Optional[str]] = ContextVar("job_url", default=None)

_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {
    "message",
    "asctime",
    "taskName",
}


class StructuredAdapter(logging.LoggerAdapter):
    """Turns keyword arguments into structured record fields."""

    #: Arguments that belong to ``logging`` itself and must reach it untouched.
    _LOGGING_KWARGS = ("exc_info", "stack_info", "stacklevel")

    def process(self, msg: Any, kwargs: dict) -> tuple[Any, dict]:
        passthrough = {k: kwargs.pop(k) for k in self._LOGGING_KWARGS if k in kwargs}
        # Everything left is a user field. It goes under a single private
        # attribute so it can never collide with LogRecord internals.
        passthrough["extra"] = {"_fields": dict(kwargs)}
        return msg, passthrough


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        job = _job_id.get()
        if job:
            payload["job_id"] = job
        url = _job_url.get()
        if url:
            payload["url"] = url
        payload.update(getattr(record, "_fields", {}) or {})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = dict(getattr(record, "_fields", {}) or {})
        job = _job_id.get()
        head = f"{record.levelname:<7} [{record.name}]"
        if job:
            head += f" {{{job[:8]}}}"
        tail = " ".join(f"{k}={v}" for k, v in fields.items())
        line = f"{head} {record.getMessage()}"
        if tail:
            line = f"{line} — {tail}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def get_logger(name: str) -> StructuredAdapter:
    return StructuredAdapter(logging.getLogger(name), {})


_configured = False
_configure_lock = threading.Lock()


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Install the root handler. Idempotent — safe to call from every entrypoint."""
    global _configured
    with _configure_lock:
        if _configured:
            return
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter() if fmt.lower() == "json" else TextFormatter())
        root = logging.getLogger()
        root.handlers[:] = [handler]
        root.setLevel(level.upper())
        # These are chatty and say nothing we do not already log ourselves.
        for noisy in ("httpx", "httpcore", "urllib3", "pymongo", "charset_normalizer", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _configured = True


@contextmanager
def job_context(job_id: Optional[str], url: Optional[str] = None) -> Iterator[None]:
    """Tag every log line emitted inside the block with this job."""
    id_token = _job_id.set(job_id)
    url_token = _job_url.set(url)
    try:
        yield
    finally:
        _job_id.reset(id_token)
        _job_url.reset(url_token)


def current_job_id() -> Optional[str]:
    return _job_id.get()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


class Metrics:
    """Thread-safe counters and timers, scoped to one process.

    Deliberately not a Prometheus client: the numbers that matter here are
    per-run (how many pages hit tier 1, how many fell through to the LLM) and
    are reported in the :class:`~models.RunReport`. Anything longer-lived
    belongs in Mongo, where the records already are.
    """

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._timings: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def incr(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counts[name] += value

    def observe(self, name: str, seconds: float) -> None:
        with self._lock:
            self._timings.setdefault(name, []).append(seconds)

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - started)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            timings = {
                name: {
                    "count": len(values),
                    "total_ms": round(sum(values) * 1000, 1),
                    "avg_ms": round(sum(values) / len(values) * 1000, 1),
                    "max_ms": round(max(values) * 1000, 1),
                }
                for name, values in self._timings.items()
                if values
            }
            return {"counters": dict(self._counts), "timings": timings}

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()
            self._timings.clear()


metrics = Metrics()

__all__ = [
    "Metrics",
    "configure_logging",
    "current_job_id",
    "get_logger",
    "job_context",
    "metrics",
]
