"""Drift: noticing the day a site changed and the pipeline stopped noticing.

The characteristic failure of an extraction pipeline is not a crash. It is a
column that quietly goes empty. The selectors still run, the model still
answers, the rows still arrive — and one field has been null for three weeks
because a site moved its price into a different element.

So fill rate is tracked per (domain, schema, field). When a field that has been
filling reliably drops below its established baseline by more than
``DRIFT_FILL_RATE_DROP``, that is an alert, not a statistic. The same signal
retires a stale selector spec in :mod:`pipeline.extract.selectors`.

Baselines need enough history to mean anything, hence ``DRIFT_MIN_SAMPLES``:
before that, a run of three pages that happen to lack a price is not evidence.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import config
from observability import get_logger
from urls import registrable_host

log = get_logger("trust.drift")


@dataclass(slots=True)
class FieldStats:
    seen: int = 0
    filled: int = 0

    @property
    def fill_rate(self) -> float:
        return self.filled / self.seen if self.seen else 0.0


@dataclass(slots=True)
class DriftAlert:
    domain: str
    field: str
    baseline: float
    current: float
    samples: int

    @property
    def drop(self) -> float:
        return round(self.baseline - self.current, 3)

    def message(self) -> str:
        return (
            f"{self.domain}: field {self.field!r} fill rate fell from "
            f"{self.baseline:.0%} to {self.current:.0%} over {self.samples} records"
        )


@dataclass
class DriftMonitor:
    """Per-domain field fill rates, with baselines and alerting.

    In-process by default. With a database it also persists the counters, so a
    baseline survives worker recycling — which matters, because
    ``worker_max_tasks_per_child`` means a worker's memory is deliberately
    short-lived.
    """

    database: object = None
    min_samples: int = field(default_factory=lambda: config.DRIFT_MIN_SAMPLES)
    max_drop: float = field(default_factory=lambda: config.DRIFT_FILL_RATE_DROP)

    _current: dict[str, dict[str, FieldStats]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(FieldStats)), init=False
    )
    _baseline: dict[str, dict[str, float]] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    # ------------------------------------------------------------------ #

    def observe(self, url: str, schema_hash: str, record: dict) -> list[DriftAlert]:
        """Record one extraction. Returns any alerts it triggered."""
        if not config.DRIFT_ENABLED or not record:
            return []

        key = f"{registrable_host(url)}|{schema_hash}"
        alerts: list[DriftAlert] = []

        with self._lock:
            stats = self._current[key]
            for name, value in record.items():
                entry = stats[name]
                entry.seen += 1
                if not _is_empty(value):
                    entry.filled += 1

            baseline = self._baseline.get(key)
            if baseline is None:
                if min((entry.seen for entry in stats.values()), default=0) >= self.min_samples:
                    self._baseline[key] = {name: entry.fill_rate for name, entry in stats.items()}
                    log.info("drift.baseline_set", key=key, fields=len(stats))
                    self._persist(key, self._baseline[key])
                return []

            for name, entry in stats.items():
                if entry.seen < self.min_samples:
                    continue
                expected = baseline.get(name)
                if expected is None or expected < 0.5:
                    # A field that was rarely filled to begin with has no
                    # baseline worth alerting against.
                    continue
                if expected - entry.fill_rate > self.max_drop:
                    alerts.append(
                        DriftAlert(
                            domain=key.split("|")[0],
                            field=name,
                            baseline=round(expected, 3),
                            current=round(entry.fill_rate, 3),
                            samples=entry.seen,
                        )
                    )

        for alert in alerts:
            log.error("drift.detected", domain=alert.domain, field=alert.field,
                      baseline=alert.baseline, current=alert.current, drop=alert.drop)
        return alerts

    # ------------------------------------------------------------------ #

    def report(self, url: str = "", schema_hash: str = "") -> dict:
        with self._lock:
            if url:
                key = f"{registrable_host(url)}|{schema_hash}"
                stats = self._current.get(key, {})
                return {
                    "key": key,
                    "baseline": self._baseline.get(key, {}),
                    "current": {name: round(entry.fill_rate, 3) for name, entry in stats.items()},
                }
            return {
                key: {name: round(entry.fill_rate, 3) for name, entry in stats.items()}
                for key, stats in self._current.items()
            }

    def _persist(self, key: str, baseline: dict[str, float]) -> None:
        if self.database is None:
            return
        try:
            self.database.runs_collection().database["drift_baselines"].replace_one(
                {"_id": key},
                {"_id": key, "baseline": baseline, "updated_at": datetime.now(timezone.utc)},
                upsert=True,
            )
        except Exception as exc:
            log.debug("drift.persist_failed", error=repr(exc))

    def load_baseline(self, url: str, schema_hash: str) -> Optional[dict[str, float]]:
        if self.database is None:
            return None
        key = f"{registrable_host(url)}|{schema_hash}"
        try:
            doc = self.database.runs_collection().database["drift_baselines"].find_one({"_id": key})
        except Exception:
            return None
        if not doc:
            return None
        with self._lock:
            self._baseline[key] = doc["baseline"]
        return doc["baseline"]

    def reset(self) -> None:
        with self._lock:
            self._current.clear()
            self._baseline.clear()


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return not value
    return False


__all__ = ["DriftAlert", "DriftMonitor", "FieldStats"]
