"""Per-host circuit breaker.

A site that has started refusing you does not get better because you asked
30,000 more times. The breaker turns a stampede into one probe every few
minutes, which is both more effective and considerably more polite.

States: ``closed`` (normal) → ``open`` (after N consecutive failures) →
``half_open`` (one probe allowed after the cooldown) → back to ``closed`` after
enough consecutive successes, or straight back to ``open`` on a single further
failure.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from config import config
from errors import CircuitOpen
from observability import get_logger
from urls import registrable_host

log = get_logger("fetch.breaker")


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _Circuit:
    state: BreakerState = BreakerState.CLOSED
    failures: int = 0
    successes: int = 0
    opened_at: float = 0.0
    total_trips: int = 0


class CircuitBreaker:
    """Failure tracking for a set of hosts.

    Hosts marked via :meth:`record_block` never re-close. A CAPTCHA wall is a
    decision, not an outage, and retrying it after a cooldown is just a slower
    way of ignoring the answer.
    """

    def __init__(
        self,
        *,
        failure_threshold: Optional[int] = None,
        cooldown_seconds: Optional[float] = None,
        recovery_successes: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.failure_threshold = failure_threshold or config.BREAKER_FAILURE_THRESHOLD
        self.cooldown_seconds = (
            cooldown_seconds if cooldown_seconds is not None else config.BREAKER_COOLDOWN
        )
        self.recovery_successes = recovery_successes or config.BREAKER_RECOVERY_SUCCESSES
        self.enabled = config.BREAKER_ENABLED if enabled is None else enabled
        self._circuits: dict[str, _Circuit] = {}
        self._permanent: dict[str, str] = {}
        self._lock = threading.Lock()

    def check(self, url: str) -> None:
        """Raise :class:`~errors.CircuitOpen` if this host is cut off."""
        if not self.enabled:
            return
        host = registrable_host(url) or url

        with self._lock:
            if host in self._permanent:
                raise CircuitOpen(host, float("inf"))

            circuit = self._circuits.get(host)
            if circuit is None or circuit.state is BreakerState.CLOSED:
                return
            if circuit.state is BreakerState.HALF_OPEN:
                return  # a probe is allowed through

            remaining = self.cooldown_seconds - (time.monotonic() - circuit.opened_at)
            if remaining > 0:
                raise CircuitOpen(host, remaining)

            circuit.state = BreakerState.HALF_OPEN
            circuit.successes = 0
            log.info("breaker.half_open", host=host)

    def allows(self, url: str) -> bool:
        try:
            self.check(url)
        except CircuitOpen:
            return False
        return True

    def record_success(self, url: str) -> None:
        if not self.enabled:
            return
        host = registrable_host(url) or url
        with self._lock:
            circuit = self._circuits.get(host)
            if circuit is None:
                return
            if circuit.state is BreakerState.HALF_OPEN:
                circuit.successes += 1
                if circuit.successes >= self.recovery_successes:
                    circuit.state = BreakerState.CLOSED
                    circuit.failures = 0
                    log.info("breaker.closed", host=host)
            else:
                circuit.failures = 0

    def record_failure(self, url: str) -> None:
        if not self.enabled:
            return
        host = registrable_host(url) or url
        with self._lock:
            circuit = self._circuits.setdefault(host, _Circuit())

            # A failure during the probe means the host is still unhealthy.
            # Reopen at once rather than granting the full threshold again.
            if circuit.state is BreakerState.HALF_OPEN:
                self._trip(host, circuit)
                return

            circuit.failures += 1
            if circuit.failures >= self.failure_threshold:
                self._trip(host, circuit)

    def record_block(self, url: str, signal: str = "anti-bot control") -> None:
        """Mark a host permanently off-limits after a hard block."""
        host = registrable_host(url) or url
        with self._lock:
            self._permanent[host] = signal
        log.error("breaker.blocked_permanently", host=host, signal=signal)

    def _trip(self, host: str, circuit: _Circuit) -> None:
        """Open the circuit. Caller holds the lock."""
        circuit.state = BreakerState.OPEN
        circuit.opened_at = time.monotonic()
        circuit.total_trips += 1
        log.warning(
            "breaker.opened",
            host=host,
            failures=circuit.failures,
            cooldown=self.cooldown_seconds,
            trips=circuit.total_trips,
        )

    def state_of(self, url: str) -> BreakerState:
        host = registrable_host(url) or url
        with self._lock:
            if host in self._permanent:
                return BreakerState.OPEN
            circuit = self._circuits.get(host)
            return circuit.state if circuit else BreakerState.CLOSED

    @property
    def blocked_hosts(self) -> dict[str, str]:
        with self._lock:
            return dict(self._permanent)

    def stats(self) -> dict[str, dict[str, object]]:
        with self._lock:
            out: dict[str, dict[str, object]] = {
                host: {
                    "state": circuit.state.value,
                    "failures": circuit.failures,
                    "trips": circuit.total_trips,
                }
                for host, circuit in self._circuits.items()
            }
            for host, signal in self._permanent.items():
                out[host] = {"state": "blocked", "signal": signal, "permanent": True}
            return out

    def reset(self, url: Optional[str] = None) -> None:
        with self._lock:
            if url is None:
                self._circuits.clear()
                self._permanent.clear()
            else:
                host = registrable_host(url) or url
                self._circuits.pop(host, None)
                self._permanent.pop(host, None)


breaker = CircuitBreaker()

__all__ = ["BreakerState", "CircuitBreaker", "breaker"]
