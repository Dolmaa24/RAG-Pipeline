"""Rate-limit state that has to hold across processes, not merely across threads.

The defect these cover: the limiter was built on ``threading`` primitives, so
each prefork child had its own token bucket. Measured with two processes and a
configured 2 requests per second, the gaps alternated ``0.001, 0.501, 0.001`` --
both processes firing together, then both sleeping. Twice the configured rate,
arriving in pairs, which is the shape a bot detector is looking for.

These talk to a real Redis rather than a fake. The whole point is the behaviour
of an atomic script under concurrent callers, and a fake that serialises calls
would pass while proving nothing.
"""

from __future__ import annotations

import time

import pytest

from pipeline.fetch import shared

HOST = "ratelimit-test.invalid"


@pytest.fixture
def state():
    backend = shared.connect()
    if backend is None:
        pytest.skip("Redis is not reachable")
    backend.clear(HOST)
    yield backend
    backend.clear(HOST)


# --------------------------------------------------------------------------- #
# The bucket
# --------------------------------------------------------------------------- #


def test_a_full_bucket_admits_the_first_request_immediately(state):
    assert state.take(HOST, rate=10.0, capacity=1.0) == 0.0


def test_the_next_request_is_told_how_long_to_wait(state):
    state.take(HOST, rate=2.0, capacity=1.0)
    wait = state.take(HOST, rate=2.0, capacity=1.0)
    assert 0.3 < wait <= 0.5, wait


def test_a_shortfall_takes_nothing(state):
    """The caller sleeps and asks again, so a refused take must not deduct.

    Deducting on refusal would let several waiters each remove a token they
    never received, and the host would go quiet for multiples of the real wait.
    """
    state.take(HOST, rate=1.0, capacity=1.0)
    first = state.take(HOST, rate=1.0, capacity=1.0)
    second = state.take(HOST, rate=1.0, capacity=1.0)
    assert second == pytest.approx(first, abs=0.05)


def test_tokens_refill_over_time(state):
    state.take(HOST, rate=20.0, capacity=1.0)
    time.sleep(0.2)
    assert state.take(HOST, rate=20.0, capacity=1.0) == 0.0


def test_a_second_connection_sees_the_same_bucket(state):
    """Two processes, one budget -- the whole reason this module exists."""
    other = shared.connect()
    state.take(HOST, rate=2.0, capacity=1.0)
    assert other.take(HOST, rate=2.0, capacity=1.0) > 0.0


# --------------------------------------------------------------------------- #
# Concurrency leases
# --------------------------------------------------------------------------- #


def test_the_cap_is_enforced(state):
    first = state.try_acquire(HOST, 1)
    assert first is not None
    assert state.try_acquire(HOST, 1) is None


def test_releasing_frees_the_slot(state):
    lease = state.try_acquire(HOST, 1)
    state.release(HOST, lease)
    assert state.try_acquire(HOST, 1) is not None


def test_the_cap_counts_across_connections(state):
    other = shared.connect()
    assert state.try_acquire(HOST, 2) is not None
    assert other.try_acquire(HOST, 2) is not None
    assert other.try_acquire(HOST, 2) is None


def test_leases_are_distinct_so_one_release_frees_one_slot(state):
    a = state.try_acquire(HOST, 2)
    b = state.try_acquire(HOST, 2)
    assert a != b
    state.release(HOST, a)
    assert state.try_acquire(HOST, 2) is not None
    assert state.try_acquire(HOST, 2) is None


def test_an_expired_lease_is_swept(state, monkeypatch):
    """A worker killed mid-request cannot release anything.

    Without expiry the count only ever rises, and the host becomes permanently
    unreachable -- a worse outage than the one the cap prevents.
    """
    monkeypatch.setattr(shared, "_LEASE_SECONDS", 0.3)
    backend = shared.SharedLimiterState(state._redis)
    assert backend.try_acquire(HOST, 1) is not None
    assert backend.try_acquire(HOST, 1) is None  # still held
    time.sleep(0.4)
    assert backend.try_acquire(HOST, 1) is not None  # swept


# --------------------------------------------------------------------------- #
# Penalties
# --------------------------------------------------------------------------- #


def test_a_penalty_is_visible_to_every_worker(state):
    """Retry-After arrives on one response, in one process, and binds all."""
    other = shared.connect()
    state.penalize(HOST, 5.0)
    assert other.penalty_remaining(HOST) > 4.0


def test_no_penalty_reads_as_zero(state):
    assert state.penalty_remaining(HOST) == 0.0


def test_a_longer_penalty_extends_and_a_shorter_one_does_not_shorten(state):
    state.penalize(HOST, 10.0)
    state.penalize(HOST, 2.0)
    assert state.penalty_remaining(HOST) > 9.0


# --------------------------------------------------------------------------- #
# Degrading rather than refusing
# --------------------------------------------------------------------------- #


def test_the_limiter_runs_without_redis(monkeypatch):
    """A pipeline that cannot reach Redis has no queue either. Fetching must
    still work, on per-process limits, rather than failing closed."""
    monkeypatch.setattr(shared, "connect", lambda: None)
    from pipeline.fetch.ratelimit import HostRateLimiter

    limiter = HostRateLimiter(requests_per_second=50.0, burst=2, max_concurrency_per_host=1)
    assert limiter._shared is None
    with limiter.slot("https://example.test/a") as waited:
        assert waited >= 0.0


def test_the_switch_turns_sharing_off(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "RATELIMIT_SHARED", False)
    assert shared.connect() is None
