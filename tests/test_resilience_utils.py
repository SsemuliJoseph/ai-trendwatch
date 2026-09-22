"""
FILE: tests/test_resilience_utils.py
MODULE 15 tests — run with:  pytest tests/test_resilience_utils.py
Time is fake (injected sleep/clock/rng): the suite never really waits.
"""
import json
import os
from types import SimpleNamespace

import pytest
import requests

from app.utils.retry import (
    NO_RETRY,
    RetryPolicy,
    http_retry_after,
    is_transient_http_error,
    retry_call,
)
from app.utils.runlock import RunLock, RunLockError


class Flaky:
    """Fails with `errors` in order, then returns 'ok'."""

    def __init__(self, *errors):
        self.errors, self.calls = list(errors), 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def call(fn, policy=RetryPolicy(max_attempts=4, base_delay=1, max_delay=10, jitter=0), **kw):
    sleeps = []
    kw.setdefault("is_retryable", lambda e: isinstance(e, ConnectionError))
    result = retry_call(fn, policy, sleep=sleeps.append, rng=lambda: 0.0, **kw)
    return result, sleeps


# =====================================================================
# retry_call
# =====================================================================
def test_success_on_the_first_try_never_sleeps():
    assert call(Flaky()) == ("ok", [])


def test_backoff_doubles_between_attempts_and_is_capped():
    result, sleeps = call(Flaky(*[ConnectionError()] * 3), RetryPolicy(max_attempts=4, base_delay=3, max_delay=10, jitter=0))
    assert result == "ok" and sleeps == [3, 6, 10]  # 3, 6, then 12 capped to 10


def test_the_last_error_is_reraised_unchanged_after_max_attempts():
    fn = Flaky(*[ConnectionError(f"boom {i}") for i in range(5)])
    with pytest.raises(ConnectionError, match="boom 3"):
        call(fn, RetryPolicy(max_attempts=4, base_delay=0, jitter=0))
    assert fn.calls == 4


def test_non_retryable_errors_propagate_immediately():
    fn = Flaky(ValueError("bad input"))
    with pytest.raises(ValueError):
        call(fn)
    assert fn.calls == 1


def test_no_retry_policy_means_exactly_one_attempt():
    fn = Flaky(ConnectionError())
    with pytest.raises(ConnectionError):
        call(fn, NO_RETRY)
    assert fn.calls == 1


def test_jitter_adds_up_to_the_configured_fraction():
    _, sleeps = call(Flaky(ConnectionError()), RetryPolicy(max_attempts=2, base_delay=4, jitter=0.5))  # rng = 0
    assert sleeps == [4]
    out = []
    retry_call(Flaky(ConnectionError()), RetryPolicy(max_attempts=2, base_delay=4, jitter=0.5),
               is_retryable=lambda e: True, sleep=out.append, rng=lambda: 0.5)
    assert out == [5.0]  # 4 * (1 + 0.5 * 0.5)


def test_server_hint_overrides_a_shorter_backoff():
    fn = Flaky(ConnectionError())
    _, sleeps = call(fn, retry_after=lambda e: 7.0)
    assert sleeps == [7.0]


def test_a_hint_longer_than_max_delay_is_not_waited_for():
    fn = Flaky(ConnectionError(), ConnectionError())
    with pytest.raises(ConnectionError):
        call(fn, RetryPolicy(max_attempts=4, base_delay=1, max_delay=10, jitter=0), retry_after=lambda e: 120.0)
    assert fn.calls == 1


def test_on_retry_receives_attempt_error_and_delay():
    seen = []
    call(Flaky(ConnectionError("x")), on_retry=lambda attempt, exc, delay: seen.append((attempt, str(exc), delay)))
    assert seen == [(1, "x", 1)]


def test_keyboard_interrupt_is_never_swallowed():
    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        call(interrupted, is_retryable=lambda e: True)


@pytest.mark.parametrize("kw", [dict(max_attempts=0), dict(base_delay=-1), dict(jitter=-0.1)])
def test_invalid_policies_are_rejected(kw):
    with pytest.raises(ValueError):
        RetryPolicy(**kw)


# =====================================================================
# HTTP helpers
# =====================================================================
def http_error(status, headers=None):
    response = requests.Response()
    response.status_code = status
    response.headers.update(headers or {})
    return requests.HTTPError(response=response)


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retryable(status):
    assert is_transient_http_error(http_error(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422])
def test_client_errors_are_not_retryable(status):
    assert is_transient_http_error(http_error(status)) is False


def test_network_failures_are_retryable_but_other_request_errors_are_not():
    assert is_transient_http_error(requests.ConnectionError())
    assert is_transient_http_error(requests.ConnectTimeout())
    assert is_transient_http_error(requests.ReadTimeout())
    assert not is_transient_http_error(requests.exceptions.InvalidURL())
    assert not is_transient_http_error(requests.exceptions.TooManyRedirects())
    assert not is_transient_http_error(ValueError())
    assert not is_transient_http_error(requests.HTTPError())  # no response attached


def test_retry_after_header_parsing():
    assert http_retry_after(http_error(429, {"Retry-After": "12"})) == 12.0
    assert http_retry_after(http_error(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})) is None
    assert http_retry_after(http_error(429, {"Retry-After": "-5"})) is None
    assert http_retry_after(http_error(503)) is None
    assert http_retry_after(requests.ConnectionError()) is None


# =====================================================================
# RunLock
# =====================================================================
def make_lock(tmp_path, **kw):
    return RunLock(tmp_path / "data" / ".run.lock", **kw)


def test_acquire_creates_the_lock_and_release_removes_it(tmp_path):
    lock = make_lock(tmp_path, pid=lambda: 4242)
    lock.acquire()
    path = tmp_path / "data" / ".run.lock"
    assert json.loads(path.read_text())["pid"] == 4242
    lock.release()
    assert not path.exists()


def test_a_second_run_is_refused_while_the_first_is_active(tmp_path):
    first = make_lock(tmp_path, pid=lambda: 1)
    first.acquire()
    second = make_lock(tmp_path, pid=lambda: 2)
    with pytest.raises(RunLockError, match="another run is active.*pid 1"):
        second.acquire()
    first.release()
    second.acquire()  # free again
    second.release()


def test_a_stale_lock_from_a_crashed_run_is_taken_over(tmp_path):
    crashed = make_lock(tmp_path, pid=lambda: 1)
    crashed.acquire()  # ...and never released
    path = tmp_path / "data" / ".run.lock"
    old = path.stat().st_mtime - 2 * 3600
    os.utime(path, (old, old))

    new = make_lock(tmp_path, pid=lambda: 2, stale_after=30 * 60)
    new.acquire()
    assert json.loads(path.read_text())["pid"] == 2
    new.release()


def test_a_lock_exactly_at_the_limit_is_still_valid(tmp_path):
    clock = SimpleNamespace(now=1_000_000.0)
    first = make_lock(tmp_path, clock=lambda: clock.now, pid=lambda: 1, stale_after=100)
    first.acquire()
    path = tmp_path / "data" / ".run.lock"
    os.utime(path, (clock.now, clock.now))
    clock.now += 100  # age == stale_after: not stale yet
    with pytest.raises(RunLockError):
        make_lock(tmp_path, clock=lambda: clock.now, pid=lambda: 2, stale_after=100).acquire()
    clock.now += 1
    make_lock(tmp_path, clock=lambda: clock.now, pid=lambda: 2, stale_after=100).acquire()


def test_release_does_not_delete_a_lock_owned_by_someone_else(tmp_path):
    ours = make_lock(tmp_path, pid=lambda: 1)
    ours.acquire()
    path = tmp_path / "data" / ".run.lock"
    path.write_text(json.dumps({"pid": 999, "started_at": 0}))  # taken over by another process
    ours.release()
    assert path.exists()


def test_release_without_acquire_and_double_release_are_harmless(tmp_path):
    lock = make_lock(tmp_path)
    lock.release()
    lock.acquire()
    lock.release()
    lock.release()


def test_context_manager_releases_even_when_the_body_raises(tmp_path):
    path = tmp_path / "data" / ".run.lock"
    with pytest.raises(RuntimeError):
        with make_lock(tmp_path):
            assert path.exists()
            raise RuntimeError("run crashed")
    assert not path.exists()


def test_a_corrupt_lock_file_still_blocks_until_it_is_stale(tmp_path):
    path = tmp_path / "data" / ".run.lock"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    with pytest.raises(RunLockError, match="unknown holder"):
        make_lock(tmp_path).acquire()
