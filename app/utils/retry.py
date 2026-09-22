"""
FILE: app/utils/retry.py
MODULE 15 — retry with exponential backoff (cross-cutting helper).

ARCHITECTURE NOTES
  * `utils/` is the cross-cutting layer: pure mechanics, no knowledge of Gemini, WhatsApp, SQLite
    or the pipeline. Callers say WHAT is retryable (`is_retryable`); this module only decides WHEN.
  * WHEN TO RETRY (the rule this project follows):
        idempotent reads (RSS/arXiv GET, Gemini analysis)   -> yes, bounded
        WhatsApp SEND                                       -> NEVER (a lost reply may still be a
                                                               posted message; see openwa_client.py)
  * BACKOFF: delay = base * 2^(attempt-1), capped at `max_delay`, plus random jitter so many clients
    do not retry in lock-step. If the server says how long to wait (Retry-After), that hint wins,
    but a hint LONGER than `max_delay` means "not worth waiting for in a short-lived run": the error is
    raised and the next scheduled run tries again.
  * `max_attempts=1` (NO_RETRY) is the default everywhere, so retrying is always an explicit opt-in
    made in one place (main.py builds the policy from Settings).
  * Clock-free and deterministic under test: `sleep` and `rng` are injectable.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import requests

T = TypeVar("T")

TRANSIENT_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3  # total tries, including the first; 1 = never retry
    base_delay: float = 2.0  # seconds before the 2nd attempt
    max_delay: float = 60.0  # cap for both the computed backoff and any server hint
    jitter: float = 0.25  # up to +25% random extra delay

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0 or self.jitter < 0:
            raise ValueError("delays and jitter must be >= 0")

    def delay_for(self, attempt: int, rng: Callable[[], float] = random.random) -> float:
        """Delay AFTER failed attempt number `attempt` (1-based)."""
        backoff = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return backoff * (1 + self.jitter * rng())


NO_RETRY = RetryPolicy(max_attempts=1)


def retry_call(
    fn: Callable[[], T],
    policy: RetryPolicy,
    *,
    is_retryable: Callable[[Exception], bool],
    retry_after: Callable[[Exception], float | None] | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Call `fn`; on a retryable failure wait and try again, up to `policy.max_attempts` total tries.

    The LAST exception is re-raised unchanged (never wrapped), so callers keep their own error handling.
    Only `Exception` is caught: KeyboardInterrupt / SystemExit always propagate immediately.
    """
    attempt = 1
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classification is delegated to is_retryable
            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise
            delay = policy.delay_for(attempt, rng)
            hint = retry_after(exc) if retry_after else None
            if hint is not None:
                if hint > policy.max_delay:
                    raise  # the server asks for longer than we are willing to wait
                delay = max(delay, hint)
            if on_retry:
                on_retry(attempt, exc, delay)
            sleep(delay)
            attempt += 1


# ----------------------------------------------------------------------
# HTTP helpers for idempotent GETs made with `requests`
# ----------------------------------------------------------------------
def is_transient_http_error(exc: Exception) -> bool:
    """Connection problems, timeouts and 408/425/429/5xx are worth retrying; 4xx client errors are not."""
    if isinstance(exc, requests.HTTPError):
        response = exc.response
        return response is not None and response.status_code in TRANSIENT_HTTP_STATUS
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))


def http_retry_after(exc: Exception) -> float | None:
    """Seconds from a numeric Retry-After header, if the failed response carried one."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        value = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None  # HTTP-date form or garbage: ignore, plain backoff applies
    return value if value >= 0 else None
