"""
FILE: app/utils/runlock.py
MODULE 15 — single-instance run lock.

WHY: two overlapping runs (a slow run still going when the scheduler fires again, or you starting
`python main.py` by hand) would duplicate Gemini spend and fight over SQLite. The lock makes the
second run exit immediately instead.

DESIGN
  * ATOMIC CREATE: `os.open(O_CREAT | O_EXCL)` either creates the file or fails: no
    check-then-create race between two processes starting at the same moment.
  * STALE = OLD, not "dead process": probing PIDs is not portable (on Windows `os.kill(pid, 0)`
    actually terminates the process!). A crashed run leaves its lock behind, so a lock older than
    `stale_after` seconds is taken over. A normal run takes about a minute; the default is 30 minutes.
  * The lock file holds {pid, started_at} purely as diagnostics for the error message.
  * LAYERED SAFETY, not a single point of failure: even if two runs did overlap through the (tiny)
    stale-takeover window, delivery stays duplicate-free because every message is claimed in SQLite
    with an atomic UPDATE before it is sent (Module 15, repository.claim_for_delivery).
  * `release()` only deletes the lock if this process still owns it.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path


class RunLockError(RuntimeError):
    """Another run holds a fresh lock."""


class RunLock:
    def __init__(
        self,
        path: Path,
        *,
        stale_after: float = 30 * 60,
        clock: Callable[[], float] = time.time,
        pid: Callable[[], int] = os.getpid,
    ) -> None:
        self._path = Path(path)
        self._stale_after = stale_after
        self._clock, self._pid = clock, pid
        self._held = False

    # ---- helpers ----
    def _age(self) -> float | None:
        """Seconds since the lock file was last written; None if it does not exist."""
        try:
            return self._clock() - self._path.stat().st_mtime
        except FileNotFoundError:
            return None

    def _describe_holder(self) -> str:
        try:
            info = json.loads(self._path.read_text(encoding="utf-8"))
            return f"pid {info.get('pid')}, started {self._age() or 0:.0f}s ago"
        except (OSError, ValueError):
            return "unknown holder"

    def _try_create(self) -> bool:
        try:
            fd = os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": self._pid(), "started_at": self._clock()}, handle)
        return True

    # ---- public API ----
    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._try_create():
            self._held = True
            return
        age = self._age()
        if age is not None and age <= self._stale_after:
            raise RunLockError(
                f"another run is active ({self._describe_holder()}). Lock file: {self._path} "
                f"(it is taken over automatically after {self._stale_after / 60:.0f} min; "
                f"if you are SURE nothing is running, delete that file)"
            )
        # Stale (or vanished between the two calls): take it over.
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass
        if not self._try_create():  # somebody else won the takeover race: they own it now
            raise RunLockError(f"another run just started ({self._describe_holder()})")
        self._held = True

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            owner = json.loads(self._path.read_text(encoding="utf-8")).get("pid")
        except (OSError, ValueError):
            return  # file gone or unreadable: nothing of ours to remove
        if owner == self._pid():
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
