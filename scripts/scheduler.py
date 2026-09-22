#!/usr/bin/env python3
"""
FILE: scripts/scheduler.py
MODULE 17 — periodic scheduling engine.

    python scripts/scheduler.py                # loop forever: run now, then every SCHEDULE_INTERVAL_MINUTES
    python scripts/scheduler.py --once          # a single tick, then exit (for testing, or if you'd rather
                                                 # drive repetition with cron / systemd timers / Task Scheduler)
    python scripts/scheduler.py --interval 5    # override the interval for this process only
    python scripts/scheduler.py --dry-run       # force every tick into rehearsal mode
    python scripts/scheduler.py --skip-first-run   # wait for the first interval instead of running immediately

Stop with Ctrl+C (or a service manager's stop signal, SIGTERM): the loop finishes the CURRENT tick, then exits.

DESIGN NOTES
  * Operations tooling: lives in scripts/, may import `app.*` and `main`; nothing in `app/` imports it.
  * WHY APScheduler over cron / systemd timers / Task Scheduler: this project is zero-budget, local, and
    cross-platform (Module 1), and those three are OS-specific and need separate setup on Linux, macOS and
    Windows. APScheduler already ships as a pinned dependency (Module 1's requirements.txt, put there for
    exactly this module) and runs identically on all three inside one long-lived process — a terminal,
    tmux/screen, or whatever background-process mechanism the OS offers; that choice is orthogonal to this
    script. BlockingScheduler (not Background/Async) fits a plain, single-purpose foreground CLI.
  * TWO LAYERS OF OVERLAP PROTECTION, DELIBERATELY:
        1. APScheduler's own `max_instances=1` — never start tick N+1 while tick N (here, in THIS process)
           is still running. Verified for real: a job held busy past its interval has the next firing(s)
           skipped, not queued.
        2. The RunLock main.main() already takes before touching the database (Module 15) — a cross-PROCESS
           guard, for when someone also runs `python main.py` by hand, or starts a second scheduler.
    Neither alone covers both cases; in ordinary single-scheduler operation, layer 2 should never fire.
  * `coalesce=True` + `misfire_grace_time=60`: if ticks were missed entirely (the machine was asleep), run
    ONE catch-up tick instead of a burst of overdue ones; a tick merely running a little late still fires.
  * A TICK NEVER CRASHES THE LOOP: APScheduler already isolates a scheduled job's own exceptions from its
    main loop (verified for real — a raising job logs and the scheduler keeps ticking). `run_job`'s own
    try/except is what protects the OTHER two call sites that invoke it directly, outside any executor:
    `--once` and the immediate "run now" tick. Only `Exception` is caught, not `BaseException` — same
    reasoning as `app/utils/retry.py`: Ctrl+C must still stop the scheduler, not just abort one tick.
  * `main.main()` re-runs `configure_logging` on every tick (Module 16) — deliberately: each tick gets its
    own run id while writing into the SAME rotating log files, so one scheduler process produces one
    continuous, greppable history with clear boundaries between runs.
  * `apscheduler`'s own logger is left at its normal level (it is NOT added to logging_setup.NOISY_LIBRARIES):
    its messages are rare (one line per scheduled job, plus a warning on a genuine overlap or misfire), not
    per-request chatter like an HTTP client's — exactly the kind of signal an operator running this for
    weeks should see, not noise to hide.
  * Exit codes: whatever `main.main()` returned, for `--once`; EXIT_OK (0) after a clean Ctrl+C/SIGTERM stop
    of the continuous loop; EXIT_CONFIG (1) if settings or `--interval` are invalid before the loop starts.
"""
from __future__ import annotations

import argparse
import signal
import sys
from collections.abc import Callable, Sequence
from datetime import timezone
from itertools import count
from pathlib import Path
from typing import Any

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

# `python scripts/scheduler.py` puts scripts/ (not the project root) on sys.path, so `import main` and
# `import app` would fail. pytest hides this (pytest.ini sets pythonpath); a subprocess test guards against
# it — same bug, same fix, as scripts/delivery_test.py (Module 13).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import main as runner  # noqa: E402
from app.config.settings import ConfigError, Settings, get_settings  # noqa: E402

JOB_ID = "trendwatch-pipeline"

_EXIT_LABELS = {
    runner.EXIT_OK: "ok",
    runner.EXIT_CONFIG: "configuration problem",
    runner.EXIT_FATAL: "fatal condition (see the log)",
    runner.EXIT_CRASH: "crashed unexpectedly (see the log)",
    runner.EXIT_BUSY: "skipped (another run was already active)",
}


def run_job(run_fn: Callable[[], int], *, tick: int) -> int:
    """Run one tick: call `run_fn`, contain any exception, return an exit code. Never raises Exception.

    KeyboardInterrupt/SystemExit are NOT caught: Ctrl+C during a tick must still stop the scheduler.
    """
    print(f"--- tick {tick}: starting pipeline run ---")
    try:
        code = run_fn()
    except Exception as exc:  # noqa: BLE001 - last line of defence; see the module docstring
        print(f"--- tick {tick}: run_fn raised {type(exc).__name__}: {exc} "
              "(this tick is skipped; the scheduler keeps going) ---")
        return runner.EXIT_CRASH
    print(f"--- tick {tick}: finished ({_EXIT_LABELS.get(code, 'unrecognised exit code')}, exit {code}) ---")
    return code


def build_scheduler(
    settings: Settings,
    *,
    interval_minutes: int | None = None,
    force_dry_run: bool = False,
    run_fn: Callable[[], int] | None = None,
    scheduler_factory: Callable[..., Any] = BlockingScheduler,
) -> tuple[Any, Callable[[], int]]:
    """Build a configured (NOT yet started) scheduler, plus the exact callable each tick will invoke.

    `run_fn` lets tests (and the sandbox demo below) replace the real pipeline entirely; the default
    calls `main.main()` exactly as the command line would, so `--dry-run` reuses main.py's own tested path.
    `scheduler_factory` lets tests substitute a fake scheduler whose `.start()` never actually blocks.
    """
    minutes = interval_minutes if interval_minutes is not None else settings.schedule_interval_minutes
    if minutes < 1:
        raise ValueError("interval_minutes must be >= 1")

    argv = ["--dry-run"] if force_dry_run else []
    actual_run_fn = run_fn or (lambda: runner.main(argv))

    scheduler = scheduler_factory(timezone=timezone.utc)
    tick_numbers = count(1)  # 1, 2, 3, ... ; tick 0 is reserved for the immediate "run now" tick, if any

    def job() -> None:
        run_job(actual_run_fn, tick=next(tick_numbers))

    scheduler.add_job(
        job,
        trigger=IntervalTrigger(minutes=minutes, timezone=timezone.utc),
        id=JOB_ID,
        max_instances=1,  # layer 1 of overlap protection (see module docstring)
        coalesce=True,  # missed ticks (machine asleep) -> one catch-up tick, not a burst
        misfire_grace_time=60,  # a tick running <60s late still fires; later than that, it's skipped
    )
    return scheduler, actual_run_fn


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    run_fn: Callable[[], int] | None = None,
    scheduler_factory: Callable[..., Any] = BlockingScheduler,
) -> int:
    parser = argparse.ArgumentParser(description="Run the AI TrendWatch pipeline on a repeating schedule.")
    parser.add_argument("--interval", type=int, metavar="MINUTES", help="override SCHEDULE_INTERVAL_MINUTES for this process")
    parser.add_argument("--dry-run", action="store_true", help="force every tick into rehearsal mode")
    parser.add_argument("--once", action="store_true", help="run a single tick now, then exit (no loop)")
    parser.add_argument("--skip-first-run", action="store_true", help="wait for the first interval instead of running immediately")
    args = parser.parse_args(argv)

    try:
        settings = settings or get_settings()
    except ConfigError as exc:
        print(f"Configuration problem: {exc}")
        return runner.EXIT_CONFIG
    if args.interval is not None and args.interval < 1:
        print("Error: --interval must be >= 1")
        return runner.EXIT_CONFIG

    scheduler, job_run_fn = build_scheduler(
        settings, interval_minutes=args.interval, force_dry_run=args.dry_run,
        run_fn=run_fn, scheduler_factory=scheduler_factory,
    )

    if args.once:
        return run_job(job_run_fn, tick=1)

    interval = args.interval if args.interval is not None else settings.schedule_interval_minutes
    run_at_start = not args.skip_first_run and settings.schedule_run_immediately
    print(
        f"Scheduler starting: every {interval} minute(s)" + (" (dry run forced)" if args.dry_run else "") + ". "
        + ("Running the first pipeline pass now." if run_at_start else f"First run in {interval} minute(s).")
        + " Press Ctrl+C to stop."
    )
    if run_at_start:
        run_job(job_run_fn, tick=0)

    # SIGTERM (how service managers ask a process to stop) is made to behave exactly like Ctrl+C (SIGINT),
    # so both paths share one shutdown message and one graceful-stop code path below.
    previous_handler = None
    if hasattr(signal, "SIGTERM"):
        previous_handler = signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        scheduler.start()  # blocks; fires `job` on the interval until interrupted
    except KeyboardInterrupt:
        print("\nStopping: letting the current tick (if any) finish...")
        scheduler.shutdown(wait=True)
        print("Scheduler stopped.")
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
    return runner.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
