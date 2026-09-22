#!/usr/bin/env python3
"""
FILE: main.py
MODULE 14 — end-to-end runner for AI TrendWatch.

    python main.py             # one full pass; sends for real ONLY if WHATSAPP_DRY_RUN=false in .env
    python main.py --dry-run   # force a rehearsal: the WhatsApp gateway is never even constructed
    python main.py --verbose   # DEBUG lines from our code + INFO from HTTP libraries (for troubleshooting)

One pass = collect -> validate -> deduplicate -> analyse (Gemini) -> triage -> select -> deliver.
Scheduling repeated passes comes in Module 17; this file stays a single, explicit run.

DESIGN NOTES
  * COMPOSITION ROOT: the ONLY place that knows every concrete class and reads Settings. Everything
    else receives its dependencies as arguments, which is why the whole system is testable with fakes.
  * SAFE BY CONSTRUCTION: dry-run is the default (Settings) and `--dry-run` can only make a run SAFER
    (there is deliberately no flag that forces live sending; that decision lives in .env).
  * A DRY RUN LEAVES THE QUEUE UNTOUCHED: previewed items are not marked sent, so the first live run
    delivers what you rehearsed. Dry-run still spends Gemini quota (analysis is real) and fills SQLite.
  * ORDER OF CONSTRUCTION: anything that can fail on configuration (Gemini key, gateway config) is built
    BEFORE the database is opened, so a config mistake never leaves a half-initialised data/ folder.
  * Exit codes: 0 ok (individual dead sources are normal and only reported) | 1 configuration problem |
    2 a fatal condition stopped analysis or delivery (quota, auth, session down, unknown delivery) |
    3 the run crashed unexpectedly (traceback logged; any half-sent item is recovered next run, never re-sent) |
    4 skipped: another run holds the run lock (normal when a scheduled run overlaps a slow one).
  * SINGLE INSTANCE: a lock file next to the database (data/.run.lock) is taken BEFORE the database is opened.
    It is always released, even on a crash or Ctrl+C. See app/utils/runlock.py.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from app.agents.workflow import Analyzer, Collector, TrendState, WorkflowConfig, WorkflowDeps, run_once
from app.ai.gemini_service import GeminiService
from app.collectors.arxiv import ArxivCollector
from app.collectors.rss import RSSCollector
from app.config.settings import ConfigError, Settings, get_settings
from app.database.repository import TrendRepository
from app.messaging.openwa_client import DryRunClient, Gateway, build_gateway
from app.utils.logging_setup import configure_logging, redact
from app.utils.retry import RetryPolicy
from app.utils.runlock import RunLock, RunLockError

logger = logging.getLogger(__name__)

EXIT_OK, EXIT_CONFIG, EXIT_FATAL, EXIT_CRASH, EXIT_BUSY = 0, 1, 2, 3, 4
DRY_RUN_CHAT_ID = "dry-run-placeholder@g.us"  # only ever used by DryRunClient, which sends nothing


def build_deps(
    settings: Settings,
    *,
    force_dry_run: bool = False,
    repository: TrendRepository | None = None,
    collectors: Sequence[Collector] | None = None,
    analyzer: Analyzer | None = None,
    gateway: Gateway | None = None,
) -> WorkflowDeps:
    """Assemble the real object graph. Every argument can be overridden (that is how tests inject fakes)."""
    retry = RetryPolicy(max_attempts=settings.retry_max_attempts, base_delay=settings.retry_base_delay_seconds)

    # 1) things that can fail on CONFIGURATION come first (nothing is created on disk yet)
    if analyzer is None:
        analyzer = GeminiService(
            settings.require_gemini_key(),  # ConfigError with a clear message if missing
            settings.gemini_model,
            max_requests_per_minute=settings.gemini_max_requests_per_minute,
            retry=retry,
        )
    if gateway is None:
        gateway = DryRunClient(max_sends=settings.max_messages_per_run) if (force_dry_run or settings.whatsapp_dry_run) \
            else build_gateway(settings)
    chat_id = settings.whatsapp_group_id or (DRY_RUN_CHAT_ID if isinstance(gateway, DryRunClient) else "")

    # 2) sources and storage
    if collectors is None:
        collectors = [
            RSSCollector(max_items_per_source=settings.max_items_per_source,
                         timeout_seconds=settings.request_timeout_seconds, retry=retry),
            ArxivCollector(max_items=settings.max_items_per_source,
                           timeout_seconds=settings.request_timeout_seconds, retry=retry),
        ]
    if repository is None:
        repository = TrendRepository(settings.database_path)

    return WorkflowDeps(
        collectors=collectors,
        repository=repository,
        analyzer=analyzer,
        config=WorkflowConfig.from_settings(settings),
        gateway=gateway,
        chat_id=chat_id,
    )


def format_summary(state: TrendState) -> str:
    m = state.get("metrics", {})
    g = m.get
    lines = [
        "Run summary",
        f"  collected {g('collected', 0)} | valid {g('valid', 0)} | new {g('new_stored', 0)} | duplicates {g('duplicates', 0)}"
        f" | rejected by validation {g('rejected_validation', 0)}",
        f"  analysed {g('analyzed', 0)} (left for the next run: {g('left_unanalyzed', 0)})"
        f" | rejected for low score {g('rejected_low_score', 0)}",
        f"  selected for delivery {g('to_deliver', 0)} | delivered {g('delivered', 0)} | previewed (dry run) {g('previewed', 0)}"
        f" | failed {g('delivery_failed', 0)} | rejected {g('delivery_rejected', 0)} | unknown {g('delivery_unknown', 0)}",
    ]
    if g("interrupted_recovered"):
        lines.append(f"  interrupted earlier deliveries resolved as 'unknown' (not re-sent): {g('interrupted_recovered')}")
    errors = state.get("errors", [])
    if errors:
        lines.append(f"Problems ({len(errors)}):")
        lines.extend(f"  - {e}" for e in errors)
    return "\n".join(lines)


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    repository: TrendRepository | None = None,
    collectors: Sequence[Collector] | None = None,
    analyzer: Analyzer | None = None,
    gateway: Gateway | None = None,
    lock_path: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run one AI TrendWatch pass.")
    parser.add_argument("--dry-run", action="store_true", help="force a rehearsal (never contacts WhatsApp)")
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging (troubleshooting)")
    args = parser.parse_args(argv)

    try:
        settings = settings or get_settings()
    except ConfigError as exc:
        print(f"Configuration problem: {exc}")
        return EXIT_CONFIG

    run_id = configure_logging(
        level="DEBUG" if args.verbose else settings.log_level,
        log_dir=settings.log_dir if settings.log_to_file else None,
        secrets=settings.secret_values(),  # exact-value redaction on top of the pattern-based one
        max_bytes=settings.log_max_mb * 1024 * 1024,
        backup_count=settings.log_backup_count,
        third_party_level="INFO" if args.verbose else "WARNING",
    )
    logger.info("run started (run id %s, pid %d, forced dry-run: %s)", run_id, os.getpid(), args.dry_run)

    lock = RunLock(lock_path or settings.database_path.parent / ".run.lock", stale_after=settings.run_lock_stale_minutes * 60)
    try:
        lock.acquire()  # BEFORE the database is opened: two runs must never even touch it together
    except RunLockError as exc:
        logger.warning("run skipped: %s", exc)
        print(f"Skipping this run: {exc}")
        return EXIT_BUSY
    try:
        return _run(args, settings, repository, collectors, analyzer, gateway)
    finally:
        lock.release()  # also on crash / Ctrl+C


def _run(args, settings, repository, collectors, analyzer, gateway) -> int:
    try:
        deps = build_deps(
            settings, force_dry_run=args.dry_run, repository=repository,
            collectors=collectors, analyzer=analyzer, gateway=gateway,
        )
    except ConfigError as exc:
        logger.error("configuration problem: %s", exc)
        print(f"Configuration problem: {exc}")
        return EXIT_CONFIG

    logger.debug("configuration: %s", settings.safe_dump())  # secrets already masked by safe_dump()
    owns_repository = repository is None
    try:
        state = run_once(deps)
    except Exception as exc:  # noqa: BLE001 - last line of defence: report, log the traceback, exit cleanly
        logger.exception("run crashed")
        # print() bypasses the logging formatter, so scrub here too: exception text can echo a credential.
        print(redact(f"The run crashed unexpectedly: {type(exc).__name__}: {exc}", settings.secret_values()) + "\n"
              "Details are in the log. An item that was mid-delivery is recovered on the next run "
              "and is never re-sent (see docs in app/agents/workflow.py).")
        return EXIT_CRASH
    finally:
        if owns_repository:
            deps.repository.close()

    print(redact(format_summary(state), settings.secret_values()))  # error strings can quote upstream responses
    if isinstance(deps.gateway, DryRunClient):
        if deps.gateway.sent:
            print(f"\nDRY RUN: {len(deps.gateway.sent)} message(s) WOULD be posted:")
            for _, text in deps.gateway.sent:
                print("\n" + text)
        print("\nDry run: nothing was sent. Set WHATSAPP_DRY_RUN=false in .env to go live.")

    metrics = state.get("metrics", {})
    code = EXIT_FATAL if metrics.get("analysis_aborted") or metrics.get("delivery_aborted") else EXIT_OK
    # ONE line that summarises the run in the log file: the first thing to read after a scheduled run.
    logger.info("run finished: exit=%d metrics=%s problems=%d", code, metrics, len(state.get("errors", [])))
    for problem in state.get("errors", []):
        logger.warning("problem: %s", problem)
    return code


if __name__ == "__main__":
    sys.exit(main())
