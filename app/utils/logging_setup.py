"""
FILE: app/utils/logging_setup.py
MODULE 16 — logging system.

    run_id = configure_logging(level="INFO", log_dir=Path("logs"), secrets=settings.secret_values())

WHAT YOU GET
  * CONSOLE (stderr) + ROTATING FILE  logs/trendwatch.log   (size-capped: default 2 MB x 5 backups, so the
    disk can never fill up). stdout stays clean for main.py's run summary, which matters under cron/systemd.
  * ONE LINE FORMAT, UTC:   2026-09-21T09:30:02Z INFO    [3f9a1c2e] app.agents.workflow: collect: 42 items
    UTC avoids daylight-saving/timezone confusion; the [run id] lets you grep everything one run did.
  * SECRET REDACTION: every line is scrubbed twice: by exact VALUE (the keys Settings holds) and by PATTERN
    (Google keys, OpenWA keys, api-key/authorization headers, ?key= query params). It happens in the
    FORMATTER, so it also covers exception tracebacks and messages logged by third-party libraries.
  * LOG-INJECTION DEFENCE: feed titles are attacker-controlled. Newlines are escaped (a scraped title cannot
    forge a fake "ERROR ..." line), and control characters (ANSI terminal escapes) and bidi overrides are
    neutralised. Tracebacks keep their real line breaks.
  * DELIVERY AUDIT LOG  logs/deliveries.log  (logger "trendwatch.audit"): a write-ahead line BEFORE every
    send ("attempt") and an outcome line after. If a run dies mid-send, the last unmatched "attempt" tells
    you exactly which message to look for in the group. Handlers flush on every record, so the line
    survives even a SIGKILL.
  * NOISE CONTROL: httpx/urllib3/SDK loggers are held at WARNING (one INFO line per HTTP call would drown
    the log) unless you ask for verbose.
  * IDEMPOTENT: calling configure_logging again (tests, repeated in-process runs) replaces its own
    handlers instead of stacking duplicates. Only handlers it created are ever removed.

ARCHITECTURE NOTES
  * `utils/` layer: standard library only. It receives secrets as plain strings; it never imports Settings.
  * Application code just does `logger = logging.getLogger(__name__)`. Where and how it is written is decided
    here, once, by the composition root (main.py).
"""
from __future__ import annotations

import logging
import re
import sys
import time
import uuid
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

AUDIT_LOGGER = "trendwatch.audit"
MAIN_LOG_NAME = "trendwatch.log"
AUDIT_LOG_NAME = "deliveries.log"
REDACTED = "[REDACTED]"
NOISY_LIBRARIES = ("httpx", "httpcore", "urllib3", "google_genai", "langgraph", "langsmith")

_HANDLER_MARK = "_trendwatch_handler"
_MIN_SECRET_LENGTH = 8  # shorter values would blank out ordinary words ("change-me", "true", ...)

# ---- redaction ----
_KEY_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),  # Google API keys
    re.compile(r"owa_k\d+_[0-9A-Za-z_\-]+"),  # OpenWA keys
)
_HEADER_PATTERN = re.compile(r"(?i)\b(x-api-key|x-goog-api-key|authorization)(['\"]?\s*[:=]\s*['\"]?)(?:Bearer\s+)?[^\s,'\"}]+")
_QUERY_PATTERN = re.compile(r"(?i)([?&](?:api[_-]?key|key|token|access_token)=)[^&\s'\"]+")

# ---- log injection ----
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")
_NAMED_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Scrub known secret VALUES first (most reliable), then anything that LOOKS like a credential."""
    for secret in sorted({s for s in secrets if len(s) >= _MIN_SECRET_LENGTH}, key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    for pattern in _KEY_PATTERNS:
        text = pattern.sub(REDACTED, text)
    text = _HEADER_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)
    return _QUERY_PATTERN.sub(lambda m: f"{m.group(1)}{REDACTED}", text)


def neutralize(text: str) -> str:
    """Make one log message ONE line of inert text: escape newlines, control and bidi-override characters."""

    def escape(match: re.Match[str]) -> str:
        char = match.group(0)
        if char in _NAMED_ESCAPES:
            return _NAMED_ESCAPES[char]
        return f"\\x{ord(char):02x}" if ord(char) < 256 else f"\\u{ord(char):04x}"

    return _CONTROL_CHARS.sub(escape, text)


class SafeFormatter(logging.Formatter):
    """UTC timestamps + injection-safe message + redacted final text (message AND traceback)."""

    def __init__(self, fmt: str, secrets: Iterable[str] = ()) -> None:
        super().__init__(fmt, datefmt="%Y-%m-%dT%H:%M:%S")
        self.converter = time.gmtime  # UTC
        self._secrets = tuple(secrets)

    def format(self, record: logging.LogRecord) -> str:
        saved = (record.msg, record.args)
        try:
            # Swap in the neutralised message for THIS formatter only; other handlers see the original.
            record.msg, record.args = neutralize(record.getMessage()), None
            text = super().format(record)
        finally:
            record.msg, record.args = saved
        return redact(text, self._secrets)


class RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
def _level_number(level: int | str) -> int:
    if isinstance(level, int):
        return level
    number = logging.getLevelName(level.strip().upper())
    if not isinstance(number, int):
        raise ValueError(f"unknown log level: {level!r}")
    return number


def _tag(handler: logging.Handler) -> logging.Handler:
    setattr(handler, _HANDLER_MARK, True)
    return handler


def reset_logging() -> None:
    """Remove ONLY the handlers this module installed and restore library levels."""
    for logger in (logging.getLogger(), logging.getLogger(AUDIT_LOGGER)):
        for handler in list(logger.handlers):
            if getattr(handler, _HANDLER_MARK, False):
                logger.removeHandler(handler)
                handler.close()
    logging.getLogger(AUDIT_LOGGER).propagate = True
    logging.getLogger(AUDIT_LOGGER).setLevel(logging.NOTSET)
    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.NOTSET)
    logging.getLogger().setLevel(logging.WARNING)


def configure_logging(
    *,
    level: int | str = "INFO",
    log_dir: Path | None = None,
    secrets: Iterable[str] = (),
    run_id: str | None = None,
    stream: TextIO | None = None,
    third_party_level: int | str = "WARNING",
    max_bytes: int = 2 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
) -> str:
    """Install console/file/audit handlers. Returns the run id stamped on every line."""
    reset_logging()  # idempotent: never stack duplicate handlers
    run_id = run_id or uuid.uuid4().hex[:8]
    number = _level_number(level)
    secrets = tuple(secrets)
    run_filter = RunIdFilter(run_id)
    line = "%(asctime)sZ %(levelname)-7s [%(run_id)s] %(name)s: %(message)s"

    root = logging.getLogger()
    root.setLevel(number)
    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(max(number, _level_number(third_party_level)))

    def attach(logger: logging.Logger, handler: logging.Handler, fmt: str, handler_level: int) -> None:
        handler.setLevel(handler_level)
        handler.setFormatter(SafeFormatter(fmt, secrets))
        handler.addFilter(run_filter)
        logger.addHandler(_tag(handler))

    if console:
        attach(root, logging.StreamHandler(stream or sys.stderr), line, number)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        attach(root, RotatingFileHandler(log_dir / MAIN_LOG_NAME, maxBytes=max_bytes, backupCount=backup_count,
                                         encoding="utf-8"), line, number)
        audit = logging.getLogger(AUDIT_LOGGER)
        audit.setLevel(logging.INFO)
        audit.propagate = False  # audit lines live in their own file, not mixed into the main log
        attach(audit, RotatingFileHandler(log_dir / AUDIT_LOG_NAME, maxBytes=max_bytes, backupCount=backup_count * 2,
                                          encoding="utf-8"), "%(asctime)sZ [%(run_id)s] %(message)s", logging.INFO)
    return run_id
