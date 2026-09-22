#!/usr/bin/env python3
"""
FILE: scripts/delivery_test.py
MODULE 13 — controlled WhatsApp delivery test.

    python scripts/delivery_test.py           # REHEARSAL (default): sends NOTHING
    python scripts/delivery_test.py --send    # ONE real test message, after you type SEND

REHEARSAL checks your configuration, shows the exact message, and (read-only) confirms the
gateway is healthy and the session is linked. It is safe to run any time.

--send additionally requires ALL of:
  * an interactive terminal (no piping, no cron, no CI): a human must be present;
  * every rehearsal check passing;
  * the exact word SEND typed at the prompt;
  * no other test message in the last 30 minutes (a cool-down, so a retry loop or an impatient
    re-run cannot spam the class group).
It sends EXACTLY ONE message (client cap max_sends=1) and never retries.

DESIGN NOTES
  * Operations tooling: lives in scripts/, may import `app.*`, is imported by nothing in `app/`.
  * The pipeline's own safety switch (WHATSAPP_DRY_RUN) is deliberately NOT touched: it stays
    `true` in .env until YOU flip it after this test succeeds. This script is an explicit,
    human-confirmed action, independent of that switch, but it applies the same live-config
    rules (group id, key and session id all required).
  * The message goes through the REAL formatter, so what you see in the group is what the
    pipeline will produce: emoji, bold title, link line. It is clearly labelled [TEST].
  * Outcome-specific guidance. The dangerous case is DELIVERY UNKNOWN (request sent, no reply):
    the message may be in the group, so the cool-down is recorded and you are told to CHECK the
    group before doing anything else.
  * Exit codes: 0 ok | 1 configuration/preflight problem | 2 send failed | 3 aborted (no send).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# `python scripts/delivery_test.py` puts scripts/ (not the project root) on sys.path, so `import app`
# would fail. pytest hides this (pytest.ini sets pythonpath), a subprocess test guards against it.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.config.settings import PROJECT_ROOT, ConfigError, Settings, get_settings  # noqa: E402
from app.messaging.formatter import FormattedMessage, format_item
from app.messaging.openwa_client import (
    OpenWAAuthError,
    OpenWAClient,
    OpenWADeliveryUnknownError,
    OpenWAError,
    OpenWAFatalError,
    OpenWARateLimitedError,
    OpenWASessionError,
    OpenWAUnavailableError,
    SendResult,
)
from app.models import SourceType, TrendItem

EXIT_OK, EXIT_PROBLEM, EXIT_SEND_FAILED, EXIT_ABORTED = 0, 1, 2, 3
COOLDOWN_SECONDS = 30 * 60
CONFIRM_WORD = "SEND"
STATE_FILE = PROJECT_ROOT / "data" / "delivery_test.json"  # data/ is git-ignored


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------
def build_test_message(now: datetime | None = None) -> FormattedMessage:
    """A clearly labelled test, produced by the production formatter."""
    item = TrendItem(
        title="[TEST] AI TrendWatch delivery check",
        url="https://example.com/ai-trendwatch-delivery-test",
        source_name="AI TrendWatch",
        source_type=SourceType.RSS,
        published_at=now or datetime.now(timezone.utc),
    ).with_analysis(
        10,
        "One-off test message from the AI TrendWatch class project to confirm delivery works. "
        "No action needed; regular updates start only after this test.",
    )
    return format_item(item)


def api_key_is_set(settings: Settings) -> bool:
    """True if OPENWA_API_KEY holds a real value (not empty, not the template placeholder)."""
    return settings.openwa_api_key.get_secret_value().strip().lower() not in ("", "change-me")


def missing_live_config(settings: Settings) -> list[str]:
    """What must be filled in .env before ANY real message can be sent (same rules as Settings)."""
    problems = []
    if not settings.whatsapp_group_id:
        problems.append("WHATSAPP_GROUP_ID is empty (Module 11: list groups, copy the id ending in @g.us)")
    if not api_key_is_set(settings):
        problems.append("OPENWA_API_KEY is not set (python scripts/setup_openwa.py key)")
    if not settings.openwa_session_id.strip():
        problems.append("OPENWA_SESSION_ID is empty (the session UUID from the OpenWA dashboard)")
    return problems


def cooldown_remaining(state_path: Path, now: datetime, cooldown_seconds: float = COOLDOWN_SECONDS) -> float:
    """Seconds until another test message is allowed (0 = allowed). A missing/corrupt file means allowed."""
    try:
        sent_at = datetime.fromisoformat(json.loads(state_path.read_text(encoding="utf-8"))["sent_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0
    return max(0.0, cooldown_seconds - (now - sent_at).total_seconds())


def record_test_send(state_path: Path, now: datetime) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"sent_at": now.isoformat()}), encoding="utf-8")


def preflight(client) -> list[Check]:
    """Read-only checks against the gateway. Sends nothing."""
    checks: list[Check] = []
    healthy = client.is_healthy()
    checks.append(Check("gateway reachable", healthy, "" if healthy else "start it: python scripts/setup_openwa.py up"))
    if not healthy:
        return checks
    try:
        client.ensure_ready()
        checks.append(Check("WhatsApp session linked and ready", True))
    except OpenWAError as exc:
        checks.append(Check("WhatsApp session linked and ready", False, str(exc)))
    return checks


def guidance_for(exc: OpenWAError) -> str:
    """Turn an adapter error into the specific next step."""
    if isinstance(exc, OpenWADeliveryUnknownError):
        return ("The gateway did not reply, so the message MAY have been posted. Open the group on the test phone "
                "and LOOK before doing anything else. Do not re-run blindly (that is how duplicates happen).")
    if isinstance(exc, OpenWAAuthError):
        return "The API key was rejected: re-read it with `python scripts/setup_openwa.py key` and update OPENWA_API_KEY."
    if isinstance(exc, OpenWASessionError):
        return "The session is not usable: open the dashboard, Start it and/or re-link the phone; check OPENWA_SESSION_ID."
    if isinstance(exc, OpenWARateLimitedError):
        wait = f" Retry after about {exc.retry_after:.0f}s." if exc.retry_after else ""
        return f"The gateway is rate-limiting requests.{wait}"
    if isinstance(exc, OpenWAUnavailableError):
        return "The request did not go through (gateway unreachable or erroring). Nothing was sent; safe to try again later."
    if isinstance(exc, OpenWAFatalError):
        return "A fatal gateway condition stopped the send."
    return "The gateway rejected this message. Nothing was posted."


def _default_client(settings: Settings) -> OpenWAClient:
    return OpenWAClient(
        settings.openwa_base_url,
        settings.openwa_api_key.get_secret_value(),
        settings.openwa_session_id,
        max_sends=1,  # hard cap: this tool can never send more than one message
    )


# ----------------------------------------------------------------------
# Command
# ----------------------------------------------------------------------
def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    client_factory: Callable[[Settings], object] = _default_client,
    input_fn: Callable[[str], str] = input,
    isatty: Callable[[], bool] = sys.stdin.isatty,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    state_path: Path = STATE_FILE,
) -> int:
    parser = argparse.ArgumentParser(description="Rehearse (default) or perform ONE controlled WhatsApp test send.")
    parser.add_argument("--send", action="store_true", help="really send one test message (asks for confirmation)")
    args = parser.parse_args(argv)

    try:
        settings = settings or get_settings()
    except ConfigError as exc:
        print(exc)
        return EXIT_PROBLEM

    mode = "SEND (one real message)" if args.send else "REHEARSAL (nothing will be sent)"
    print(f"Mode: {mode}")
    print(f"  gateway : {settings.openwa_base_url}")
    print(f"  session : {settings.openwa_session_id or '<missing>'}")
    print(f"  group   : {settings.whatsapp_group_id or '<missing>'}")
    print(f"  api key : {'<set>' if api_key_is_set(settings) else '<missing>'}")

    problems = missing_live_config(settings)
    if problems:
        print("\nCannot continue. Fill these in .env first:\n  - " + "\n  - ".join(problems))
        return EXIT_PROBLEM

    message = build_test_message(now_fn())
    print("\n--- message that would be posted ---\n" + message.text + "\n------------------------------------")

    client = client_factory(settings)
    checks = preflight(client)
    for check in checks:
        print(f"  [{'OK' if check.ok else 'FAIL'}] {check.name}" + (f": {check.detail}" if check.detail else ""))
    if not all(c.ok for c in checks):
        print("\nPreflight failed. Nothing was sent.")
        return EXIT_PROBLEM

    if not args.send:
        print("\nRehearsal passed. To post ONE real test message:  python scripts/delivery_test.py --send")
        return EXIT_OK

    # ---- real send: every gate must pass ----
    if not isatty():
        print("\nRefusing to send: this needs an interactive terminal (a human must confirm).")
        return EXIT_ABORTED
    wait = cooldown_remaining(state_path, now_fn())
    if wait > 0:
        print(f"\nRefusing to send: a test message was sent recently. Wait {wait / 60:.0f} more minute(s) "
              f"(or delete {state_path} if you are certain nothing was sent).")
        return EXIT_ABORTED
    print(f"\nThis will post the message above to group {settings.whatsapp_group_id}.")
    print("Have you told the group (or its admin) that a test message is coming?")
    try:
        answer = input_fn(f"Type {CONFIRM_WORD} to post it, anything else to abort: ").strip()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer != CONFIRM_WORD:
        print("Aborted. Nothing was sent.")
        return EXIT_ABORTED

    try:
        result: SendResult = client.send_text(settings.whatsapp_group_id, message.text)
    except OpenWAError as exc:
        print(f"\nSEND FAILED ({type(exc).__name__}): {exc}\n{guidance_for(exc)}")
        if isinstance(exc, OpenWADeliveryUnknownError):
            record_test_send(state_path, now_fn())  # outcome unknown -> block an immediate blind re-send
        return EXIT_SEND_FAILED

    record_test_send(state_path, now_fn())
    print(f"\nSent. Gateway message id: {result.message_id or 'n/a'}")
    print("NOW CHECK THE GROUP on the test phone: the message should appear once, with a bold title,\n"
          "emoji and a link line. Keep WHATSAPP_DRY_RUN=true until you are happy with that.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
