#!/usr/bin/env python3
"""
FILE: scripts/setup_openwa.py
MODULE 11 — install, start and inspect the local OpenWA WhatsApp gateway.

    python scripts/setup_openwa.py up        # start the gateway
    python scripts/setup_openwa.py status    # gateway health check
    python scripts/setup_openwa.py key       # print the admin key
    python scripts/setup_openwa.py sessions  # list session UUIDs
    python scripts/setup_openwa.py groups <SESSION_ID>  # list groups
    python scripts/setup_openwa.py down      # stop the container

ARCHITECTURE NOTES
  * This is OPERATIONS tooling, not application code, so it lives in scripts/ (outside `app/`)
    and never touches the pipeline. The pipeline talks to the gateway ONLY through the HTTP
    adapter built in Module 12; nothing here is imported by `app/`.
  * OpenWA is a SEPARATE program (NestJS + Chromium). We do not vendor or modify it: it is cloned
    into ./openwa (git-ignored: that folder will hold your WhatsApp login credentials) and run
    with its own docker-compose.dev.yml, which builds the image locally on first start.
  * IDEMPOTENT and NON-DESTRUCTIVE: re-running `up` never re-clones, never overwrites your
    openwa/.env, and `down` keeps ./openwa/data (paired session + admin key).
  * Every side effect (subprocess, HTTP, clock) is an injectable parameter, so the logic is
    unit-tested without Docker or a network.
  * Facts verified against the OpenWA docs: repo rmyndharis/OpenWA, compose file
    docker-compose.dev.yml, port 2785 bound to 127.0.0.1, container name `openwa-api`, public
    health route /api/health/ready, admin key written to data/.api-key on first start.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
import uuid

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATEWAY_DIR = PROJECT_ROOT / "openwa"  # git-ignored
ENV_TEMPLATE = PROJECT_ROOT / "openwa.env.example"

REPO_URL = "https://github.com/rmyndharis/OpenWA.git"
COMPOSE_FILE = "docker-compose.dev.yml"
CONTAINER_NAME = "openwa-api"
PORT = 2785
BASE_URL = f"http://127.0.0.1:{PORT}"  # 127.0.0.1, not "localhost": avoids Windows IPv6 (::1) surprises
HEALTH_URL = f"{BASE_URL}/api/health/ready"

Run = Callable[..., "subprocess.CompletedProcess[str]"]


class SetupError(RuntimeError):
    """A problem the user can fix; printed as a message, not a traceback."""


# ----------------------------------------------------------------------
# Building blocks (each injectable / individually testable)
# ----------------------------------------------------------------------
def check_prerequisites(
    which: Callable[[str], str | None] = shutil.which, run: Run = subprocess.run
) -> list[str]:
    """Return human-readable problems (empty list = good to go)."""
    problems: list[str] = []
    if not which("git"):
        problems.append("git is not installed -> https://git-scm.com/downloads")
    if not which("docker"):
        problems.append(
            "Docker is not installed -> https://docs.docker.com/get-docker/"
        )
        return problems

    def ok(cmd: list[str]) -> bool:
        try:
            return run(cmd, capture_output=True, text=True, timeout=60).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    if not ok(["docker", "compose", "version"]):
        problems.append(
            "`docker compose` (v2) is not available -> update Docker Desktop / install the compose plugin"
        )
    elif not ok(["docker", "info"]):
        problems.append(
            "Docker is installed but its daemon is not running -> start Docker Desktop, then retry"
        )
    return problems


def clone_gateway(gateway_dir: Path, run: Run = subprocess.run) -> bool:
    """Shallow-clone OpenWA into `gateway_dir`. Returns True if cloned now, False if already there."""
    if (gateway_dir / COMPOSE_FILE).exists():
        return False
    if gateway_dir.exists() and any(gateway_dir.iterdir()):
        raise SetupError(
            f"{gateway_dir} exists but is not an OpenWA checkout; move or delete it and retry"
        )
    run(["git", "clone", "--depth", "1", REPO_URL, str(gateway_dir)], check=True)
    return True


def gateway_revision(gateway_dir: Path, run: Run = subprocess.run) -> str:
    """Short git commit of the checkout: record it so a working setup is reproducible."""
    try:
        result = run(
            ["git", "-C", str(gateway_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def ensure_gateway_env(gateway_dir: Path, template: Path = ENV_TEMPLATE) -> bool:
    """Create openwa/.env from our template if absent. NEVER overwrites. True if created."""
    target = gateway_dir / ".env"
    if target.exists():
        return False
    shutil.copyfile(template, target)
    return True


def compose(gateway_dir: Path, *args: str, run: Run = subprocess.run) -> None:
    run(["docker", "compose", "-f", COMPOSE_FILE, *args], cwd=gateway_dir, check=True)


def wait_until_ready(
    url: str = HEALTH_URL,
    *,
    timeout: float = 300.0,
    interval: float = 3.0,
    get: Callable[..., requests.Response] = requests.get,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Poll the (public) health route until HTTP 200 or `timeout`. Connection errors just mean 'not yet'."""
    deadline = clock() + timeout
    while True:
        try:
            if get(url, timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        if clock() >= deadline:
            return False
        sleep(interval)


def read_admin_key(gateway_dir: Path, run: Run = subprocess.run) -> str | None:
    """Admin key OpenWA generated on first start. Host file first, then `docker exec` (file may be root-owned)."""
    try:
        text = (gateway_dir / "data" / ".api-key").read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        pass
    try:
        result = run(
            ["docker", "exec", CONTAINER_NAME, "cat", "/app/data/.api-key"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout.strip() or None) if result.returncode == 0 else None


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------
NEXT_STEPS = f"""
OpenWA is running.  Dashboard: {BASE_URL}

NEXT (one time):
  1. Get the admin key:     python scripts/setup_openwa.py key
      (the dashboard asks for it)
  2. Dashboard -> Sessions -> create a session (e.g. "trendwatch") -> Start
  3. On the DEDICATED TEST PHONE: WhatsApp -> Settings -> Linked devices
      -> Link a device, then scan the QR code shown in the dashboard.
  4. Run `python scripts/setup_openwa.py sessions` to list session UUIDs,
      then `python scripts/setup_openwa.py groups <SESSION_ID>` to list
      groups for one session.
  5. Copy the session id, group id and admin key into .env
      (see the Module 11 notes).
"""


def _request_json(
    path: str,
    api_key: str,
    *,
    params: dict[str, object] | None = None,
    get: Callable[..., requests.Response] = requests.get,
) -> object:
    try:
        url = f"{BASE_URL}{path}"
        response = get(
            url,
            headers={"X-API-Key": api_key},
            params=params,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise SetupError(f"OpenWA request failed: {exc}") from exc
    if response.status_code != 200:
        detail = response.text.strip().replace("\n", " ")
        message = (
            f"OpenWA returned HTTP {response.status_code}: "
            f"{detail[:200] or '<empty>'}"
        )
        raise SetupError(message)
    try:
        return response.json()
    except ValueError as exc:
        raise SetupError("OpenWA returned non-JSON data") from exc


def _format_session_row(item: object) -> str:
    if not isinstance(item, dict):
        return f"<unexpected session payload: {item!r}>"
    session_id = str(item.get("id", "<missing>"))
    name = str(item.get("name") or "<unnamed>")
    status = str(item.get("status") or "<unknown>")
    phone = str(item.get("phone") or "")
    return f"{session_id:<36}  {status:<12}  {name:<20}  {phone}"


def _format_group_row(item: object) -> str:
    if not isinstance(item, dict):
        return f"<unexpected group payload: {item!r}>"
    group_id = str(item.get("id", "<missing>"))
    name = str(item.get("name") or "<unnamed>")
    linked_parent = str(item.get("linkedParentJID") or "")
    return f"{group_id:<36}  {name:<28}  {linked_parent}"


def cmd_up(gateway_dir: Path) -> int:
    problems = check_prerequisites()
    if problems:
        print("Cannot continue:\n  - " + "\n  - ".join(problems))
        return 1
    if clone_gateway(gateway_dir):
        print(f"Cloned OpenWA into {gateway_dir}")
    print(
        f"OpenWA revision: {gateway_revision(gateway_dir)}  (note it down; upgrades can change the API)"
    )
    if ensure_gateway_env(gateway_dir):
        print(f"Wrote {gateway_dir / '.env'} from {ENV_TEMPLATE.name}")
    print(
        "Starting the gateway. The FIRST start builds the image (Chromium included): allow several minutes ..."
    )
    compose(gateway_dir, "up", "-d", "--build")
    if not wait_until_ready():
        print(
            f"The gateway did not become healthy in time. Inspect:  docker compose -f {COMPOSE_FILE} logs openwa "
            f"(run inside {gateway_dir})"
        )
        return 1
    print(NEXT_STEPS)
    return 0


def cmd_down(gateway_dir: Path) -> int:
    if not (gateway_dir / COMPOSE_FILE).exists():
        print("Nothing to stop: OpenWA has not been set up here.")
        return 1
    compose(gateway_dir, "down")
    print(
        "Gateway stopped. Your paired session and API key are kept in",
        gateway_dir / "data",
    )
    return 0


def cmd_status() -> int:
    ready = wait_until_ready(timeout=0)
    print(f"OpenWA at {BASE_URL}: {'READY' if ready else 'NOT reachable / not ready'}")
    return 0 if ready else 1


def cmd_key(gateway_dir: Path) -> int:
    key = read_admin_key(gateway_dir)
    if key is None:
        print(
            "No admin key found. Has the gateway been started? (python scripts/setup_openwa.py up)\n"
            f"Fallback:  docker compose -f {COMPOSE_FILE} logs openwa   (the key starts with owa_k1_)"
        )
        return 1
    print(key)
    return 0


def cmd_sessions(
    gateway_dir: Path,
    *,
    get: Callable[..., requests.Response] = requests.get,
) -> int:
    key = read_admin_key(gateway_dir)
    if key is None:
        print(
            "No admin key found. Has the gateway been started? (python scripts/setup_openwa.py up)\n"
            f"Fallback:  docker compose -f {COMPOSE_FILE} logs openwa   (the key starts with owa_k1_)"
        )
        return 1
    sessions = _request_json("/api/sessions", key, get=get)
    if not isinstance(sessions, list):
        raise SetupError("unexpected sessions payload: expected a JSON list")
    print("SESSION ID                         STATUS  NAME          PHONE")
    if not sessions:
        print("<no sessions found>")
        return 0
    for item in sessions:
        print(_format_session_row(item))
    return 0


def cmd_groups(
    gateway_dir: Path,
    session_id: str,
    *,
    get: Callable[..., requests.Response] = requests.get,
) -> int:
    session_id = session_id.strip()
    try:
        uuid.UUID(session_id)
    except ValueError:
        print(
            "Error: session_id must be the UUID from "
            "python scripts/setup_openwa.py sessions, not the dashboard name "
            "or <SESSION_ID>"
        )
        return 1

    key = read_admin_key(gateway_dir)
    if key is None:
        print(
            "No admin key found. Has the gateway been started? "
            "(python scripts/setup_openwa.py up)\n"
            f"Fallback:  docker compose -f {COMPOSE_FILE} logs openwa   "
            "(the key starts with owa_k1_)"
        )
        return 1
    groups = _request_json(
        f"/api/sessions/{session_id}/groups",
        key,
        params={"limit": 100, "offset": 0},
        get=get,
    )
    if not isinstance(groups, list):
        raise SetupError("unexpected groups payload: expected a JSON list")
    print("GROUP ID                          NAME        LINKED PARENT")
    if not groups:
        print("<no groups found>")
        return 0
    for item in groups:
        print(_format_group_row(item))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manage the local OpenWA WhatsApp gateway.",
    )
    parser.add_argument(
        "command",
        choices=["up", "down", "status", "key", "sessions", "groups"],
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_GATEWAY_DIR,
        help="OpenWA checkout (default: ./openwa)",
    )
    parser.add_argument(
        "session_id",
        nargs="?",
        help="OpenWA session UUID (required for the groups command)",
    )
    args = parser.parse_args(argv)
    try:
        return {
            "up": lambda: cmd_up(args.dir),
            "down": lambda: cmd_down(args.dir),
            "status": cmd_status,
            "key": lambda: cmd_key(args.dir),
            "sessions": lambda: cmd_sessions(args.dir),
            "groups": lambda: cmd_groups(args.dir, args.session_id or ""),
        }[args.command]()
    except SetupError as exc:
        print(f"Error: {exc}")
        return 1
    except subprocess.CalledProcessError as exc:
        command_text = " ".join(map(str, exc.cmd))
        print(f"Error: command failed ({exc.returncode}): {command_text}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
