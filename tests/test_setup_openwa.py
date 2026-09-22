"""
FILE: tests/test_setup_openwa.py
MODULE 11 tests — run with:  pytest tests/test_setup_openwa.py
Docker, git and the network are all replaced by fakes: these tests check OUR logic
(ordering, idempotency, safety), not OpenWA itself.
"""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from scripts import setup_openwa as s


def done(returncode=0, stdout=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout)


class FakeRun:
    """Records commands; `results` maps a command prefix -> completed-process stand-in."""

    def __init__(self, results=None, default=None):
        self.calls, self.results, self.default = [], results or {}, default or done()

    def __call__(self, cmd, **kwargs):
        self.calls.append((list(cmd), kwargs))
        for prefix, result in self.results.items():
            if list(cmd[: len(prefix)]) == list(prefix):
                if isinstance(result, Exception):
                    raise result
                return result
        return self.default


# =====================================================================
# Prerequisites
# =====================================================================
def which_only(*present):
    return lambda name: f"/usr/bin/{name}" if name in present else None


def test_all_prerequisites_present():
    assert s.check_prerequisites(which_only("git", "docker"), FakeRun()) == []


def test_missing_git_and_docker_are_both_reported():
    problems = s.check_prerequisites(which_only(), FakeRun())
    assert len(problems) == 2 and any("git" in p for p in problems) and any("Docker is not installed" in p for p in problems)


def test_docker_compose_v2_missing():
    run = FakeRun({("docker", "compose", "version"): done(1)})
    assert "docker compose" in s.check_prerequisites(which_only("git", "docker"), run)[0]


def test_docker_daemon_not_running():
    run = FakeRun({("docker", "info"): done(1)})
    (problem,) = s.check_prerequisites(which_only("git", "docker"), run)
    assert "daemon is not running" in problem


def test_prerequisite_probe_that_crashes_counts_as_failure():
    run = FakeRun({("docker", "compose", "version"): FileNotFoundError()})
    assert s.check_prerequisites(which_only("git", "docker"), run)


# =====================================================================
# Clone / config (idempotency and non-destruction)
# =====================================================================
def test_clone_runs_a_shallow_clone_when_missing(tmp_path):
    run = FakeRun()
    target = tmp_path / "openwa"
    assert s.clone_gateway(target, run) is True
    assert run.calls[0][0] == ["git", "clone", "--depth", "1", s.REPO_URL, str(target)]


def test_clone_is_skipped_when_a_checkout_already_exists(tmp_path):
    (tmp_path / s.COMPOSE_FILE).write_text("services: {}")
    run = FakeRun()
    assert s.clone_gateway(tmp_path, run) is False
    assert run.calls == []


def test_clone_refuses_to_use_an_unrelated_non_empty_directory(tmp_path):
    (tmp_path / "precious.txt").write_text("do not touch")
    with pytest.raises(s.SetupError, match="not an OpenWA checkout"):
        s.clone_gateway(tmp_path, FakeRun())


def test_empty_existing_directory_is_acceptable_for_cloning(tmp_path):
    assert s.clone_gateway(tmp_path, FakeRun()) is True


def test_gateway_env_is_created_once_and_never_overwritten(tmp_path):
    template = tmp_path / "template.env"
    template.write_text("BIND_HOST=127.0.0.1\n")
    gateway = tmp_path / "gw"
    gateway.mkdir()

    assert s.ensure_gateway_env(gateway, template) is True
    (gateway / ".env").write_text("BIND_HOST=127.0.0.1\nMY_EDIT=1\n")  # user customises
    assert s.ensure_gateway_env(gateway, template) is False
    assert "MY_EDIT=1" in (gateway / ".env").read_text()


def test_revision_is_reported_or_unknown():
    assert s.gateway_revision(Path("."), FakeRun({("git",): done(0, "abc1234\n")})) == "abc1234"
    assert s.gateway_revision(Path("."), FakeRun({("git",): OSError()})) == "unknown"


def test_compose_invocation_uses_the_dev_file_in_the_gateway_directory(tmp_path):
    run = FakeRun()
    s.compose(tmp_path, "up", "-d", "--build", run=run)
    cmd, kwargs = run.calls[0]
    assert cmd == ["docker", "compose", "-f", s.COMPOSE_FILE, "up", "-d", "--build"]
    assert kwargs["cwd"] == tmp_path and kwargs["check"] is True


# =====================================================================
# Health polling
# =====================================================================
class Clock:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_wait_until_ready_polls_until_200():
    t = Clock()
    responses = iter([SimpleNamespace(status_code=503), requests.ConnectionError("refused"), SimpleNamespace(status_code=200)])

    def fake_get(url, timeout):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        return item

    assert s.wait_until_ready(get=fake_get, clock=t.clock, sleep=t.sleep, interval=3) is True
    assert t.sleeps == [3, 3]


def test_wait_until_ready_gives_up_after_the_timeout():
    t = Clock()
    ok = s.wait_until_ready(get=lambda url, timeout: SimpleNamespace(status_code=503), clock=t.clock, sleep=t.sleep,
                            timeout=10, interval=4)
    assert ok is False and sum(t.sleeps) >= 10


def test_wait_with_zero_timeout_makes_exactly_one_attempt():
    calls = []
    s.wait_until_ready(get=lambda url, timeout: calls.append(url) or SimpleNamespace(status_code=500), timeout=0)
    assert calls == [s.HEALTH_URL]


# =====================================================================
# Admin key
# =====================================================================
def test_admin_key_is_read_from_the_host_file(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / ".api-key").write_text("owa_k1_secret\n")
    run = FakeRun()
    assert s.read_admin_key(tmp_path, run) == "owa_k1_secret"
    assert run.calls == []  # no docker call needed


def test_admin_key_falls_back_to_docker_exec_when_the_file_is_unreadable(tmp_path):
    run = FakeRun({("docker", "exec"): done(0, "owa_k1_from_container\n")})
    assert s.read_admin_key(tmp_path, run) == "owa_k1_from_container"
    assert run.calls[0][0] == ["docker", "exec", s.CONTAINER_NAME, "cat", "/app/data/.api-key"]


@pytest.mark.parametrize("result", [done(1, ""), done(0, "  \n"), OSError("no docker")])
def test_admin_key_missing_everywhere_returns_none(tmp_path, result):
    assert s.read_admin_key(tmp_path, FakeRun({("docker", "exec"): result})) is None


# =====================================================================
# Command orchestration
# =====================================================================
@pytest.fixture
def steps(monkeypatch, tmp_path):
    """Replace every side-effecting step with a recorder; returns the call log."""
    log = []
    monkeypatch.setattr(s, "check_prerequisites", lambda *a, **k: log.append("prereq") or [])
    monkeypatch.setattr(s, "clone_gateway", lambda d, *a, **k: log.append("clone") or True)
    monkeypatch.setattr(s, "gateway_revision", lambda *a, **k: "abc1234")
    monkeypatch.setattr(s, "ensure_gateway_env", lambda *a, **k: log.append("env") or True)
    monkeypatch.setattr(s, "compose", lambda d, *args, **k: log.append("compose " + " ".join(args)))
    monkeypatch.setattr(s, "wait_until_ready", lambda *a, **k: log.append("wait") or True)
    return log


def test_up_runs_the_steps_in_order(steps, tmp_path, capsys):
    assert s.cmd_up(tmp_path / "openwa") == 0
    assert steps == ["prereq", "clone", "env", "compose up -d --build", "wait"]
    out = capsys.readouterr().out
    assert "abc1234" in out and "Linked devices" in out and "DEDICATED TEST PHONE" in out


def test_up_stops_before_touching_anything_when_prerequisites_fail(steps, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(s, "check_prerequisites", lambda *a, **k: ["Docker is not installed"])
    assert s.cmd_up(tmp_path / "openwa") == 1
    assert steps == []
    assert "Docker is not installed" in capsys.readouterr().out


def test_up_reports_failure_and_a_log_hint_when_the_gateway_never_gets_healthy(steps, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(s, "wait_until_ready", lambda *a, **k: False)
    assert s.cmd_up(tmp_path / "openwa") == 1
    assert "logs openwa" in capsys.readouterr().out


def test_main_turns_a_failing_subprocess_into_a_message_not_a_traceback(monkeypatch, capsys):
    def boom(*a, **k):
        raise subprocess.CalledProcessError(128, ["git", "clone"])

    monkeypatch.setattr(s, "cmd_up", boom)
    assert s.main(["up"]) == 1
    assert "command failed (128)" in capsys.readouterr().out


def test_down_without_a_setup_is_a_clear_no_op(tmp_path, capsys):
    assert s.cmd_down(tmp_path) == 1
    assert "not been set up" in capsys.readouterr().out


def test_key_command_prints_only_the_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(s, "read_admin_key", lambda *a, **k: "owa_k1_abc")
    assert s.cmd_key(tmp_path) == 0
    assert capsys.readouterr().out.strip() == "owa_k1_abc"


def test_key_command_explains_the_fallback_when_no_key_exists(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(s, "read_admin_key", lambda *a, **k: None)
    assert s.cmd_key(tmp_path) == 1
    assert "owa_k1_" in capsys.readouterr().out


# =====================================================================
# Gateway config template: security invariants
# =====================================================================
def active_settings() -> dict[str, str]:
    lines = s.ENV_TEMPLATE.read_text(encoding="utf-8").splitlines()
    pairs = (line.split("=", 1) for line in lines if line.strip() and not line.lstrip().startswith("#"))
    return {k.strip(): v.strip() for k, v in pairs}


def test_gateway_is_bound_to_localhost_only():
    assert active_settings()["BIND_HOST"] == "127.0.0.1"


def test_the_public_dev_api_key_is_never_enabled():
    assert "ALLOW_DEV_API_KEY" not in active_settings()


def test_lower_ban_risk_engine_is_pinned():
    assert active_settings()["ENGINE_TYPE"] == "whatsapp-web.js"


def test_gateway_secrets_directory_is_git_ignored():
    gitignore = (s.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "openwa/" in gitignore  # holds WhatsApp login credentials + the admin key
