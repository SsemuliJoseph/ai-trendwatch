"""
FILE: tests/test_scheduler.py
MODULE 17 tests — run with:  pytest tests/test_scheduler.py
No real waiting: `--once` and a fake scheduler cover the CLI; the real APScheduler wiring is checked by
inspecting the built (unstarted) job, not by actually running its loop.
"""
import signal

import pytest

import main as runner
from app.config.settings import ConfigError, Settings
from scripts import scheduler as sched


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)


def settings_for(tmp_path, **kw) -> Settings:
    fields = dict(database_path=tmp_path / "data" / "t.db", gemini_api_key="AIzaSyREALKEY", log_dir=tmp_path / "logs")
    return Settings(_env_file=None, **{**fields, **kw})


# =====================================================================
# run_job: never lets an exception escape, always returns a code
# =====================================================================
def test_success_prints_the_ok_label_and_returns_the_code(capsys):
    assert sched.run_job(lambda: runner.EXIT_OK, tick=3) == runner.EXIT_OK
    out = capsys.readouterr().out
    assert "tick 3: starting" in out and "tick 3: finished (ok, exit 0)" in out  # label AND the raw code, for grepping


@pytest.mark.parametrize(
    "code, label",
    [
        (runner.EXIT_CONFIG, "configuration problem"),
        (runner.EXIT_FATAL, "fatal condition"),
        (runner.EXIT_CRASH, "crashed unexpectedly"),
        (runner.EXIT_BUSY, "skipped (another run was already active)"),
        (99, "unrecognised exit code"),  # an exit code we did not anticipate still gets a readable line, not a crash
    ],
)
def test_every_exit_code_gets_a_readable_label(code, label, capsys):
    assert sched.run_job(lambda: code, tick=1) == code
    assert label in capsys.readouterr().out


def test_an_exception_from_run_fn_is_contained_and_reported(capsys):
    def boom():
        raise RuntimeError("pipeline exploded")

    assert sched.run_job(boom, tick=7) == runner.EXIT_CRASH
    out = capsys.readouterr().out
    assert "tick 7: run_fn raised RuntimeError: pipeline exploded" in out and "scheduler keeps going" in out


def test_keyboard_interrupt_is_not_swallowed():
    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        sched.run_job(interrupted, tick=1)


# =====================================================================
# build_scheduler: wiring, without ever starting the real loop
# =====================================================================
def one_job(scheduler):
    (job,) = scheduler.get_jobs()
    return job


def test_default_interval_comes_from_settings(tmp_path):
    scheduler, _ = sched.build_scheduler(settings_for(tmp_path, schedule_interval_minutes=45))
    assert one_job(scheduler).trigger.interval.total_seconds() == 45 * 60


def test_an_explicit_interval_overrides_settings(tmp_path):
    scheduler, _ = sched.build_scheduler(settings_for(tmp_path), interval_minutes=5)
    assert one_job(scheduler).trigger.interval.total_seconds() == 5 * 60


def test_zero_or_negative_interval_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        sched.build_scheduler(settings_for(tmp_path), interval_minutes=0)


def test_the_job_has_both_layers_of_overlap_protection(tmp_path):
    job = one_job(sched.build_scheduler(settings_for(tmp_path))[0])
    assert job.max_instances == 1 and job.coalesce is True and job.misfire_grace_time == 60
    assert job.id == sched.JOB_ID


def test_dry_run_makes_the_default_run_fn_pass_dash_dash_dry_run_to_main(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(runner, "main", lambda argv=None, **kw: captured.setdefault("argv", argv) or 0)
    _, run_fn = sched.build_scheduler(settings_for(tmp_path), force_dry_run=True)
    run_fn()
    assert captured["argv"] == ["--dry-run"]


def test_without_dry_run_the_default_run_fn_calls_main_with_no_extra_args(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(runner, "main", lambda argv=None, **kw: captured.setdefault("argv", argv) or 0)
    _, run_fn = sched.build_scheduler(settings_for(tmp_path))
    run_fn()
    assert captured["argv"] == []


def test_a_custom_run_fn_replaces_the_default_entirely(tmp_path):
    calls = []
    _, run_fn = sched.build_scheduler(settings_for(tmp_path), run_fn=lambda: calls.append(1) or 0)
    run_fn()
    assert calls == [1]


def test_a_custom_scheduler_factory_is_used_instead_of_the_real_one(tmp_path):
    built = []

    class Fake:
        def __init__(self, **kw):
            built.append(kw)

        def add_job(self, *a, **kw):
            self.job = kw

        def get_jobs(self):
            return [type("J", (), self.job)()]

    scheduler, _ = sched.build_scheduler(settings_for(tmp_path), scheduler_factory=Fake)
    assert isinstance(scheduler, Fake) and built  # the real BlockingScheduler was never touched


# =====================================================================
# CLI: --once (never starts the real loop)
# =====================================================================
def test_once_runs_exactly_one_tick_and_never_builds_a_real_running_scheduler(tmp_path, capsys):
    calls = []
    code = sched.main(["--once"], settings=settings_for(tmp_path), run_fn=lambda: calls.append(1) or runner.EXIT_OK)
    assert code == runner.EXIT_OK and calls == [1]
    assert "tick 1: starting" in capsys.readouterr().out


def test_once_returns_whatever_the_tick_returned():
    assert sched.main(["--once"], run_fn=lambda: runner.EXIT_FATAL, settings=Settings(_env_file=None, gemini_api_key="k")) == runner.EXIT_FATAL


def test_once_with_a_crashing_run_fn_still_exits_cleanly_with_a_readable_code(tmp_path):
    def boom():
        raise RuntimeError("x")

    assert sched.main(["--once"], settings=settings_for(tmp_path), run_fn=boom) == runner.EXIT_CRASH


# =====================================================================
# CLI: configuration and validation
# =====================================================================
def test_a_missing_gemini_key_is_a_configuration_error(tmp_path):
    # The check lives in build_deps (the composition root every tick goes through). This used to call
    # `sched.main(["--once"])` with NO run_fn, which executes runner.main() -> get_settings() -> the developer's
    # REAL .env: a full live pipeline (real Gemini calls, real WhatsApp delivery) inside a unit test.
    # Rule for this file: any call that can reach a tick must inject `run_fn`.
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        runner.build_deps(settings_for(tmp_path, gemini_api_key=""))


def test_negative_interval_flag_is_rejected_before_anything_is_built(tmp_path, capsys):
    code = sched.main(["--interval", "-1"], settings=settings_for(tmp_path))
    assert code == runner.EXIT_CONFIG and "must be >= 1" in capsys.readouterr().out


def test_zero_interval_flag_is_rejected(tmp_path):
    assert sched.main(["--interval", "0"], settings=settings_for(tmp_path)) == runner.EXIT_CONFIG


# =====================================================================
# CLI: continuous mode, via a fake scheduler (never actually blocks)
# =====================================================================
class FakeScheduler:
    """Enough of BlockingScheduler's surface for main()'s continuous-mode branch."""

    def __init__(self, on_start=None, **kw):
        self.started = False
        self.shutdown_called = False
        self._on_start = on_start or (lambda: None)

    def add_job(self, *a, **kw):
        pass

    def get_jobs(self):
        return []

    def start(self):
        self.started = True
        self._on_start()  # simulates whatever happens while "running" (e.g. Ctrl+C arriving)

    def shutdown(self, wait=True):
        self.shutdown_called = True


def factory(on_start=None):
    return lambda **kw: FakeScheduler(on_start=on_start, **kw)


def test_run_at_start_fires_tick_zero_before_the_loop_starts(tmp_path, capsys):
    calls = []
    sched.main(
        [], settings=settings_for(tmp_path), run_fn=lambda: calls.append(1) or runner.EXIT_OK,
        scheduler_factory=factory(),
    )
    assert calls == [1]
    assert "tick 0: starting" in capsys.readouterr().out


def test_skip_first_run_flag_means_no_immediate_tick(tmp_path, capsys):
    calls = []
    sched.main(
        ["--skip-first-run"], settings=settings_for(tmp_path), run_fn=lambda: calls.append(1) or runner.EXIT_OK,
        scheduler_factory=factory(),
    )
    assert calls == []
    out = capsys.readouterr().out
    assert "First run in" in out and "tick 0" not in out


def test_schedule_run_immediately_setting_is_honoured(tmp_path):
    calls = []
    sched.main(
        [], settings=settings_for(tmp_path, schedule_run_immediately=False),
        run_fn=lambda: calls.append(1) or runner.EXIT_OK, scheduler_factory=factory(),
    )
    assert calls == []


def test_ctrl_c_during_start_stops_gracefully_and_returns_ok(tmp_path, capsys):
    built = {}

    def raise_interrupt():
        raise KeyboardInterrupt

    def make(**kw):
        s = FakeScheduler(on_start=raise_interrupt, **kw)
        built["scheduler"] = s
        return s

    code = sched.main([], settings=settings_for(tmp_path), run_fn=lambda: runner.EXIT_OK, scheduler_factory=make)
    assert code == runner.EXIT_OK
    assert built["scheduler"].started and built["scheduler"].shutdown_called
    out = capsys.readouterr().out
    assert "Stopping: letting the current tick" in out and "Scheduler stopped." in out


def test_sigterm_is_treated_like_ctrl_c(tmp_path, capsys):
    def send_sigterm():
        signal.raise_signal(signal.SIGTERM)  # what a service manager's "stop" does

    code = sched.main([], settings=settings_for(tmp_path), run_fn=lambda: runner.EXIT_OK, scheduler_factory=factory(send_sigterm))
    assert code == runner.EXIT_OK
    assert "Scheduler stopped." in capsys.readouterr().out


def test_the_sigterm_handler_is_restored_after_the_run(tmp_path):
    original = signal.getsignal(signal.SIGTERM)
    sched.main([], settings=settings_for(tmp_path), run_fn=lambda: runner.EXIT_OK, scheduler_factory=factory(lambda: (_ for _ in ()).throw(KeyboardInterrupt())))
    assert signal.getsignal(signal.SIGTERM) is original


def test_startup_banner_mentions_the_interval_and_dry_run_flag(tmp_path, capsys):
    sched.main(["--interval", "10", "--dry-run"], settings=settings_for(tmp_path),
              run_fn=lambda: runner.EXIT_OK, scheduler_factory=factory())
    out = capsys.readouterr().out
    assert "every 10 minute(s)" in out and "dry run forced" in out


# =====================================================================
# Invocation: the documented command line must work outside pytest
# =====================================================================
def test_runs_as_a_plain_file_from_any_directory(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "scheduler.py"), "--help"],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0 and "usage:" in result.stdout, result.stderr
