"""
FILE: tests/test_main.py
MODULE 14 tests — run with:  pytest tests/test_main.py
Exercises main.py's composition root and CLI with injected fakes: no network, no API keys.
"""
from datetime import datetime, timedelta, timezone

import pytest

import main as runner
from app.ai.gemini_service import AnalysisRun, GeminiService
from app.collectors.arxiv import ArxivCollector
from app.collectors.rss import RSSCollector
from app.config.settings import ConfigError, Settings
from app.database.repository import TrendRepository
from app.messaging.openwa_client import DryRunClient, OpenWAAuthError, OpenWAClient, SendResult
from app.models import SourceType, TrendItem

GROUP = "120363000000000000@g.us"
SESSION = "8f3c2b1a-9d4e-4c7a-8b2f-1e6d5a4c3b2a"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)


def settings_for(tmp_path, **kw) -> Settings:
    # log_dir is ALWAYS redirected into tmp_path: a test run must never write into the real logs/ folder
    fields = dict(database_path=tmp_path / "data" / "t.db", gemini_api_key="AIzaSyREALKEY", log_dir=tmp_path / "logs")
    return Settings(_env_file=None, **{**fields, **kw})


LIVE = dict(whatsapp_dry_run=False, whatsapp_group_id=GROUP, openwa_api_key="owa_k1_SECRETKEY", openwa_session_id=SESSION)


def make(n: int) -> TrendItem:
    return TrendItem(
        title=f"Notable AI story number {n}", url=f"https://example.com/{n}", source_name="Src",
        source_type=SourceType.RSS, published_at=datetime.now(timezone.utc) - timedelta(hours=1 + n * 0.01),
    )


class FakeCollector:
    def __init__(self, items=()):
        self._items, self.failures = list(items), {}

    def collect(self):
        return list(self._items)


class FakeAnalyzer:
    def __init__(self, abort=False):
        self.abort, self.calls = abort, 0

    def analyze_items(self, items):
        items = list(items)
        self.calls += 1
        if self.abort:
            return AnalysisRun(unanalyzed=items, errors=["GeminiQuotaError: quota exceeded"], aborted=True)
        return AnalysisRun(analyzed=[i.with_analysis(9, f"Summary of {i.title}.") for i in items])


class FakeGateway:
    def __init__(self, error=None):
        self.error, self.sent = error, []

    def ensure_ready(self):
        return None

    def send_text(self, chat_id, text):
        if self.error:
            raise self.error
        self.sent.append((chat_id, text))
        return SendResult(message_id=f"m{len(self.sent)}")


# =====================================================================
# build_deps: the composition root
# =====================================================================
def test_defaults_are_dry_run_with_a_placeholder_destination(tmp_path):
    deps = runner.build_deps(settings_for(tmp_path), repository=TrendRepository(":memory:"))
    assert isinstance(deps.gateway, DryRunClient) and deps.gateway._max_sends == 5
    assert deps.chat_id == runner.DRY_RUN_CHAT_ID


def test_a_configured_group_is_used_as_the_destination_even_in_dry_run(tmp_path):
    deps = runner.build_deps(settings_for(tmp_path, whatsapp_group_id=GROUP), repository=TrendRepository(":memory:"))
    assert isinstance(deps.gateway, DryRunClient) and deps.chat_id == GROUP


def test_live_settings_build_the_real_client(tmp_path):
    deps = runner.build_deps(settings_for(tmp_path, **LIVE), repository=TrendRepository(":memory:"))
    assert isinstance(deps.gateway, OpenWAClient) and deps.chat_id == GROUP


def test_force_dry_run_overrides_live_settings_and_never_builds_the_real_client(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("the real gateway must not be constructed in a forced dry run")

    monkeypatch.setattr(runner, "build_gateway", forbidden)
    deps = runner.build_deps(settings_for(tmp_path, **LIVE), force_dry_run=True, repository=TrendRepository(":memory:"))
    assert isinstance(deps.gateway, DryRunClient)


def test_default_sources_and_analyzer_are_built_from_settings(tmp_path):
    settings = settings_for(tmp_path, max_items_per_source=7, request_timeout_seconds=9, gemini_model="my-model")
    deps = runner.build_deps(settings, repository=TrendRepository(":memory:"))
    rss, arxiv = deps.collectors
    assert isinstance(rss, RSSCollector) and (rss._max_items, rss._timeout) == (7, 9)
    assert isinstance(arxiv, ArxivCollector) and (arxiv._max_items, arxiv._timeout) == (7, 9)
    assert isinstance(deps.analyzer, GeminiService) and deps.analyzer._model == "my-model"
    assert (deps.config.min_relevance_score, deps.config.max_analyze_per_run, deps.config.max_messages_per_run) == (6, 20, 5)


def test_a_missing_gemini_key_is_a_configuration_error_raised_before_the_database_is_touched(tmp_path):
    settings = settings_for(tmp_path, gemini_api_key="")
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        runner.build_deps(settings)
    assert not settings.database_path.exists() and not settings.database_path.parent.exists()


# =====================================================================
# main(): dry run
# =====================================================================
def run_main(tmp_path, *args, settings=None, **overrides):
    overrides.setdefault("collectors", [FakeCollector(make(i) for i in range(1, 4))])
    overrides.setdefault("analyzer", FakeAnalyzer())
    return runner.main(list(args), settings=settings or settings_for(tmp_path), **overrides)


def test_dry_run_prints_summary_previews_and_says_nothing_was_sent(tmp_path, capsys):
    code = run_main(tmp_path)
    out = capsys.readouterr().out
    assert code == runner.EXIT_OK
    assert "collected 3" in out and "analysed 3" in out and "previewed (dry run) 3" in out
    assert "DRY RUN: 3 message(s) WOULD be posted" in out and "*Notable AI story number" in out
    assert "nothing was sent" in out


def test_the_database_is_created_and_dry_run_leaves_items_queued_for_a_later_live_run(tmp_path):
    settings = settings_for(tmp_path)
    run_main(tmp_path, settings=settings)
    with TrendRepository(settings.database_path) as repo:
        assert repo.count_by_status() == {"new": 0, "analyzed": 3, "sending": 0, "sent": 0, "rejected": 0}


def test_a_repository_passed_in_stays_open_but_one_created_by_main_is_closed(tmp_path):
    repo = TrendRepository(":memory:")
    run_main(tmp_path, repository=repo)
    assert repo.count_by_status()["analyzed"] == 3  # still usable: main did not close a repo it does not own
    repo.close()


# =====================================================================
# main(): live path (fake gateway)
# =====================================================================
def test_live_run_delivers_marks_sent_and_a_second_run_posts_nothing(tmp_path, capsys):
    gateway = FakeGateway()
    settings = settings_for(tmp_path, **LIVE)
    assert run_main(tmp_path, settings=settings, gateway=gateway) == runner.EXIT_OK
    assert len(gateway.sent) == 3 and all(chat == GROUP for chat, _ in gateway.sent)
    assert "delivered 3" in capsys.readouterr().out

    run_main(tmp_path, settings=settings, gateway=gateway, collectors=[FakeCollector(make(i) for i in range(1, 4))])
    assert len(gateway.sent) == 3  # nothing new, nothing re-posted


def test_forced_dry_run_flag_beats_live_settings(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "build_gateway", lambda s: (_ for _ in ()).throw(AssertionError("live gateway built")))
    code = run_main(tmp_path, "--dry-run", settings=settings_for(tmp_path, **LIVE))
    assert code == runner.EXIT_OK and "DRY RUN" in capsys.readouterr().out


# =====================================================================
# Exit codes and hygiene
# =====================================================================
def test_missing_gemini_key_exits_with_a_readable_message(tmp_path, capsys):
    code = runner.main([], settings=settings_for(tmp_path, gemini_api_key=""), collectors=[FakeCollector()])
    out = capsys.readouterr().out
    assert code == runner.EXIT_CONFIG and "Configuration problem" in out and "GEMINI_API_KEY" in out


def test_quota_abort_exits_2_but_still_reports(tmp_path, capsys):
    code = run_main(tmp_path, analyzer=FakeAnalyzer(abort=True))
    out = capsys.readouterr().out
    assert code == runner.EXIT_FATAL and "GeminiQuotaError" in out and "Problems (1)" in out


def test_a_fatal_delivery_error_exits_2(tmp_path, capsys):
    code = run_main(tmp_path, settings=settings_for(tmp_path, **LIVE), gateway=FakeGateway(error=OpenWAAuthError("bad key")))
    assert code == runner.EXIT_FATAL and "OpenWAAuthError" in capsys.readouterr().out


def test_a_dead_source_is_reported_but_is_not_a_failure(tmp_path, capsys):
    broken = FakeCollector([make(1)])
    broken.failures = {"Feed X": "HTTPError: 500"}
    code = run_main(tmp_path, collectors=[broken])
    assert code == runner.EXIT_OK and "Feed X" in capsys.readouterr().out


def test_no_secret_is_ever_printed(tmp_path, capsys):
    run_main(tmp_path, settings=settings_for(tmp_path, **LIVE), gateway=FakeGateway())
    run_main(tmp_path, settings=settings_for(tmp_path, **LIVE), gateway=FakeGateway(error=OpenWAAuthError("nope")),
             collectors=[FakeCollector([make(9)])])
    out = capsys.readouterr().out
    assert "REALKEY" not in out and "SECRETKEY" not in out


def test_format_summary_is_readable_with_partial_state():
    text = runner.format_summary({"metrics": {"collected": 5, "valid": 4}, "errors": ["a problem"]})
    assert "collected 5 | valid 4" in text and "Problems (1):" in text and "- a problem" in text
    assert "delivered 0" in text  # missing counters default to 0 instead of crashing


# =====================================================================
# Module 15: single instance, crash handling, retry wiring
# =====================================================================
import os  # noqa: E402

from app.utils.runlock import RunLock  # noqa: E402


def lock_file(tmp_path):
    return tmp_path / "data" / ".run.lock"


def test_a_second_run_is_skipped_while_the_first_holds_the_lock(tmp_path, capsys):
    holder = RunLock(lock_file(tmp_path), pid=lambda: 4242)
    holder.acquire()
    analyzer = FakeAnalyzer()
    code = run_main(tmp_path, analyzer=analyzer)
    out = capsys.readouterr().out
    assert code == runner.EXIT_BUSY and "Skipping this run" in out and "pid 4242" in out
    assert analyzer.calls == 0 and not (tmp_path / "data" / "t.db").exists()  # nothing touched, not even the database
    holder.release()


def test_the_lock_is_released_after_a_normal_run_so_the_next_one_can_start(tmp_path):
    assert run_main(tmp_path) == runner.EXIT_OK
    assert not lock_file(tmp_path).exists()
    assert run_main(tmp_path) == runner.EXIT_OK


def test_a_stale_lock_from_a_crashed_run_does_not_block_forever(tmp_path):
    RunLock(lock_file(tmp_path), pid=lambda: 1).acquire()  # never released
    old = lock_file(tmp_path).stat().st_mtime - 3 * 3600
    os.utime(lock_file(tmp_path), (old, old))
    assert run_main(tmp_path) == runner.EXIT_OK


def test_an_unexpected_crash_is_reported_exits_3_and_still_releases_the_lock(tmp_path, capsys):
    class Exploding(FakeAnalyzer):
        def analyze_items(self, items):
            raise RuntimeError("kaboom")

    code = run_main(tmp_path, analyzer=Exploding())
    out = capsys.readouterr().out
    assert code == runner.EXIT_CRASH and "RuntimeError: kaboom" in out and "never re-sent" in out
    assert not lock_file(tmp_path).exists()


def test_ctrl_c_still_releases_the_lock(tmp_path):
    class Interrupted(FakeAnalyzer):
        def analyze_items(self, items):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_main(tmp_path, analyzer=Interrupted())
    assert not lock_file(tmp_path).exists()


def test_a_configuration_error_also_releases_the_lock(tmp_path):
    assert runner.main([], settings=settings_for(tmp_path, gemini_api_key=""), collectors=[FakeCollector()]) == runner.EXIT_CONFIG
    assert not lock_file(tmp_path).exists()


def test_the_lock_path_and_staleness_come_from_settings(tmp_path):
    settings = settings_for(tmp_path, run_lock_stale_minutes=5)
    RunLock(lock_file(tmp_path), pid=lambda: 1).acquire()
    old = lock_file(tmp_path).stat().st_mtime - 6 * 60  # older than the configured 5 minutes
    os.utime(lock_file(tmp_path), (old, old))
    assert run_main(tmp_path, settings=settings) == runner.EXIT_OK


def test_the_retry_policy_from_settings_reaches_every_network_component(tmp_path):
    settings = settings_for(tmp_path, retry_max_attempts=4, retry_base_delay_seconds=1.5)
    deps = runner.build_deps(settings, repository=TrendRepository(":memory:"))
    rss, arxiv = deps.collectors
    for component in (rss, arxiv, deps.analyzer):
        assert (component._retry.max_attempts, component._retry.base_delay) == (4, 1.5)


def test_exit_codes_are_distinct():
    codes = [runner.EXIT_OK, runner.EXIT_CONFIG, runner.EXIT_FATAL, runner.EXIT_CRASH, runner.EXIT_BUSY]
    assert len(set(codes)) == 5


def test_interrupted_deliveries_are_mentioned_in_the_summary():
    text = runner.format_summary({"metrics": {"interrupted_recovered": 2}})
    assert "interrupted earlier deliveries" in text and "(not re-sent): 2" in text


# =====================================================================
# Module 16: logging
# =====================================================================
import re  # noqa: E402


def main_log(tmp_path):
    return (tmp_path / "logs" / "trendwatch.log").read_text(encoding="utf-8")


def audit_log(tmp_path):
    return (tmp_path / "logs" / "deliveries.log").read_text(encoding="utf-8")


def test_test_settings_never_point_at_the_real_logs_folder(tmp_path):
    assert settings_for(tmp_path).log_dir == tmp_path / "logs"


def test_a_run_logs_a_start_and_a_finish_line_sharing_one_run_id(tmp_path):
    run_main(tmp_path)
    log = main_log(tmp_path)
    started = re.search(r"\[([0-9a-f]{8})\] main: run started", log) or re.search(r"\[([0-9a-f]{8})\] __main__: run started", log)
    finished = re.search(r"\[([0-9a-f]{8})\] \S+: run finished: exit=0 metrics=\{.*'collected': 3", log)
    assert started and finished and started.group(1) == finished.group(1)


def test_problems_are_written_to_the_log_as_warnings(tmp_path):
    broken = FakeCollector([make(1)])
    broken.failures = {"Feed X": "HTTPError: 500"}
    run_main(tmp_path, collectors=[broken])
    assert re.search(r"WARNING .*problem: FakeCollector\[Feed X\]: HTTPError: 500", main_log(tmp_path))


def test_no_secret_reaches_the_console_or_the_log_even_when_a_crash_message_contains_it(tmp_path, capsys):
    class Leaky(FakeAnalyzer):
        def analyze_items(self, items):
            raise RuntimeError("upstream rejected AIzaSyREALKEY and owa_k1_SECRETKEY")

    code = run_main(tmp_path, settings=settings_for(tmp_path, **LIVE), analyzer=Leaky(), gateway=FakeGateway())
    captured = capsys.readouterr()
    assert code == runner.EXIT_CRASH
    for text in (captured.out, captured.err, main_log(tmp_path)):
        assert "REALKEY" not in text and "SECRETKEY" not in text
    assert "[REDACTED]" in main_log(tmp_path) and "Traceback" in main_log(tmp_path)


def test_verbose_adds_debug_lines_and_the_configuration_dump_is_masked(tmp_path):
    run_main(tmp_path, "--verbose", settings=settings_for(tmp_path, **LIVE), gateway=FakeGateway())
    log = main_log(tmp_path)
    assert "DEBUG" in log and "configuration:" in log
    assert "'gemini_api_key': '<set>'" in log and "REALKEY" not in log and "SECRETKEY" not in log


def test_normal_runs_do_not_emit_debug_lines(tmp_path):
    run_main(tmp_path)
    assert " DEBUG " not in main_log(tmp_path)


def test_file_logging_can_be_turned_off(tmp_path, capsys):
    run_main(tmp_path, settings=settings_for(tmp_path, log_to_file=False))
    assert not (tmp_path / "logs").exists()
    assert "run finished" in capsys.readouterr().err  # the console still gets the lines (stderr)


def test_a_skipped_run_is_logged(tmp_path):
    holder = RunLock(lock_file(tmp_path), pid=lambda: 4242)
    holder.acquire()
    run_main(tmp_path)
    assert re.search(r"WARNING .*run skipped: another run is active", main_log(tmp_path))
    holder.release()


def test_every_real_send_leaves_an_attempt_and_a_sent_line_in_the_audit_log(tmp_path):
    run_main(tmp_path, settings=settings_for(tmp_path, **LIVE), gateway=FakeGateway())
    audit = audit_log(tmp_path)
    assert audit.count(" attempt item=") == 3 and audit.count(" sent item=") == 3
    assert "attempt" not in main_log(tmp_path)  # audit trail is kept apart from the main log
    assert f"chat={GROUP}" in audit and "gateway_id=m1" in audit


def test_a_dry_run_audits_previews_never_sends(tmp_path):
    run_main(tmp_path, "--dry-run")
    audit = audit_log(tmp_path)
    assert audit.count(" preview item=") == 3 and " sent item=" not in audit and " attempt item=" not in audit


def test_after_a_kill_mid_send_the_audit_log_names_the_message_to_look_for(tmp_path):
    class DiesOnSecond(FakeGateway):
        def send_text(self, chat_id, text):
            if self.sent:
                raise KeyboardInterrupt  # stands in for the process being stopped mid-send
            return super().send_text(chat_id, text)

    with pytest.raises(KeyboardInterrupt):
        run_main(tmp_path, settings=settings_for(tmp_path, **LIVE), gateway=DiesOnSecond())
    lines = audit_log(tmp_path).splitlines()
    attempts = [l for l in lines if " attempt item=" in l]
    sent_ids = {re.search(r"item=(\w+)", l).group(1) for l in lines if " sent item=" in l}
    unresolved = [l for l in attempts if re.search(r"item=(\w+)", l).group(1) not in sent_ids]
    assert len(attempts) == 2 and len(unresolved) == 1 and "title='Notable AI story number" in unresolved[0]
