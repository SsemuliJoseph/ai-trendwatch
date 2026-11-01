"""
FILE: tests/test_delivery_test.py
MODULE 13 tests — run with:  pytest tests/test_delivery_test.py
A FakeClient stands in for the gateway: nothing here can send a real WhatsApp message.
"""
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from app.config.settings import Settings
from app.messaging.openwa_client import (
    OpenWAAuthError,
    OpenWADeliveryUnknownError,
    OpenWARateLimitedError,
    OpenWARejectedError,
    OpenWASessionError,
    OpenWAUnavailableError,
    SendResult,
)
from scripts import delivery_test as dt

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)
GROUP = "120363000000000000@g.us"
KEY = "owa_k1_SUPERSECRET"
SESSION = "8f3c2b1a-9d4e-4c7a-8b2f-1e6d5a4c3b2a"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)


def complete_settings(**kw) -> Settings:
    fields = dict(whatsapp_group_id=GROUP, openwa_api_key=KEY, openwa_session_id=SESSION)
    return Settings(_env_file=None, **{**fields, **kw})


class FakeClient:
    def __init__(self, healthy=True, ready_error=None, send_error=None):
        self.healthy, self.ready_error, self.send_error = healthy, ready_error, send_error
        self.sent = []

    def is_healthy(self):
        return self.healthy

    def ensure_ready(self):
        if self.ready_error:
            raise self.ready_error

    def send_text(self, chat_id, text):
        if self.send_error:
            raise self.send_error
        self.sent.append((chat_id, text))
        return SendResult(message_id="gw-123")

    def send_image(self, chat_id, image, mimetype, caption):
        if self.send_error:
            raise self.send_error
        self.images = getattr(self, "images", []) + [(chat_id, mimetype, len(image), caption)]
        return SendResult(message_id="gw-img-1")


def run(args=(), *, settings=None, client=None, answer="SEND", tty=True, tmp_path=None, now=NOW):
    """Invoke main() with everything injected. Returns (exit_code, client)."""
    client = client or FakeClient()
    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    code = dt.main(
        list(args),
        settings=settings or complete_settings(),
        client_factory=lambda s: client,
        input_fn=fake_input,
        isatty=lambda: tty,
        now_fn=lambda: now,
        state_path=(tmp_path / "state.json") if tmp_path else dt.STATE_FILE,
    )
    client.prompts = prompts
    return code, client


# =====================================================================
# Building blocks
# =====================================================================
def test_test_message_is_labelled_and_produced_by_the_production_formatter():
    text = dt.build_test_message(NOW).text
    assert "*[TEST] AI TrendWatch delivery check*" in text and text.startswith("📰 *")  # same layout as real items
    assert "19 Sep" in text and "https://example.com/" in text  # old date -> absolute date, not "Today"


def test_missing_live_config_lists_every_gap():
    problems = dt.missing_live_config(Settings(_env_file=None))
    assert len(problems) == 3
    assert any("GROUP_ID" in p for p in problems) and any("API_KEY" in p for p in problems) and any("SESSION_ID" in p for p in problems)
    assert dt.missing_live_config(complete_settings()) == []
    assert any("API_KEY" in p for p in dt.missing_live_config(complete_settings(openwa_api_key="change-me")))


def test_cooldown_logic(tmp_path):
    state = tmp_path / "s.json"
    assert dt.cooldown_remaining(state, NOW) == 0  # no file
    dt.record_test_send(state, NOW)
    assert dt.cooldown_remaining(state, NOW + timedelta(minutes=10)) == pytest.approx(20 * 60)
    assert dt.cooldown_remaining(state, NOW + timedelta(minutes=31)) == 0
    state.write_text("{corrupt")
    assert dt.cooldown_remaining(state, NOW) == 0  # a broken file must not lock you out forever


def test_record_creates_missing_directories(tmp_path):
    dt.record_test_send(tmp_path / "a" / "b" / "state.json", NOW)
    assert (tmp_path / "a" / "b" / "state.json").exists()


# =====================================================================
# Rehearsal (default): nothing is ever sent
# =====================================================================
def test_rehearsal_with_incomplete_config_explains_what_to_fill_in(capsys, tmp_path):
    code, client = run(settings=Settings(_env_file=None), tmp_path=tmp_path)
    out = capsys.readouterr().out
    assert code == dt.EXIT_PROBLEM and client.sent == []
    assert "WHATSAPP_GROUP_ID" in out and "Cannot continue" in out


def test_rehearsal_success_shows_the_message_and_sends_nothing(capsys, tmp_path):
    code, client = run(tmp_path=tmp_path)
    out = capsys.readouterr().out
    assert code == dt.EXIT_OK and client.sent == [] and client.prompts == []  # never even asks
    assert "REHEARSAL" in out and "[TEST] AI TrendWatch delivery check" in out and "[OK] gateway reachable" in out


@pytest.mark.parametrize(
    "client, expected",
    [
        (FakeClient(healthy=False), "gateway reachable"),
        (FakeClient(ready_error=OpenWASessionError("session is 'qr_ready', not 'ready'")), "qr_ready"),
    ],
)
def test_failed_preflight_blocks_everything(client, expected, capsys, tmp_path):
    code, client = run(["--send"], client=client, tmp_path=tmp_path)
    out = capsys.readouterr().out
    assert code == dt.EXIT_PROBLEM and client.sent == [] and client.prompts == []
    assert "[FAIL]" in out and expected in out and "Nothing was sent" in out


# =====================================================================
# --send: every gate
# =====================================================================
def test_send_requires_an_interactive_terminal(capsys, tmp_path):
    code, client = run(["--send"], tty=False, tmp_path=tmp_path)
    assert code == dt.EXIT_ABORTED and client.sent == [] and client.prompts == []
    assert "interactive terminal" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["", "send", "yes", "SEND it", "y"])
def test_anything_other_than_the_exact_word_aborts(answer, capsys, tmp_path):
    code, client = run(["--send"], answer=answer, tmp_path=tmp_path)
    assert code == dt.EXIT_ABORTED and client.sent == []
    assert "Aborted" in capsys.readouterr().out
    assert not (tmp_path / "state.json").exists()  # an aborted attempt must not start a cooldown


@pytest.mark.parametrize("interrupt", [EOFError(), KeyboardInterrupt()])
def test_ctrl_c_or_closed_stdin_at_the_prompt_aborts_cleanly(interrupt, tmp_path):
    code, client = run(["--send"], answer=interrupt, tmp_path=tmp_path)
    assert code == dt.EXIT_ABORTED and client.sent == []


def test_cooldown_blocks_a_second_send(capsys, tmp_path):
    dt.record_test_send(tmp_path / "state.json", NOW - timedelta(minutes=5))
    code, client = run(["--send"], tmp_path=tmp_path)
    out = capsys.readouterr().out
    assert code == dt.EXIT_ABORTED and client.sent == [] and client.prompts == []
    assert "25 more minute" in out


def test_confirmed_send_posts_exactly_one_message_to_the_configured_group(capsys, tmp_path):
    code, client = run(["--send"], tmp_path=tmp_path)
    out = capsys.readouterr().out
    assert code == dt.EXIT_OK
    assert len(client.sent) == 1
    chat_id, text = client.sent[0]
    assert chat_id == GROUP and text == dt.build_test_message(NOW).text
    assert "gw-123" in out and "CHECK THE GROUP" in out and "WHATSAPP_DRY_RUN=true" in out
    assert (tmp_path / "state.json").exists()  # cooldown started


def test_a_second_run_right_after_a_successful_send_is_blocked(tmp_path):
    run(["--send"], tmp_path=tmp_path)
    code, client = run(["--send"], tmp_path=tmp_path, now=NOW + timedelta(minutes=1))
    assert code == dt.EXIT_ABORTED and client.sent == []


def test_the_default_client_is_capped_at_one_message():
    client = dt._default_client(complete_settings())
    assert client._max_sends == 1


# =====================================================================
# --send: failure outcomes
# =====================================================================
@pytest.mark.parametrize(
    "error, phrase, cooldown_recorded",
    [
        (OpenWADeliveryUnknownError("no reply"), "LOOK before doing anything else", True),  # may be posted!
        (OpenWAAuthError("bad key"), "OPENWA_API_KEY", False),
        (OpenWASessionError("not ready"), "re-link", False),
        (OpenWARateLimitedError("slow", retry_after=30), "30s", False),
        (OpenWAUnavailableError("down"), "safe to try again", False),
        (OpenWARejectedError("bad text", status_code=422), "rejected this message", False),
    ],
)
def test_send_failures_give_specific_guidance(error, phrase, cooldown_recorded, capsys, tmp_path):
    code, client = run(["--send"], client=FakeClient(send_error=error), tmp_path=tmp_path)
    out = capsys.readouterr().out
    assert code == dt.EXIT_SEND_FAILED and "SEND FAILED" in out and phrase in out
    assert (tmp_path / "state.json").exists() is cooldown_recorded


# =====================================================================
# Hygiene
# =====================================================================
def test_the_api_key_is_never_printed(capsys, tmp_path):
    run(["--send"], tmp_path=tmp_path)
    run(["--send"], client=FakeClient(send_error=OpenWAAuthError("nope")), tmp_path=tmp_path, now=NOW + timedelta(hours=2))
    assert "SUPERSECRET" not in capsys.readouterr().out


def test_invalid_configuration_is_reported_without_a_traceback(monkeypatch, capsys):
    monkeypatch.setenv("LOG_LEVEL", "loud")
    dt.get_settings.cache_clear()
    try:
        code = dt.main([], client_factory=lambda s: FakeClient())
    finally:
        dt.get_settings.cache_clear()
    assert code == dt.EXIT_PROBLEM and "Invalid configuration" in capsys.readouterr().out


# =====================================================================
# Invocation: the documented command line must work outside pytest
# =====================================================================
@pytest.mark.parametrize("script", ["delivery_test.py", "setup_openwa.py"])
def test_scripts_run_as_plain_files_from_any_directory(script, tmp_path):
    """Regression: `python scripts/x.py` puts scripts/ (not the project root) on sys.path. pytest's
    `pythonpath` setting hides an `import app` failure, so this uses a real subprocess with no PYTHONPATH."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(dt.PROJECT_ROOT / "scripts" / script), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_image_flag_sends_one_picture_with_the_production_caption(tmp_path):
    code, client = run(["--send", "--image"], tmp_path=tmp_path)
    assert code == dt.EXIT_OK and client.sent == []  # no separate text message
    (chat, mimetype, size, caption), = client.images
    assert (chat, mimetype) == (GROUP, "image/png") and size > 1000
    assert caption == dt.build_test_message(NOW).text  # same layout as real items


def test_image_flag_is_ignored_in_rehearsal(tmp_path):
    code, client = run(["--image"], tmp_path=tmp_path)
    assert code == dt.EXIT_OK and client.sent == [] and not getattr(client, "images", [])
