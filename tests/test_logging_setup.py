"""
FILE: tests/test_logging_setup.py
MODULE 16 tests — run with:  pytest tests/test_logging_setup.py
Everything writes to tmp_path or in-memory streams; nothing touches your real logs/ folder.
"""
import io
import logging
import re

import pytest

from app.utils.logging_setup import (
    AUDIT_LOGGER,
    AUDIT_LOG_NAME,
    MAIN_LOG_NAME,
    REDACTED,
    SafeFormatter,
    configure_logging,
    neutralize,
    redact,
    reset_logging,
)

GEMINI_KEY = "AIzaSyD-1234567890abcdefghijklmnopqrstuv"
OPENWA_KEY = "owa_k1_SuperSecretValue123"
LINE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ (DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+\[[0-9a-f]{8}\] \S+: .+")


def read(path):
    return path.read_text(encoding="utf-8")


# =====================================================================
# Handlers, format, files
# =====================================================================
def test_console_and_file_get_the_same_utc_lines_with_a_run_id(tmp_path):
    stream = io.StringIO()
    run_id = configure_logging(level="INFO", log_dir=tmp_path / "logs", stream=stream, run_id="3f9a1c2e")
    logging.getLogger("app.demo").info("collected %d items", 42)

    for text in (stream.getvalue(), read(tmp_path / "logs" / MAIN_LOG_NAME)):
        assert run_id == "3f9a1c2e" and "[3f9a1c2e] app.demo: collected 42 items" in text
        assert LINE.match(text.strip())
        assert "Z INFO" in text  # UTC marker


def test_a_random_run_id_is_generated_when_none_is_given(tmp_path):
    a = configure_logging(stream=io.StringIO())
    b = configure_logging(stream=io.StringIO())
    assert re.fullmatch(r"[0-9a-f]{8}", a) and a != b


def test_level_filters_below_the_threshold(tmp_path):
    stream = io.StringIO()
    configure_logging(level="WARNING", log_dir=tmp_path, stream=stream)
    log = logging.getLogger("app.demo")
    log.info("quiet")
    log.warning("loud")
    assert "loud" in stream.getvalue() and "quiet" not in stream.getvalue()
    assert "quiet" not in read(tmp_path / MAIN_LOG_NAME)


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging(level="LOUD", stream=io.StringIO())


def test_log_directory_is_created_and_not_created_when_file_logging_is_off(tmp_path):
    configure_logging(log_dir=tmp_path / "a" / "b", stream=io.StringIO())
    assert (tmp_path / "a" / "b" / MAIN_LOG_NAME).exists()
    configure_logging(log_dir=None, stream=io.StringIO())
    assert not (tmp_path / "never").exists()


def test_console_can_be_disabled():
    stream = io.StringIO()
    configure_logging(stream=stream, console=False)
    logging.getLogger("app.demo").warning("nothing on the console")
    assert stream.getvalue() == ""


def test_configuring_twice_does_not_duplicate_lines(tmp_path):
    stream = io.StringIO()
    configure_logging(log_dir=tmp_path, stream=stream)
    configure_logging(log_dir=tmp_path, stream=stream)
    logging.getLogger("app.demo").warning("once")
    assert stream.getvalue().count("once") == 1
    assert read(tmp_path / MAIN_LOG_NAME).count("once") == 1


def test_reset_removes_only_our_handlers():
    foreign = logging.NullHandler()
    logging.getLogger().addHandler(foreign)
    try:
        configure_logging(stream=io.StringIO())
        reset_logging()
        assert foreign in logging.getLogger().handlers
        assert not [h for h in logging.getLogger().handlers if getattr(h, "_trendwatch_handler", False)]
    finally:
        logging.getLogger().removeHandler(foreign)


def test_files_are_size_capped_with_a_bounded_number_of_backups(tmp_path):
    configure_logging(log_dir=tmp_path, stream=io.StringIO(), max_bytes=1000, backup_count=2)
    log = logging.getLogger("app.demo")
    for i in range(300):
        log.info("line number %d with some padding to fill the file quickly", i)
    files = sorted(p.name for p in tmp_path.glob(f"{MAIN_LOG_NAME}*"))
    assert files == [MAIN_LOG_NAME, f"{MAIN_LOG_NAME}.1", f"{MAIN_LOG_NAME}.2"]  # never more than 1 + backup_count


def test_exceptions_are_logged_with_their_traceback(tmp_path):
    configure_logging(log_dir=tmp_path, stream=io.StringIO())
    try:
        1 / 0
    except ZeroDivisionError:
        logging.getLogger("app.demo").exception("run crashed")
    text = read(tmp_path / MAIN_LOG_NAME)
    assert "run crashed" in text and "Traceback (most recent call last)" in text and "ZeroDivisionError" in text


# =====================================================================
# Noisy third-party libraries
# =====================================================================
def test_http_client_chatter_is_hidden_unless_verbose(tmp_path):
    stream = io.StringIO()
    configure_logging(level="DEBUG", stream=stream)  # even at DEBUG for OUR code
    logging.getLogger("httpx").info("HTTP Request: POST https://example.com")
    logging.getLogger("httpx").warning("real problem")
    logging.getLogger("app.demo").debug("our own debug line")
    assert "HTTP Request" not in stream.getvalue()
    assert "real problem" in stream.getvalue() and "our own debug line" in stream.getvalue()

    stream = io.StringIO()
    configure_logging(level="DEBUG", stream=stream, third_party_level="INFO")
    logging.getLogger("httpx").info("HTTP Request: POST https://example.com")
    assert "HTTP Request" in stream.getvalue()


# =====================================================================
# Redaction
# =====================================================================
@pytest.mark.parametrize(
    "text",
    [
        f"key is {GEMINI_KEY}",
        f"header x-goog-api-key: {GEMINI_KEY}",
        f"X-API-Key: {OPENWA_KEY}",
        f"sending with {OPENWA_KEY} now",
        "Authorization: Bearer abcdefghijklmnop",
        "GET https://api.example.com/v1?key=abc123SECRET&page=2",
        "GET https://api.example.com/v1?page=2&api_key=abc123SECRET",
        "{'x-api-key': 'topsecretvalue'}",
    ],
)
def test_credentials_are_redacted_by_pattern(text):
    out = redact(text)
    assert REDACTED in out
    for leaked in (GEMINI_KEY, OPENWA_KEY, "abc123SECRET", "abcdefghijklmnop", "topsecretvalue"):
        assert leaked not in out


def test_known_secret_values_are_redacted_even_when_they_match_no_pattern():
    assert redact("token=hunter2hunter2 and hunter2hunter2 again", ["hunter2hunter2"]).count(REDACTED) >= 2
    assert "hunter2hunter2" not in redact("x hunter2hunter2 y", ["hunter2hunter2"])


def test_short_or_empty_secrets_never_blank_out_ordinary_text():
    # Values under 8 characters are ignored: replacing "true" or "abc" everywhere would wreck every log line.
    # (Template placeholders such as "change-me" are kept out earlier, by Settings.secret_values().)
    assert redact("true stays and abc stays", ["", "true", "abc"]) == "true stays and abc stays"


def test_normal_text_is_left_alone():
    text = "collect: 42 items, 3 errors, https://arxiv.org/abs/2609.01234"
    assert redact(text, [GEMINI_KEY]) == text


def test_secrets_never_reach_console_file_or_tracebacks(tmp_path):
    stream = io.StringIO()
    configure_logging(log_dir=tmp_path, stream=stream, secrets=["my-very-secret-token-value"])
    log = logging.getLogger("app.demo")
    log.warning("calling api with %s", "my-very-secret-token-value")
    try:
        raise RuntimeError(f"upstream said: bad key {OPENWA_KEY} / my-very-secret-token-value")
    except RuntimeError:
        log.exception("failed")
    logging.getLogger("httpx").warning("GET https://x.test/?key=%s", GEMINI_KEY)

    for text in (stream.getvalue(), read(tmp_path / MAIN_LOG_NAME)):
        assert "my-very-secret-token-value" not in text and OPENWA_KEY not in text and GEMINI_KEY not in text
        assert REDACTED in text


# =====================================================================
# Log injection
# =====================================================================
def test_neutralize_escapes_newlines_controls_and_bidi():
    assert neutralize("a\nb\r\tc") == "a\\nb\\r\\tc"
    assert neutralize("\x1b[2Jclear") == "\\x1b[2Jclear"  # ANSI escape can no longer reach a terminal
    assert neutralize("evil\u202etxt") == "evil\\u202etxt"
    assert neutralize("plain ünïcode ✓") == "plain ünïcode ✓"  # ordinary text untouched


def test_a_scraped_title_cannot_forge_log_lines(tmp_path):
    stream = io.StringIO()
    configure_logging(log_dir=tmp_path, stream=stream)
    evil = "Nice title\n2026-09-21T00:00:00Z ERROR   [deadbeef] app.fake: SYSTEM COMPROMISED"
    logging.getLogger("app.demo").info("new item: %s", evil)
    for text in (stream.getvalue(), read(tmp_path / MAIN_LOG_NAME)):
        assert len(text.strip().splitlines()) == 1  # still exactly one physical line
        assert "\\n2026-09-21T00:00:00Z ERROR" in text  # visible, but as inert text


def test_the_original_record_is_not_modified_for_other_handlers():
    record = logging.LogRecord("app.demo", logging.INFO, __file__, 1, "value %s", ("a\nb",), None)
    SafeFormatter("%(message)s").format(record)
    assert record.msg == "value %s" and record.args == ("a\nb",)


# =====================================================================
# Delivery audit log
# =====================================================================
def test_audit_lines_go_to_their_own_file_and_not_the_main_log(tmp_path):
    stream = io.StringIO()
    configure_logging(log_dir=tmp_path, stream=stream, run_id="abcd1234")
    logging.getLogger(AUDIT_LOGGER).info("attempt item=%s title=%r", "d79f11ca", "Notable story")
    logging.getLogger("app.demo").info("ordinary")

    audit = read(tmp_path / AUDIT_LOG_NAME)
    assert "[abcd1234] attempt item=d79f11ca title='Notable story'" in audit and "ordinary" not in audit
    assert "attempt item" not in read(tmp_path / MAIN_LOG_NAME) and "attempt item" not in stream.getvalue()


def test_audit_lines_are_on_disk_immediately_before_the_process_could_die(tmp_path):
    configure_logging(log_dir=tmp_path, stream=io.StringIO())
    logging.getLogger(AUDIT_LOGGER).info("attempt item=x")
    # read the file WHILE the handler is still open: a write-ahead line must not sit in a buffer
    assert "attempt item=x" in read(tmp_path / AUDIT_LOG_NAME)


def test_audit_lines_stay_visible_when_file_logging_is_off():
    stream = io.StringIO()
    configure_logging(level="INFO", log_dir=None, stream=stream)
    logging.getLogger(AUDIT_LOGGER).info("attempt item=y")
    assert "attempt item=y" in stream.getvalue()  # falls back to the console instead of vanishing
