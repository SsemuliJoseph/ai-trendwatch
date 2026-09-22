"""
FILE: tests/test_settings.py
Tests for app/config/settings.py (Module 2 code, first covered here because Module 11 added
the OpenWA guards). `_env_file=None` + a scrubbed environment make every test independent of
YOUR real .env and shell variables.
"""
import pytest
from pydantic import ValidationError

from app.config.settings import PROJECT_ROOT, ConfigError, Settings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)


def build(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


LIVE = dict(
    whatsapp_dry_run=False,
    whatsapp_group_id="120363000000000000@g.us",
    openwa_api_key="owa_k1_realkey",
    openwa_session_id="8f3c2b1a-9d4e-4c7a-8b2f-1e6d5a4c3b2a",
)


# ---------- defaults ----------
def test_safe_defaults():
    s = build()
    assert s.whatsapp_dry_run is True  # never sends until deliberately switched off
    assert s.openwa_base_url == "http://127.0.0.1:2785"
    assert s.openwa_session_id == ""
    assert (s.min_relevance_score, s.max_analyze_per_run, s.max_messages_per_run) == (6, 20, 5)


def test_database_path_is_anchored_to_the_project_root():
    assert build().database_path == PROJECT_ROOT / "data" / "trendwatch.db"
    assert build(database_path="/tmp/x.db").database_path.as_posix() == "/tmp/x.db"


def test_settings_are_immutable():
    with pytest.raises(ValidationError):
        build().log_level = "DEBUG"


# ---------- live-sending guard ----------
def test_live_sending_is_allowed_only_when_everything_is_configured():
    assert build(**LIVE).whatsapp_dry_run is False


@pytest.mark.parametrize(
    "missing, message",
    [
        ("whatsapp_group_id", "WHATSAPP_GROUP_ID"),
        ("openwa_api_key", "OPENWA_API_KEY"),
        ("openwa_session_id", "OPENWA_SESSION_ID"),
    ],
)
def test_live_sending_requires_each_of_group_key_and_session(missing, message):
    config = {**LIVE, missing: ""}
    with pytest.raises(ValidationError, match=message):
        build(**config)


def test_placeholder_api_key_does_not_count_as_configured():
    with pytest.raises(ValidationError, match="OPENWA_API_KEY"):
        build(**{**LIVE, "openwa_api_key": "change-me"})


def test_dry_run_needs_no_whatsapp_configuration():
    build(whatsapp_dry_run=True)  # no group, key or session: fine


# ---------- field validation ----------
@pytest.mark.parametrize("bad", ["256700000000@c.us", "120363000000000000", "someone@s.whatsapp.net"])
def test_only_group_ids_are_accepted(bad):
    with pytest.raises(ValidationError, match="GROUP id"):
        build(whatsapp_group_id=bad)


def test_group_id_is_trimmed():
    assert build(whatsapp_group_id="  120363000000000000@g.us ").whatsapp_group_id == "120363000000000000@g.us"


def test_base_url_is_normalised_and_validated():
    assert build(openwa_base_url=" http://127.0.0.1:2785/ ").openwa_base_url == "http://127.0.0.1:2785"
    with pytest.raises(ValidationError):
        build(openwa_base_url="127.0.0.1:2785")


def test_log_level_is_case_insensitive_and_restricted():
    assert build(log_level=" debug ").log_level == "DEBUG"
    with pytest.raises(ValidationError):
        build(log_level="loud")


@pytest.mark.parametrize("field, value", [("min_relevance_score", 11), ("max_analyze_per_run", 0), ("max_messages_per_run", 21)])
def test_numeric_bounds(field, value):
    with pytest.raises(ValidationError):
        build(**{field: value})


# ---------- secrets ----------
def test_secrets_are_masked_everywhere_they_could_leak():
    s = build(gemini_api_key="AIzaSySECRET", openwa_api_key="owa_k1_SECRET")
    assert "SECRET" not in repr(s) and "SECRET" not in str(s)
    dump = s.safe_dump()
    assert dump["gemini_api_key"] == "<set>" and dump["openwa_api_key"] == "<set>"
    assert build().safe_dump()["gemini_api_key"] == "<missing>"


def test_require_gemini_key():
    assert build(gemini_api_key="AIzaSyReal").require_gemini_key() == "AIzaSyReal"
    for unusable in ("", "your-gemini-api-key-here"):
        with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
            build(gemini_api_key=unusable).require_gemini_key()


# ---------- Module 15: resilience settings ----------
def test_resilience_defaults():
    s = build()
    assert (s.retry_max_attempts, s.retry_base_delay_seconds, s.run_lock_stale_minutes) == (3, 2.0, 30)


@pytest.mark.parametrize(
    "field, value",
    [("retry_max_attempts", 0), ("retry_max_attempts", 7), ("retry_base_delay_seconds", 0.0),
     ("retry_base_delay_seconds", 61), ("run_lock_stale_minutes", 0)],
)
def test_resilience_bounds(field, value):
    with pytest.raises(ValidationError):
        build(**{field: value})


# ---------- Module 16: logging settings + secret_values ----------
def test_logging_defaults_and_anchoring():
    s = build()
    assert (s.log_to_file, s.log_max_mb, s.log_backup_count) == (True, 2, 5)
    assert s.log_dir == PROJECT_ROOT / "logs"  # relative default is anchored to the project root
    assert build(log_dir="/var/log/trendwatch").log_dir.as_posix() == "/var/log/trendwatch"


@pytest.mark.parametrize("field, value", [("log_max_mb", 0), ("log_max_mb", 101), ("log_backup_count", 0), ("log_backup_count", 51)])
def test_logging_bounds(field, value):
    with pytest.raises(ValidationError):
        build(**{field: value})


def test_secret_values_returns_only_real_secrets_for_redaction():
    assert build().secret_values() == []  # nothing configured
    assert build(gemini_api_key="your-gemini-api-key-here", openwa_api_key="change-me").secret_values() == []  # placeholders
    real = build(gemini_api_key="AIzaSyREAL", openwa_api_key="owa_k1_REAL", github_token="ghp_REAL")
    assert sorted(real.secret_values()) == sorted(["AIzaSyREAL", "owa_k1_REAL", "ghp_REAL"])
