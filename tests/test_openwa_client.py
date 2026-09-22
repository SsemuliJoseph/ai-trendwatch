"""
FILE: tests/test_openwa_client.py
MODULE 12 tests — run with:  pytest tests/test_openwa_client.py
HTTP is mocked with `responses`; there is no OpenWA, no network, no WhatsApp.
"""
import json
from urllib.parse import urlsplit

import pytest
import requests
import responses

from app.config.settings import Settings
from app.messaging.openwa_client import (
    MAX_TEXT_CHARS,
    DryRunClient,
    OpenWAAuthError,
    OpenWAClient,
    OpenWADeliveryUnknownError,
    OpenWAError,
    OpenWAFatalError,
    OpenWARateLimitedError,
    OpenWARejectedError,
    OpenWASendLimitError,
    OpenWASessionError,
    OpenWAUnavailableError,
    SendResult,
    build_gateway,
    validate_outbound,
)

BASE = "http://127.0.0.1:2785"
SESSION = "8f3c2b1a-9d4e-4c7a-8b2f-1e6d5a4c3b2a"
KEY = "owa_k1_SECRETKEY"
GROUP = "120363000000000000@g.us"
SEND_URL = f"{BASE}/api/sessions/{SESSION}/messages/send-text"
SESSION_URL = f"{BASE}/api/sessions/{SESSION}"


def client(**kw) -> OpenWAClient:
    kw.setdefault("sleep", lambda s: None)
    return OpenWAClient(BASE, KEY, SESSION, **kw)


# =====================================================================
# Sending: request shape and results
# =====================================================================
@responses.activate
def test_send_posts_the_documented_request_and_returns_the_gateway_id():
    responses.add(responses.POST, SEND_URL, json={"messageId": "true_120363@g.us_ABC"}, status=201)
    result = client().send_text(GROUP, "Hello group")

    assert result == SendResult(message_id="true_120363@g.us_ABC", dry_run=False)
    request = responses.calls[0].request
    assert json.loads(request.body) == {"chatId": GROUP, "text": "Hello group"}
    assert request.headers["X-API-Key"] == KEY
    assert "AI-TrendWatch" in request.headers["User-Agent"]


@responses.activate
@pytest.mark.parametrize(
    "reply, expected",
    [
        (dict(json={"id": 42}, status=200), "42"),  # alternative field name, 200 instead of 201
        (dict(json={"unrelated": True}, status=201), None),
        (dict(body="created", status=201), None),  # not JSON at all: still a success
        (dict(json={"id": True}, status=201), None),  # booleans are not ids
    ],
)
def test_any_2xx_is_success_and_the_id_is_best_effort(reply, expected):
    responses.add(responses.POST, SEND_URL, **reply)
    assert client().send_text(GROUP, "hi").message_id == expected


@responses.activate
def test_session_id_is_url_quoted_so_it_cannot_alter_the_path():
    sneaky = OpenWAClient(BASE, KEY, "a/../b", sleep=lambda s: None)
    responses.add(responses.POST, f"{BASE}/api/sessions/a%2F..%2Fb/messages/send-text", json={}, status=201)
    sneaky.send_text(GROUP, "hi")
    assert urlsplit(responses.calls[0].request.url).path == "/api/sessions/a%2F..%2Fb/messages/send-text"


# =====================================================================
# Safety rails
# =====================================================================
@responses.activate
@pytest.mark.parametrize(
    "chat_id, text",
    [
        ("256700000000@c.us", "hi"),  # individual chat
        ("256700000000", "hi"),
        ("@g.us", "hi"),  # suffix only
        (GROUP, ""),
        (GROUP, "   \n "),
        (GROUP, "x" * (MAX_TEXT_CHARS + 1)),
    ],
)
def test_invalid_messages_are_refused_before_any_http_call(chat_id, text):
    with pytest.raises(ValueError):
        client().send_text(chat_id, text)
    assert len(responses.calls) == 0


def test_validation_accepts_a_normal_group_message():
    validate_outbound(GROUP, "x" * MAX_TEXT_CHARS)


@responses.activate
def test_hard_send_cap_counts_attempts_including_failed_ones():
    responses.add(responses.POST, SEND_URL, json={}, status=201)
    responses.add(responses.POST, SEND_URL, status=503)
    c = client(max_sends=2)
    c.send_text(GROUP, "one")
    with pytest.raises(OpenWAUnavailableError):
        c.send_text(GROUP, "two")  # failed, but the attempt is spent
    with pytest.raises(OpenWASendLimitError):
        c.send_text(GROUP, "three")
    assert len(responses.calls) == 2  # the third never reached the network


class FakeTime:
    def __init__(self):
        self.now, self.sleeps = 100.0, []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(round(seconds, 3))
        self.now += seconds


@responses.activate
def test_sends_are_spaced_by_the_minimum_interval_plus_jitter():
    responses.add(responses.POST, SEND_URL, json={}, status=201)
    t = FakeTime()
    c = client(min_send_interval=8, delay_jitter=0.25, rng=lambda: 0.5, clock=t.clock, sleep=t.sleep)  # 8 * 1.125 = 9
    c.send_text(GROUP, "one")
    assert t.sleeps == []  # first message goes out immediately
    c.send_text(GROUP, "two")
    assert t.sleeps == [9.0]
    t.now += 20  # plenty of time passes on its own
    c.send_text(GROUP, "three")
    assert t.sleeps == [9.0]  # no extra wait needed


@responses.activate
def test_no_spacing_configured_means_no_sleep():
    responses.add(responses.POST, SEND_URL, json={}, status=201)
    t = FakeTime()
    c = client(clock=t.clock, sleep=t.sleep)
    c.send_text(GROUP, "one")
    c.send_text(GROUP, "two")
    assert t.sleeps == []


# =====================================================================
# Error classification
# =====================================================================
@responses.activate
@pytest.mark.parametrize(
    "status, expected, fatal",
    [
        (401, OpenWAAuthError, True),
        (403, OpenWAAuthError, True),
        (404, OpenWASessionError, True),
        (409, OpenWASessionError, True),
        (429, OpenWARateLimitedError, True),
        (500, OpenWAUnavailableError, False),
        (502, OpenWAUnavailableError, False),
        (503, OpenWAUnavailableError, False),
        (422, OpenWARejectedError, False),
    ],
)
def test_http_status_maps_to_the_right_error_and_fatality(status, expected, fatal):
    responses.add(responses.POST, SEND_URL, body="details from gateway", status=status)
    with pytest.raises(expected) as info:
        client().send_text(GROUP, "hi")
    assert isinstance(info.value, OpenWAFatalError) is fatal
    assert info.value.status_code == status


@responses.activate
def test_rate_limit_error_carries_retry_after():
    responses.add(responses.POST, SEND_URL, status=429, headers={"Retry-After": "12"})
    with pytest.raises(OpenWARateLimitedError) as info:
        client().send_text(GROUP, "hi")
    assert info.value.retry_after == 12.0


@responses.activate
def test_rate_limit_without_a_usable_header_has_no_retry_after():
    responses.add(responses.POST, SEND_URL, status=429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    with pytest.raises(OpenWARateLimitedError) as info:
        client().send_text(GROUP, "hi")
    assert info.value.retry_after is None


@responses.activate
def test_a_400_is_a_per_message_rejection_when_the_session_is_healthy():
    responses.add(responses.POST, SEND_URL, body="bad text", status=400)
    responses.add(responses.GET, SESSION_URL, json={"status": "ready"})
    with pytest.raises(OpenWARejectedError) as info:
        client().send_text(GROUP, "hi")
    assert not isinstance(info.value, OpenWAFatalError)


@responses.activate
def test_a_400_from_an_unstarted_session_becomes_a_fatal_session_error():
    responses.add(responses.POST, SEND_URL, body="session not started", status=400)
    responses.add(responses.GET, SESSION_URL, json={"status": "disconnected"})
    with pytest.raises(OpenWASessionError, match="disconnected"):
        client().send_text(GROUP, "hi")


@responses.activate
def test_connect_failures_are_unavailable_and_safe_to_retry_later():
    responses.add(responses.POST, SEND_URL, body=requests.exceptions.ConnectionError("refused"))
    with pytest.raises(OpenWAUnavailableError):
        client().send_text(GROUP, "hi")


@responses.activate
def test_read_timeout_on_send_is_delivery_unknown_and_is_never_retried():
    responses.add(responses.POST, SEND_URL, body=requests.exceptions.ReadTimeout("slow"))
    with pytest.raises(OpenWADeliveryUnknownError) as info:
        client().send_text(GROUP, "hi")
    assert len(responses.calls) == 1  # exactly one attempt: no hidden retry => no duplicate post
    assert not isinstance(info.value, (OpenWAFatalError, OpenWAUnavailableError))


@responses.activate
def test_read_timeout_on_a_read_only_call_is_just_unavailable():
    responses.add(responses.GET, SESSION_URL, body=requests.exceptions.ReadTimeout("slow"))
    with pytest.raises(OpenWAUnavailableError):
        client().session_status()


@responses.activate
@pytest.mark.parametrize(
    "reply",
    [
        dict(status=401, body="nope"),
        dict(status=429, body="slow down"),
        dict(status=500, body="boom"),
        dict(body=requests.exceptions.ConnectionError("refused")),
        dict(body=requests.exceptions.ReadTimeout("slow")),
    ],
)
def test_the_api_key_never_appears_in_any_error_message(reply):
    responses.add(responses.POST, SEND_URL, **reply)
    with pytest.raises(OpenWAError) as info:
        client().send_text(GROUP, "hi")
    assert KEY not in str(info.value) and "SECRETKEY" not in repr(info.value)


# =====================================================================
# Session readiness & health
# =====================================================================
@responses.activate
def test_ensure_ready_passes_only_for_a_ready_session():
    responses.add(responses.GET, SESSION_URL, json={"status": "READY", "engineLoaded": True})
    client().ensure_ready()  # case-insensitive


@responses.activate
@pytest.mark.parametrize("status", ["qr_ready", "initializing", "disconnected", "failed", "authenticating"])
def test_ensure_ready_rejects_every_other_status_and_says_which(status):
    responses.add(responses.GET, SESSION_URL, json={"status": status})
    with pytest.raises(OpenWASessionError, match=status):
        client().ensure_ready()


@responses.activate
def test_a_ready_status_without_a_live_engine_is_not_ready():
    responses.add(responses.GET, SESSION_URL, json={"status": "ready", "engineLoaded": False})
    with pytest.raises(OpenWASessionError, match="engine"):
        client().ensure_ready()


@responses.activate
@pytest.mark.parametrize("payload", [{"name": "trendwatch"}, {"status": ""}, ["not", "an", "object"], {"status": 5}])
def test_unexpected_session_payloads_are_reported_not_guessed(payload):
    responses.add(responses.GET, SESSION_URL, json=payload)
    with pytest.raises(OpenWAError, match="status"):
        client().ensure_ready()


@responses.activate
def test_non_json_session_reply_is_an_error():
    responses.add(responses.GET, SESSION_URL, body="<html>proxy error</html>", status=200)
    with pytest.raises(OpenWAError, match="non-JSON"):
        client().session_status()


@responses.activate
def test_health_check_is_public_and_tolerates_failures():
    responses.add(responses.GET, f"{BASE}/api/health/ready", status=200)
    assert client().is_healthy() is True
    assert "X-API-Key" not in responses.calls[0].request.headers  # least disclosure: no key on a public route

    responses.replace(responses.GET, f"{BASE}/api/health/ready", status=503)
    assert client().is_healthy() is False
    responses.replace(responses.GET, f"{BASE}/api/health/ready", body=requests.exceptions.ConnectionError())
    assert client().is_healthy() is False


# =====================================================================
# Construction
# =====================================================================
@pytest.mark.parametrize(
    "args",
    [("127.0.0.1:2785", KEY, SESSION), (BASE, "  ", SESSION), (BASE, KEY, ""), (BASE, KEY, "   ")],
)
def test_constructor_fails_fast_on_missing_configuration(args):
    with pytest.raises(ValueError):
        OpenWAClient(*args)


@responses.activate
def test_trailing_slash_in_base_url_is_harmless():
    responses.add(responses.POST, SEND_URL, json={}, status=201)
    OpenWAClient(BASE + "/", KEY, SESSION).send_text(GROUP, "hi")


# =====================================================================
# Dry run + factory
# =====================================================================
def test_dry_run_records_messages_and_never_touches_the_network():
    dry = DryRunClient()
    dry.ensure_ready()
    result = dry.send_text(GROUP, "would be sent")
    assert dry.sent == [(GROUP, "would be sent")]
    assert result.dry_run is True and result.message_id == "dry-run-1"


def test_dry_run_applies_the_same_rules_and_cap_as_the_real_client():
    dry = DryRunClient(max_sends=1)
    with pytest.raises(ValueError):
        dry.send_text("256700000000@c.us", "hi")
    with pytest.raises(ValueError):
        dry.send_text(GROUP, " ")
    dry.send_text(GROUP, "one")
    with pytest.raises(OpenWASendLimitError):
        dry.send_text(GROUP, "two")
    assert len(dry.sent) == 1


@pytest.fixture
def clean_env(monkeypatch):
    for field in Settings.model_fields:
        monkeypatch.delenv(field.upper(), raising=False)


def test_factory_defaults_to_the_dry_run_client(clean_env):
    gateway = build_gateway(Settings(_env_file=None))
    assert isinstance(gateway, DryRunClient) and gateway._max_sends == 5


def test_factory_builds_a_real_client_from_a_complete_live_configuration(clean_env):
    settings = Settings(
        _env_file=None, whatsapp_dry_run=False, whatsapp_group_id=GROUP, openwa_api_key=KEY,
        openwa_session_id=SESSION, message_delay_seconds=10, max_messages_per_run=3,
    )
    gateway = build_gateway(settings)
    assert isinstance(gateway, OpenWAClient)
    assert (gateway._interval, gateway._max_sends, gateway._base) == (10, 3, BASE)
    assert gateway._api_key == KEY  # the secret was unwrapped exactly once, here
