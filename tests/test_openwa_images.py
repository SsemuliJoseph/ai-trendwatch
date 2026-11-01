"""
FILE: tests/test_openwa_images.py
MODULE 20 tests — run with:  pytest tests/test_openwa_images.py
OpenWAClient.send_image / DryRunClient.send_image. HTTP mocked with `responses`; no OpenWA, no WhatsApp.
"""
import base64
import json

import pytest
import responses

from app.messaging.openwa_client import (
    MAX_CAPTION_CHARS,
    MAX_IMAGE_BYTES,
    DryRunClient,
    OpenWAClient,
    OpenWADeliveryUnknownError,
    OpenWARejectedError,
    OpenWASendLimitError,
    OpenWAUnavailableError,
)

BASE = "http://127.0.0.1:2785"
SESSION = "8f3c2b1a-9d4e-4c7a-8b2f-1e6d5a4c3b2a"
GROUP = "120363000000000000@g.us"
IMAGE_URL = f"{BASE}/api/sessions/{SESSION}/messages/send-image"
TEXT_URL = f"{BASE}/api/sessions/{SESSION}/messages/send-text"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x01" * 500


def client(**kw) -> OpenWAClient:
    kw.setdefault("sleep", lambda s: None)
    return OpenWAClient(BASE, "owa_k1_SECRETKEY", SESSION, **kw)


@responses.activate
def test_image_is_sent_as_base64_with_the_text_as_caption():
    responses.add(responses.POST, IMAGE_URL, json={"messageId": "true_1@g.us_AB"}, status=201)
    result = client().send_image(GROUP, PNG, "image/png", "caption text")
    body = json.loads(responses.calls[0].request.body)
    assert body == {"chatId": GROUP, "base64": base64.b64encode(PNG).decode(), "mimetype": "image/png", "caption": "caption text"}
    assert responses.calls[0].request.headers["X-API-Key"] == "owa_k1_SECRETKEY"
    assert result.message_id == "true_1@g.us_AB" and not result.dry_run


@pytest.mark.parametrize(
    "chat, image, mimetype, caption, match",
    [
        ("628123456789@c.us", PNG, "image/png", "hi", "GROUP"),  # individual chats stay blocked
        (GROUP, PNG, "image/png", "   ", "empty"),
        (GROUP, PNG, "image/gif", "hi", "unsupported image type"),
        (GROUP, PNG, "image/svg+xml", "hi", "unsupported image type"),
        (GROUP, b"", "image/png", "hi", "empty image"),
        (GROUP, PNG, "image/png", "c" * (MAX_CAPTION_CHARS + 1), "caption"),
        (GROUP, b"\x00" * (MAX_IMAGE_BYTES + 1), "image/png", "hi", "limit"),
    ],
)
@responses.activate
def test_invalid_image_sends_are_refused_before_any_request(chat, image, mimetype, caption, match):
    with pytest.raises(ValueError, match=match):
        client().send_image(chat, image, mimetype, caption)
    assert len(responses.calls) == 0


@responses.activate
def test_text_and_image_share_the_per_run_cap():
    responses.add(responses.POST, IMAGE_URL, json={"id": "1"}, status=201)
    responses.add(responses.POST, TEXT_URL, json={"id": "2"}, status=201)
    gateway = client(max_sends=2)
    gateway.send_image(GROUP, PNG, "image/png", "one")
    gateway.send_text(GROUP, "two")
    with pytest.raises(OpenWASendLimitError):
        gateway.send_image(GROUP, PNG, "image/png", "three")
    assert len(responses.calls) == 2


@responses.activate
def test_image_sends_are_paced_like_text_sends():
    responses.add(responses.POST, IMAGE_URL, json={"id": "1"}, status=201)
    slept, now = [], [100.0]
    gateway = client(min_send_interval=8.0, delay_jitter=0.0, clock=lambda: now[0], sleep=slept.append)
    gateway.send_image(GROUP, PNG, "image/png", "one")
    now[0] += 3.0
    gateway.send_image(GROUP, PNG, "image/png", "two")
    assert slept == [pytest.approx(5.0)]


@responses.activate
def test_a_refused_image_is_a_rejection_and_a_read_timeout_is_unknown():
    responses.add(responses.POST, IMAGE_URL, json={"message": "bad media"}, status=400)
    responses.add(responses.GET, f"{BASE}/api/sessions/{SESSION}", json={"status": "ready"}, status=200)
    with pytest.raises(OpenWARejectedError):
        client().send_image(GROUP, PNG, "image/png", "hi")

    import requests

    responses.replace(responses.POST, IMAGE_URL, body=requests.ReadTimeout("slow"))
    with pytest.raises(OpenWADeliveryUnknownError):  # never re-sent blindly: that is how duplicates happen
        client().send_image(GROUP, PNG, "image/png", "hi")


@responses.activate
def test_unreachable_gateway_is_unavailable_not_unknown():
    import requests

    responses.add(responses.POST, IMAGE_URL, body=requests.ConnectionError("refused"))
    with pytest.raises(OpenWAUnavailableError):
        client().send_image(GROUP, PNG, "image/png", "hi")


def test_dry_run_records_caption_and_image_and_makes_no_request():
    gateway = DryRunClient(max_sends=2)
    result = gateway.send_image(GROUP, PNG, "image/png", "the caption")
    assert result.dry_run and gateway.sent == [(GROUP, "the caption")]
    assert gateway.images == [(GROUP, "image/png", len(PNG))]
    gateway.send_text(GROUP, "plain")
    with pytest.raises(OpenWASendLimitError):
        gateway.send_image(GROUP, PNG, "image/png", "over the cap")


def test_dry_run_applies_the_same_validation():
    gateway = DryRunClient()
    with pytest.raises(ValueError):
        gateway.send_image("628123456789@c.us", PNG, "image/png", "hi")
    with pytest.raises(ValueError):
        gateway.send_image(GROUP, PNG, "image/gif", "hi")
