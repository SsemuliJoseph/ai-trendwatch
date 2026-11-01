"""
FILE: app/messaging/openwa_client.py
MODULE 12 — OpenWA messaging adapter (the ONLY code that talks to the gateway).

ARCHITECTURE NOTES
  * ISOLATED HTTP ADAPTER: plain REST calls via `requests`. No Node, no Puppeteer, no OpenWA
    internals leak past this file, so replacing the gateway later touches only this module.
    Layer rule: imports `app.models`-level concepts only through text/ids, plus `app.config`
    (allowed) for the factory at the bottom.
  * ONE INTERFACE, TWO IMPLEMENTATIONS (`Gateway` Protocol):
        OpenWAClient  real HTTP
        DryRunClient  no network at all; same validation, same limits, records what it WOULD send
    `build_gateway(settings)` picks one from WHATSAPP_DRY_RUN, so the safe default lives in a
    single place and the rest of the pipeline cannot tell the difference.
  * SAFETY RAILS enforced HERE (defence in depth: even a buggy caller cannot bypass them):
      - recipient must be a GROUP id ending in '@g.us' (no DMs, no cold messaging);
      - refuses empty or oversized text;
      - `max_sends` hard cap per client instance (attempts count, failed ones included);
      - minimum spacing between sends with random jitter (a fixed 8.000 s beat looks robotic);
      - `ensure_ready()` checks the session is linked BEFORE anything is sent.
  * AT-MOST-ONCE SENDING: there is NO automatic retry. If the request reached the gateway but the
    reply timed out, the message may already be in the group; retrying would double-post. That
    case raises `OpenWADeliveryUnknownError` so the caller decides (Module 13/14). Failures where
    the request provably never left (connect errors) are `OpenWAUnavailableError`.
  * ERROR CLASSES tell the sender what to do next:
        OpenWAFatalError family (Auth, Session, RateLimited, SendLimit) -> STOP the run
        everything else                                                  -> this item failed; decide per item
  * A 2xx means "the gateway handed it to WhatsApp", not "delivered". We never claim more.
  * The API key is sent only in the X-API-Key header and never appears in errors or logs.

API FACTS (verified against OpenWA's docs):  POST /api/sessions/{id}/messages/send-text
  {"chatId": "...@g.us", "text": "..."}  ->  201;  GET /api/sessions/{id} -> status 'ready' when linked;
  409 = session not connected, 404 = unknown session, 400 = bad request OR unstarted session,
  429 = rate limited;  GET /api/health/ready is public.
"""
from __future__ import annotations

import logging
import random
import base64
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

import requests

from app.config.settings import Settings

logger = logging.getLogger(__name__)

GROUP_SUFFIX = "@g.us"
MAX_TEXT_CHARS = 4000  # far above what the formatter emits (~1.2k); a guard against runaway text
MAX_CAPTION_CHARS = 1000  # OpenWA's image-caption limit is 1024; stay a little under it
MAX_IMAGE_BYTES = 5_000_000  # same cap the image fetcher enforces; a guard against runaway uploads
IMAGE_MIMETYPES = ("image/jpeg", "image/png", "image/webp")
USER_AGENT = "AI-TrendWatch/0.1 (academic project; OpenWA client)"


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
class OpenWAError(RuntimeError):
    """Base class. Plain instances/subclasses below the fatal family are PER-ITEM failures."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OpenWAFatalError(OpenWAError):
    """Further sends in this run are pointless or unsafe. The sender must stop."""


class OpenWAAuthError(OpenWAFatalError):
    """401/403: wrong or under-privileged API key."""


class OpenWASessionError(OpenWAFatalError):
    """Session unknown, not started, disconnected or not linked (404 / 409 / status != ready)."""


class OpenWARateLimitedError(OpenWAFatalError):
    """429 from the gateway. `retry_after` (seconds) is set when the gateway said so."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class OpenWASendLimitError(OpenWAFatalError):
    """Our own per-client hard cap (`max_sends`) was reached."""


class OpenWAUnavailableError(OpenWAError):
    """Gateway unreachable, or a 5xx. For connect failures the request never left: safe to retry later."""


class OpenWARejectedError(OpenWAError):
    """The gateway rejected THIS message (e.g. 400/422 validation). Other messages may still work."""


class OpenWADeliveryUnknownError(OpenWAError):
    """The request was sent but no reply arrived (read timeout). The message MAY be in the group.
    Never retry blindly: that is how duplicates happen."""


# ----------------------------------------------------------------------
# Shared contract
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SendResult:
    message_id: str | None  # gateway's id if it returned one (shape not guaranteed)
    dry_run: bool = False


class Gateway(Protocol):
    def ensure_ready(self) -> None: ...

    def send_text(self, chat_id: str, text: str) -> SendResult: ...

    def send_image(self, chat_id: str, image: bytes, mimetype: str, caption: str) -> SendResult: ...


def validate_image(image: bytes, mimetype: str, caption: str) -> None:
    """Image-specific rules every gateway applies (ValueError = caller bug). The caption is checked for
    group/empty/length by `validate_outbound` as well; this adds the media rules."""
    if mimetype not in IMAGE_MIMETYPES:
        raise ValueError(f"unsupported image type {mimetype!r}; allowed: {', '.join(IMAGE_MIMETYPES)}")
    if not image:
        raise ValueError("refusing to send an empty image")
    if len(image) > MAX_IMAGE_BYTES:
        raise ValueError(f"image is {len(image)} bytes; the limit is {MAX_IMAGE_BYTES}")
    if len(caption) > MAX_CAPTION_CHARS:
        raise ValueError(f"caption is {len(caption)} chars; the limit is {MAX_CAPTION_CHARS}")


def validate_outbound(chat_id: str, text: str) -> None:
    """Rules EVERY gateway implementation applies (ValueError = caller bug, not a runtime failure)."""
    if not chat_id.endswith(GROUP_SUFFIX) or len(chat_id) <= len(GROUP_SUFFIX):
        raise ValueError(f"recipient must be a WhatsApp GROUP id ending in {GROUP_SUFFIX!r}: individual chats are blocked")
    if not text.strip():
        raise ValueError("refusing to send an empty message")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"message is {len(text)} chars; the limit is {MAX_TEXT_CHARS}")


# ----------------------------------------------------------------------
# Real gateway client
# ----------------------------------------------------------------------
class OpenWAClient:
    dry_run = False  # real sends: the delivery stage claims items in SQLite before sending

    def __init__(
        self,
        base_url: str,
        api_key: str,
        session_id: str,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        min_send_interval: float = 0.0,
        delay_jitter: float = 0.25,
        max_sends: int | None = None,
        session: requests.Session | None = None,  # injectable for tests
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not api_key.strip():
            raise ValueError("api_key is required")
        if not session_id.strip():
            raise ValueError("session_id is required (the UUID OpenWA returned when the session was created)")
        self._base = base_url
        self._api_key = api_key
        self._session_path = f"/api/sessions/{quote(session_id.strip(), safe='')}"  # no path injection
        self._timeout = (connect_timeout, read_timeout)
        self._interval = max(0.0, min_send_interval)
        self._jitter = max(0.0, delay_jitter)
        self._max_sends = max_sends
        self._http = session or requests.Session()
        self._clock, self._sleep, self._rng = clock, sleep, rng
        self._sends = 0
        self._last_send: float | None = None

    # ---- read-only checks ----
    def is_healthy(self) -> bool:
        """Gateway process is up (public route: deliberately sent WITHOUT the API key)."""
        try:
            response = self._http.get(f"{self._base}/api/health/ready", timeout=self._timeout,
                                      headers={"User-Agent": USER_AGENT})
        except requests.RequestException:
            return False
        return response.status_code == 200

    def session_status(self) -> str:
        """Lower-cased session status ('ready', 'qr_ready', 'disconnected', ...). Raises on API problems."""
        data = self._json(self._request("GET", self._session_path))
        status = data.get("status") if isinstance(data, dict) else None
        if not isinstance(status, str) or not status:
            raise OpenWAError("unexpected session payload: no 'status' field (did OpenWA's API change?)")
        if data.get("engineLoaded") is False:  # documented flag: no live engine behind this session
            return "engine-not-loaded"
        return status.lower()

    def ensure_ready(self) -> None:
        """Raise OpenWASessionError unless the phone is linked and the session is live."""
        status = self.session_status()
        if status != "ready":
            raise OpenWASessionError(
                f"session is '{status}', not 'ready'. Open the dashboard ({self._base}) and Start / re-link the session."
            )

    # ---- sending ----
    def send_text(self, chat_id: str, text: str) -> SendResult:
        validate_outbound(chat_id, text)
        response = self._dispatch(f"{self._session_path}/messages/send-text", {"chatId": chat_id, "text": text})
        message_id = _extract_message_id(response)
        logger.info("sent %d chars to %s (gateway id: %s)", len(text), chat_id, message_id or "n/a")
        return SendResult(message_id=message_id)

    def send_image(self, chat_id: str, image: bytes, mimetype: str, caption: str) -> SendResult:
        """ONE message: the picture with the text as its caption. Counts as one send for pacing and the run cap."""
        validate_outbound(chat_id, caption)
        validate_image(image, mimetype, caption)
        body = {"chatId": chat_id, "base64": base64.b64encode(image).decode("ascii"), "mimetype": mimetype, "caption": caption}
        response = self._dispatch(f"{self._session_path}/messages/send-image", body)
        message_id = _extract_message_id(response)
        logger.info("sent image (%d bytes) + %d-char caption to %s (gateway id: %s)",
                    len(image), len(caption), chat_id, message_id or "n/a")
        return SendResult(message_id=message_id)

    def _dispatch(self, path: str, body: dict) -> requests.Response:
        """The single guarded send path (run cap -> pacing -> request) shared by text and image sends."""
        if self._max_sends is not None and self._sends >= self._max_sends:
            raise OpenWASendLimitError(f"send limit of {self._max_sends} reached for this run")

        self._pace()
        self._sends += 1  # attempts count: a failed attempt may still have reached WhatsApp
        self._last_send = self._clock()
        try:
            return self._request("POST", path, json_body=body, is_send=True)
        except OpenWARejectedError as exc:
            if exc.status_code == 400:  # an UNSTARTED session also answers 400 -> tell the two apart
                self.ensure_ready()
            raise

    # ---- internals ----
    def _pace(self) -> None:
        if self._last_send is None or self._interval <= 0:
            return
        target = self._interval * (1 + self._jitter * self._rng())  # rng() in [0, 1)
        remaining = target - (self._clock() - self._last_send)
        if remaining > 0:
            self._sleep(remaining)

    def _json(self, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise OpenWAError("gateway returned a non-JSON response") from exc

    def _request(self, method: str, path: str, *, json_body: dict | None = None, is_send: bool = False) -> requests.Response:
        try:
            response = self._http.request(
                method, f"{self._base}{path}", json=json_body, timeout=self._timeout,
                headers={"X-API-Key": self._api_key, "User-Agent": USER_AGENT},
            )
        except requests.ReadTimeout as exc:
            if is_send:  # request was delivered, answer was not: outcome unknown
                raise OpenWADeliveryUnknownError("no reply from the gateway after sending; the message may or may not have been posted") from exc
            raise OpenWAUnavailableError("gateway did not answer in time") from exc
        except requests.RequestException as exc:  # connect errors, DNS, refused, ...: nothing was sent
            raise OpenWAUnavailableError(f"gateway unreachable ({type(exc).__name__})") from exc
        return self._check(response)

    def _check(self, response: requests.Response) -> requests.Response:
        code = response.status_code
        if 200 <= code < 300:
            return response
        detail = f"HTTP {code}: {response.text.strip()[:200]}"  # response body only; never our headers
        if code in (401, 403):
            raise OpenWAAuthError(f"API key rejected or lacks permission ({detail})", status_code=code)
        if code == 429:
            raise OpenWARateLimitedError(f"gateway rate limit hit ({detail})", retry_after=_retry_after(response))
        if code == 404:
            raise OpenWASessionError(f"session not found: check OPENWA_SESSION_ID ({detail})", status_code=code)
        if code == 409:
            raise OpenWASessionError(f"session not connected ({detail})", status_code=code)
        if code >= 500:
            raise OpenWAUnavailableError(f"gateway error ({detail})", status_code=code)
        raise OpenWARejectedError(f"request rejected ({detail})", status_code=code)


def _retry_after(response: requests.Response) -> float | None:
    try:
        return float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None


def _extract_message_id(response: requests.Response) -> str | None:
    """Best-effort: the reply shape is not part of any contract we rely on."""
    try:
        data = response.json()
    except ValueError:
        return None
    if isinstance(data, dict):
        for key in ("messageId", "id"):
            value = data.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value):
                return str(value)
    return None


# ----------------------------------------------------------------------
# Dry-run twin (default while WHATSAPP_DRY_RUN=true)
# ----------------------------------------------------------------------
class DryRunClient:
    """Same rules, same limits, ZERO network. Records what a real run would have sent."""

    dry_run = True  # rehearsal: the delivery stage neither claims nor marks items

    def __init__(self, max_sends: int | None = None) -> None:
        self._max_sends = max_sends
        self.sent: list[tuple[str, str]] = []  # (chat_id, text or image caption)
        self.images: list[tuple[str, str, int]] = []  # (chat_id, mimetype, bytes) for the sends that carried a picture

    def ensure_ready(self) -> None:
        return None  # nothing to link in a dry run

    def send_text(self, chat_id: str, text: str) -> SendResult:
        validate_outbound(chat_id, text)
        if self._max_sends is not None and len(self.sent) >= self._max_sends:
            raise OpenWASendLimitError(f"send limit of {self._max_sends} reached for this run")
        self.sent.append((chat_id, text))
        logger.info("[dry-run] would send %d chars to %s", len(text), chat_id)
        return SendResult(message_id=f"dry-run-{len(self.sent)}", dry_run=True)

    def send_image(self, chat_id: str, image: bytes, mimetype: str, caption: str) -> SendResult:
        validate_outbound(chat_id, caption)
        validate_image(image, mimetype, caption)
        if self._max_sends is not None and len(self.sent) >= self._max_sends:
            raise OpenWASendLimitError(f"send limit of {self._max_sends} reached for this run")
        self.sent.append((chat_id, caption))
        self.images.append((chat_id, mimetype, len(image)))
        logger.info("[dry-run] would send a %s image (%d bytes) + %d-char caption to %s", mimetype, len(image), len(caption), chat_id)
        return SendResult(message_id=f"dry-run-{len(self.sent)}", dry_run=True)


def build_gateway(settings: Settings) -> Gateway:
    """The single switch between real sending and dry-run (Settings guarantees a live config is complete)."""
    if settings.whatsapp_dry_run:
        return DryRunClient(max_sends=settings.max_messages_per_run)
    return OpenWAClient(
        settings.openwa_base_url,
        settings.openwa_api_key.get_secret_value(),
        settings.openwa_session_id,
        min_send_interval=settings.message_delay_seconds,
        max_sends=settings.max_messages_per_run,
    )
