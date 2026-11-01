"""
FILE: app/media/fetcher.py
Find the preview image an article publishes about itself (<meta property="og:image">) and download it.

WHY THE ARTICLE'S OWN IMAGE
  It is the picture the publisher chose to represent the story, the same one WhatsApp or any social
  site would show in a link card, and it is attributed by the link in the message. Nothing is invented.

SAFETY (feed URLs are attacker-influenced, so fetching them is an SSRF risk)
  * http/https only, no credentials in the URL, a bounded number of redirects.
  * Every hop (page, redirect, image) must resolve ONLY to public addresses: loopback, private,
    link-local and cloud-metadata ranges are refused. Residual risk, stated plainly: the name is
    resolved once for the check and again by `requests`, so a hostile DNS server could in theory
    answer differently the second time (rebinding). For a local, low-volume class project that
    reads at most 5 pages per run this is acceptable; a hosted service would need pinned connections.
  * Hard byte caps (page head, image), a short timeout, and the image must be a real JPEG/PNG/WebP:
    the BYTES are checked, the Content-Type header is not trusted.
  * An honest User-Agent. Sites that block bots are not worked around: no image -> the caller falls
    back to a generated card.
  * Never raises: an image is decoration, and a failure here must not stop a delivery.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from collections.abc import Callable
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import requests

from app.media import Image
from app.models import SourceType, TrendItem

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; AI-TrendWatch/1.0; academic class project; link-preview fetch)"
MAX_HTML_BYTES = 400_000  # the <head> is near the top; never download a whole 5 MB page
MAX_IMAGE_BYTES = 5_000_000
MIN_IMAGE_BYTES = 2_000  # smaller than this is a tracking pixel or icon, not a picture
MAX_REDIRECTS = 3
_REDIRECT_CODES = (301, 302, 303, 307, 308)

# Candidate meta tags, best first. Secure/explicit variants before the generic ones.
_IMAGE_KEYS = ("og:image:secure_url", "og:image:url", "og:image", "twitter:image", "twitter:image:src")


class _HeadParser(HTMLParser):
    """Collects <meta> image URLs and stops at </head>: nothing after it is needed."""

    class Done(Exception):
        pass

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "body":
            raise self.Done
        if tag != "meta":
            return
        data = {k.lower(): (v or "").strip() for k, v in attrs}
        key = (data.get("property") or data.get("name") or "").lower()
        if key in _IMAGE_KEYS and data.get("content"):
            self.found.setdefault(key, data["content"])

    def handle_endtag(self, tag: str) -> None:
        if tag == "head":
            raise self.Done


def extract_image_url(html: str, base_url: str) -> str | None:
    """Best preview-image URL declared in the page head, made absolute; None if there is none."""
    parser = _HeadParser()
    try:
        parser.feed(html)
    except _HeadParser.Done:
        pass
    for key in _IMAGE_KEYS:
        if key in parser.found:
            return urljoin(base_url, parser.found[key])
    return None


def sniff_image_type(data: bytes) -> str | None:
    """Real type from the leading bytes; None for anything WhatsApp image sends should not carry (GIF, SVG, HTML...)."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


class ArticleImageFetcher:
    def __init__(
        self,
        *,
        timeout: float = 6.0,
        session: requests.Session | None = None,  # injectable for tests
        resolver: Callable[..., list] = socket.getaddrinfo,  # injectable for tests
    ) -> None:
        self._timeout = timeout
        self._http = session or requests.Session()
        self._resolve = resolver

    # ------------------------------------------------------------------
    def fetch(self, item: TrendItem) -> Image | None:
        """The article's preview image, or None. Never raises."""
        if item.source_type is SourceType.ARXIV:
            return None  # arXiv abstract pages publish no preview image: skip the pointless request
        try:
            page = self._get(item.url, MAX_HTML_BYTES, truncate_ok=True)
            if page is None:
                return None
            body, final_url = page
            image_url = extract_image_url(body.decode("utf-8", errors="replace"), final_url)
            if not image_url:
                return None
            fetched = self._get(image_url, MAX_IMAGE_BYTES, truncate_ok=False)
            if fetched is None:
                return None
            data, _ = fetched
            mimetype = sniff_image_type(data)
            if mimetype is None or len(data) < MIN_IMAGE_BYTES:
                return None
            return Image(data=data, mimetype=mimetype, origin="article")
        except Exception as exc:  # decoration must never break delivery
            logger.info("no article image for %s (%s: %s)", item.url, type(exc).__name__, exc)
            return None

    # ------------------------------------------------------------------
    def _is_public(self, url: str) -> bool:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            return False
        try:
            infos = self._resolve(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
            addresses = {ipaddress.ip_address(info[4][0]) for info in infos}
        except (OSError, ValueError):  # unresolvable, or an address form we cannot vet (e.g. scoped IPv6)
            return False
        return bool(addresses) and all(address.is_global for address in addresses)

    def _get(self, url: str, max_bytes: int, *, truncate_ok: bool) -> tuple[bytes, str] | None:
        """GET with manual, re-validated redirects and a byte cap. -> (body, final_url) or None."""
        for _ in range(MAX_REDIRECTS + 1):
            if not self._is_public(url):
                logger.info("image fetch refused (not a public http(s) address): %s", url)
                return None
            response = self._http.get(
                url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,image/*;q=0.8"},
                timeout=self._timeout, stream=True, allow_redirects=False,
            )
            try:
                if response.status_code in _REDIRECT_CODES:
                    location = response.headers.get("Location")
                    if not location:
                        return None
                    url = urljoin(url, location)
                    continue
                if response.status_code != 200:
                    return None
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=16_384):
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > max_bytes:
                        if truncate_ok:
                            break
                        return None  # an oversized image is rejected, never truncated
                return b"".join(chunks)[: max_bytes if truncate_ok else None], url
            finally:
                response.close()
        return None  # too many redirects
