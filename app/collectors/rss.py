"""
FILE: app/collectors/rss.py
MODULE 4 — RSS/Atom collector.

ARCHITECTURE NOTES
  * Layer rule: imports ONLY `app.models` (+ third-party). It knows nothing of
    Gemini, OpenWA, SQLite or `app.config`. Limits/timeouts are therefore
    constructor ARGUMENTS; the orchestrator (agents/) reads Settings and injects them.
  * FETCH vs PARSE split:
        _fetch()     -> the only network I/O (requests, with timeout + User-Agent)
        parse_feed() -> PURE function: bytes in, list[TrendItem] out.
    Parsing is deterministic and unit-tested with fixture XML, no network needed.
    (feedparser is used only as a parser; we do our own HTTP so we control timeouts.)
  * ISOLATED FAILURES: one dead/malformed feed is logged and recorded in
    `collector.failures`; the remaining feeds still run. A collector must never
    take the whole pipeline down.
  * RETRIES ARE OPT-IN (`retry=RetryPolicy(...)`, default none): a GET is idempotent, so transient failures
    (connection errors, timeouts, 408/425/429/5xx) are retried with exponential backoff and Retry-After.
    Permanent errors (404, 403, ...) fail at once.
  * Deterministic code only — nothing in this file is an "LLM/agent".

USAGE
    from app.collectors.rss import RSSCollector
    items = RSSCollector(max_items_per_source=20, timeout_seconds=15).collect()

LIVE CHECK (hits the real feeds):   python -m app.collectors.rss
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

import feedparser
import requests
from pydantic import ValidationError

from app.models import SourceType, TrendItem
from app.utils.retry import NO_RETRY, RetryPolicy, http_retry_after, is_transient_http_error, retry_call

logger = logging.getLogger(__name__)

USER_AGENT = "AI-TrendWatch/0.1 (academic project; RSS reader)"
_ACCEPT = "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8"
_WHITESPACE = re.compile(r"\s+")
_EPOCH = datetime.min.replace(tzinfo=timezone.utc)  # sort key for undated items


class FeedParseError(ValueError):
    """The response was not a usable RSS/Atom document (e.g. an HTML error page)."""


@dataclass(frozen=True)
class FeedSource:
    """One RSS/Atom feed to poll. `tags` are attached to every item from it."""

    name: str
    url: str
    tags: tuple[str, ...] = ()


# Free, public feeds. Publishers move/retire feed URLs over time: run
# `python -m app.collectors.rss` to see which ones are alive; dead ones only
# produce a logged warning and never break the run.
DEFAULT_FEEDS: tuple[FeedSource, ...] = (
    FeedSource("Hacker News", "https://news.ycombinator.com/rss", ("hackernews",)),
    FeedSource("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", ("ai",)),
    FeedSource("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", ("ai",)),
    FeedSource("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed", ("ai",)),
    FeedSource("Google AI Blog", "https://blog.google/technology/ai/rss/", ("ai", "google")),
    FeedSource("OpenAI News", "https://openai.com/news/rss.xml", ("ai", "openai")),
    FeedSource("Hugging Face Blog", "https://huggingface.co/blog/feed.xml", ("ai", "huggingface")),
    FeedSource("Ars Technica", "https://feeds.arstechnica.com/arstechnica/technology-lab", ("tech",)),
)


# ----------------------------------------------------------------------
# Deterministic helpers
# ----------------------------------------------------------------------
class _TextExtractor(HTMLParser):
    """Collects visible text; drops <script>/<style> bodies; block tags become spaces."""

    _SKIP = {"script", "style"}
    _BLOCK = {"p", "br", "div", "li", "ul", "ol", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)  # &amp; -> &, &#39; -> ', ...
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def strip_html(text: str | None) -> str:
    """HTML fragment -> single-line plain text. Safe on plain text and on None."""
    if not text:
        return ""
    extractor = _TextExtractor()
    extractor.feed(text)
    extractor.close()
    return _WHITESPACE.sub(" ", "".join(extractor.parts)).strip()


def _to_datetime(parsed: Any) -> datetime | None:
    """feedparser gives a UTC-normalised time.struct_time (or None) -> aware datetime."""
    if not parsed:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def parse_feed(content: bytes | str, source: FeedSource, limit: int | None = None) -> list[TrendItem]:
    """PURE: raw feed document -> newest-first list of TrendItem (at most `limit`).

    Raises FeedParseError if the document has no entries AND is not a recognisable feed
    (HTML error page, plain text, empty body).
    Individual bad entries (no link, no title, invalid URL) are skipped, not fatal.
    """
    feed = feedparser.parse(content)

    if not feed.entries:
        # Two signals are needed: `bozo` (malformed/truncated XML) is NOT raised by
        # feedparser for an HTML error page served with HTTP 200, while `version`
        # ('rss20', 'atom10', ...) is only set when a real feed format was recognised.
        if feed.bozo or not feed.get("version"):
            reason = feed.get("bozo_exception") or "no RSS/Atom format recognised"
            raise FeedParseError(f"{source.name}: not a valid feed ({reason})")
        return []  # well-formed, recognised feed that simply has no entries

    items: list[TrendItem] = []
    skipped = 0
    for entry in feed.entries:
        summary = entry.get("summary") or ""
        if not summary and entry.get("content"):
            summary = entry["content"][0].get("value", "")
        entry_tags = [t.get("term") for t in entry.get("tags", []) if t.get("term")]
        try:
            items.append(
                TrendItem(
                    title=strip_html(entry.get("title")),
                    url=entry.get("link") or "",
                    source_name=source.name,
                    source_type=SourceType.RSS,
                    content=strip_html(summary),
                    author=(entry.get("author") or None),
                    published_at=_to_datetime(entry.get("published_parsed") or entry.get("updated_parsed")),
                    tags=[*source.tags, *entry_tags],  # source tags first so the model's tag cap keeps them
                )
            )
        except ValidationError as exc:
            skipped += 1
            logger.debug("%s: skipped entry (%s)", source.name, exc.errors()[0]["msg"])

    if skipped:
        logger.info("%s: skipped %d unusable entries", source.name, skipped)

    # Feed order is usually newest-first but not guaranteed. Sort explicitly;
    # Python's sort is stable, so undated items keep their original relative order.
    items.sort(key=lambda i: i.published_at or _EPOCH, reverse=True)
    return items[:limit] if limit is not None else items


# ----------------------------------------------------------------------
# Collector (network I/O lives here)
# ----------------------------------------------------------------------
class RSSCollector:
    """Polls a list of feeds sequentially and returns TrendItems."""

    def __init__(
        self,
        feeds: Sequence[FeedSource] = DEFAULT_FEEDS,
        *,
        max_items_per_source: int = 20,
        timeout_seconds: float = 15,
        session: requests.Session | None = None,  # injectable for tests
        retry: RetryPolicy = NO_RETRY,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._feeds = tuple(feeds)
        self._max_items = max_items_per_source
        self._timeout = timeout_seconds
        self._session = session or requests.Session()
        self._retry, self._sleep = retry, sleep
        self.failures: dict[str, str] = {}  # source name -> error, reset on every collect()

    def _fetch(self, url: str) -> bytes:
        def attempt() -> bytes:
            response = self._session.get(
                url,
                timeout=self._timeout,  # ALWAYS set: requests waits forever by default
                headers={"User-Agent": USER_AGENT, "Accept": _ACCEPT},
            )
            response.raise_for_status()  # 4xx/5xx -> requests.HTTPError
            return response.content

        return retry_call(
            attempt,
            self._retry,
            is_retryable=is_transient_http_error,
            retry_after=http_retry_after,
            on_retry=lambda n, exc, delay: logger.info("retrying %s after attempt %d (%s) in %.1fs", url, n, exc, delay),
            sleep=self._sleep,
        )

    def collect(self) -> list[TrendItem]:
        self.failures = {}
        collected: list[TrendItem] = []
        for feed in self._feeds:
            try:
                batch = parse_feed(self._fetch(feed.url), feed, limit=self._max_items)
            except (requests.RequestException, FeedParseError) as exc:
                self.failures[feed.name] = f"{type(exc).__name__}: {exc}"
                logger.warning("RSS source %r failed: %s", feed.name, exc)
                continue
            logger.info("RSS source %r: %d items", feed.name, len(batch))
            collected.extend(batch)
        return collected


if __name__ == "__main__":
    # Manual live check. No app.config import on purpose (layer rule) -> small fixed limit.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    collector = RSSCollector(max_items_per_source=3)
    results = collector.collect()
    print(f"\nCollected {len(results)} items from {len(DEFAULT_FEEDS) - len(collector.failures)}"
          f"/{len(DEFAULT_FEEDS)} feeds\n")
    for item in results[:12]:
        when = item.published_at.strftime("%Y-%m-%d %H:%M") if item.published_at else "no date"
        print(f"[{item.source_name}] {when} | {item.title[:70]}")
    for name, err in collector.failures.items():
        print(f"FAILED  {name}: {err}")
