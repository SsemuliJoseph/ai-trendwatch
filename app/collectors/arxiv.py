"""
FILE: app/collectors/arxiv.py
MODULE 5 — arXiv collector (latest papers from chosen categories).

ARCHITECTURE NOTES
  * Same layer rules and shape as rss.py: imports only `app.models` (+ shared
    public names from the sibling rss module), limits/timeouts are constructor
    arguments, `parse_arxiv()` is PURE, `ArxivCollector` holds the network I/O,
    failures are recorded in `.failures` instead of raised.
  * DUCK-TYPED CONTRACT shared with RSSCollector:  `collect() -> list[TrendItem]`
    and `failures: dict[str, str]`. The Module 9 workflow can treat every
    collector identically without a base class (add a Protocol later if needed).
  * Why not reuse rss.parse_feed? arXiv needs source-specific mapping:
        - canonical URL WITHOUT the version suffix (…/abs/2609.01234v2 -> …/abs/2609.01234)
          so v1/v2/v3 of one paper share ONE id and are never re-sent;
        - several authors -> "A, B, C et al.";
        - category codes (cs.CL ...) become tags and name the source;
        - abstracts are PLAIN TEXT with LaTeX ($p<q$), so they must NOT go through
          an HTML stripper (it would eat text between '<' and '>').
    The ~5 duplicated lines (UTC conversion) are accepted until a third collector
    justifies extracting collectors/_common.py ("rule of three").
  * arXiv API etiquette: at most 1 request / 3 s and one connection. One combined
    query (cat:A OR cat:B ...) means ONE request per collect(), well inside that limit.
  * arXiv reports query errors as an Atom feed with HTTP 200 whose single entry has
    an id under /api/errors -> detected and raised as FeedParseError.

USAGE
    items = ArxivCollector(["cs.AI", "cs.CL"], max_items=20).collect()

LIVE CHECK (hits the real API):   python -m app.collectors.arxiv
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

import feedparser
import requests
from pydantic import ValidationError

from app.collectors.rss import USER_AGENT, FeedParseError
from app.models import SourceType, TrendItem
from app.utils.retry import NO_RETRY, RetryPolicy, http_retry_after, is_transient_http_error, retry_call

logger = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
DEFAULT_CATEGORIES: tuple[str, ...] = ("cs.AI", "cs.CL", "cs.LG")
MAX_RESULTS_CAP = 100  # keeps a single response small and polite

_CATEGORY_RE = re.compile(r"^[A-Za-z][A-Za-z\-]*(\.[A-Za-z\-]+)?$")  # e.g. cs.AI, stat.ML, hep-th
_ABS_PREFIX = re.compile(r"^https?://(?:www\.)?arxiv\.org/abs/", re.IGNORECASE)
_VERSION_SUFFIX = re.compile(r"v\d+$")
_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


# ----------------------------------------------------------------------
# Deterministic helpers
# ----------------------------------------------------------------------
def build_query(categories: Sequence[str]) -> str:
    """['cs.AI','cs.CL'] -> 'cat:cs.AI OR cat:cs.CL'. Validates codes (no query injection)."""
    if not categories:
        raise ValueError("at least one arXiv category is required")
    for cat in categories:
        if not _CATEGORY_RE.match(cat):
            raise ValueError(f"invalid arXiv category code: {cat!r}")
    return " OR ".join(f"cat:{c}" for c in categories)


def _paper_id(entry_id: str) -> str | None:
    """'http://arxiv.org/abs/2609.01234v2' -> '2609.01234'; old style keeps its slash
    ('hep-th/9901001v1' -> 'hep-th/9901001'). None if this is not an abstract id."""
    entry_id = entry_id.strip()
    if not _ABS_PREFIX.match(entry_id):
        return None
    return _VERSION_SUFFIX.sub("", _ABS_PREFIX.sub("", entry_id)) or None


def _format_authors(entry: Any) -> str | None:
    names = [a["name"].strip() for a in entry.get("authors", []) if a.get("name")]
    if not names:
        return None
    return ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")


def _to_datetime(parsed: Any) -> datetime | None:
    if not parsed:
        return None
    try:
        return datetime(*parsed[:6], tzinfo=timezone.utc)  # feedparser normalises to UTC
    except (TypeError, ValueError):
        return None


def parse_arxiv(content: bytes | str, limit: int | None = None) -> list[TrendItem]:
    """PURE: arXiv Atom document -> newest-first list of TrendItem (at most `limit`).

    Raises FeedParseError for: non-feed bodies, malformed XML with no entries,
    and arXiv's in-band API error entries.
    """
    feed = feedparser.parse(content)

    # arXiv signals a bad query with HTTP 200 + an entry whose id lives under /api/errors.
    for entry in feed.entries:
        if "/api/errors" in entry.get("id", ""):
            raise FeedParseError(f"arXiv API error: {(entry.get('summary') or 'unknown').strip()}")

    if not feed.entries:
        # bozo = malformed XML; missing `version` = not an Atom/RSS document at all (e.g. HTML page).
        if feed.bozo or not feed.get("version"):
            reason = feed.get("bozo_exception") or "no Atom format recognised"
            raise FeedParseError(f"arXiv: not a valid feed ({reason})")
        return []  # valid feed, zero results

    items: list[TrendItem] = []
    skipped = 0
    for entry in feed.entries:
        paper_id = _paper_id(entry.get("id", ""))
        if paper_id is None:
            skipped += 1
            continue
        categories = [t["term"] for t in entry.get("tags", []) if t.get("term")]
        primary = (entry.get("arxiv_primary_category") or {}).get("term") or (categories[0] if categories else None)
        try:
            items.append(
                TrendItem(
                    title=entry.get("title", ""),  # newlines/double spaces collapsed by the model
                    url=f"https://arxiv.org/abs/{paper_id}",
                    source_name=f"arXiv {primary}" if primary else "arXiv",
                    source_type=SourceType.ARXIV,
                    content=entry.get("summary", ""),  # plain-text abstract: deliberately NOT html-stripped
                    author=_format_authors(entry),
                    published_at=_to_datetime(entry.get("published_parsed")),  # v1 submission date
                    tags=["arxiv", "research", *categories],
                )
            )
        except ValidationError as exc:
            skipped += 1
            logger.debug("arXiv: skipped entry %s (%s)", paper_id, exc.errors()[0]["msg"])

    if skipped:
        logger.info("arXiv: skipped %d unusable entries", skipped)

    items.sort(key=lambda i: i.published_at or _EPOCH, reverse=True)
    return items[:limit] if limit is not None else items


# ----------------------------------------------------------------------
# Collector (network I/O lives here)
# ----------------------------------------------------------------------
class ArxivCollector:
    """Fetches the most recently submitted papers in the given categories."""

    name = "arxiv"

    def __init__(
        self,
        categories: Sequence[str] = DEFAULT_CATEGORIES,
        *,
        max_items: int = 20,
        timeout_seconds: float = 15,
        session: requests.Session | None = None,  # injectable for tests
        retry: RetryPolicy = NO_RETRY,  # opt-in; arXiv rate-limits with 429, so backoff genuinely helps here
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._query = build_query(categories)  # fail fast on bad category codes
        self._max_items = max(1, min(max_items, MAX_RESULTS_CAP))
        self._timeout = timeout_seconds
        self._session = session or requests.Session()
        self._retry, self._sleep = retry, sleep
        self.failures: dict[str, str] = {}  # {'arxiv': error}, reset on every collect()

    def _fetch(self) -> bytes:
        def attempt() -> bytes:
            response = self._session.get(
                API_URL,
                params={
                    "search_query": self._query,
                    "sortBy": "submittedDate",  # newest submissions first
                    "sortOrder": "descending",
                    "start": 0,
                    "max_results": self._max_items,
                },
                timeout=self._timeout,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()  # includes 429 when rate-limited
            return response.content

        return retry_call(
            attempt,
            self._retry,
            is_retryable=is_transient_http_error,
            retry_after=http_retry_after,
            on_retry=lambda n, exc, delay: logger.info("retrying arXiv after attempt %d (%s) in %.1fs", n, exc, delay),
            sleep=self._sleep,
        )

    def collect(self) -> list[TrendItem]:
        self.failures = {}
        try:
            items = parse_arxiv(self._fetch(), limit=self._max_items)
        except (requests.RequestException, FeedParseError) as exc:
            self.failures[self.name] = f"{type(exc).__name__}: {exc}"
            logger.warning("arXiv collection failed: %s", exc)
            return []
        logger.info("arXiv: %d items", len(items))
        return items


if __name__ == "__main__":
    # Manual live check. No app.config import on purpose (layer rule) -> small fixed limit.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    collector = ArxivCollector(max_items=5)
    results = collector.collect()
    print(f"\nCollected {len(results)} papers\n")
    for item in results:
        when = item.published_at.strftime("%Y-%m-%d %H:%M") if item.published_at else "no date"
        print(f"[{item.source_name}] {when} | {item.title[:70]}\n    {item.url} | {item.author}")
    for name, err in collector.failures.items():
        print(f"FAILED  {name}: {err}")
