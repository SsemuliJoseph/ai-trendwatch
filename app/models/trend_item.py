"""
FILE: app/models/trend_item.py
MODULE 3 — Data model: TrendItem.

ARCHITECTURE NOTES
  * LEAF layer: pure data + deterministic normalisation. No I/O, no imports from
    other `app.*` packages, so collectors / database / ai / messaging can all
    share ONE vocabulary without depending on each other.
  * NORMALISE, DON'T JUDGE. This model cleans data (trim, UTC, canonical URL,
    truncate) and rejects only *structurally* invalid input (no title, not an
    http(s) URL). Quality rules ("too old", "no date", "too short") belong to
    the validation engine (Module 7), so policy can change without touching the schema.
  * IMMUTABLE (`frozen=True`): a pipeline stage never edits an item in place; it
    returns a new one (see `with_analysis`). Prevents hidden state bugs between
    LangGraph nodes.
  * `extra="forbid"`: a collector that misspells a field fails loudly at the
    source instead of silently losing data.
  * STABLE IDENTITY: `id` = hash of the canonical URL. The same article reached
    via different tracking links / http-vs-https / www gets the SAME id, which is
    what makes SQLite deduplication (Modules 6-7) a simple primary-key check.
    `id` is a normal stored field (derived once, at creation), so items
    round-trip through JSON/SQLite unchanged.
  * LIFECYCLE in ONE model: collectors fill the "collected" fields; the AI stage
    fills `relevance_score` + `ai_summary`. Terminology: parsing/validation are
    deterministic functions; only the Gemini stage is "LLM/agent" work.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---- Size caps: bound memory, Gemini token cost and WhatsApp message length ----
TITLE_MAX = 300
CONTENT_MAX = 2000
SUMMARY_MAX = 1000
TAGS_MAX = 10

# Query params that never change which page you land on.
_TRACKING_PARAMS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid", "igshid", "ref_src"})
_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[\W_]+")  # unicode-aware: keeps letters of any script


class SourceType(str, Enum):
    """Where an item came from. `str` mixin -> serialises as plain 'rss' etc."""

    RSS = "rss"
    ARXIV = "arxiv"
    GITHUB = "github"


def canonicalize_url(raw: str) -> str:
    """Return a normalised absolute http(s) URL, or raise ValueError.

    lowercases scheme+host, drops #fragment and tracking params (utm_*, fbclid, ...),
    sorts remaining query params, removes trailing '/' from the path.
    Idempotent: canonicalize_url(canonicalize_url(x)) == canonicalize_url(x).
    """
    parts = urlsplit(raw.strip())
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        raise ValueError(f"not an absolute http(s) URL: {raw!r}")
    query = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not (k.lower().startswith("utm_") or k.lower() in _TRACKING_PARAMS)
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def _make_id(canonical_url: str) -> str:
    """16-hex-char id (64 bits: collision-free at this project's scale).

    Scheme and a leading 'www.' are ignored so http/https and www variants
    of the same page collapse to one identity.
    """
    p = urlsplit(canonical_url)
    key = f"{p.netloc.removeprefix('www.')}{p.path}?{p.query}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class TrendItem(BaseModel):
    """One technology/AI update, from collection through AI analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    # ---- Identity (auto-derived from `url` when omitted) ----
    id: str = Field(min_length=1)

    # ---- Filled by collectors ----
    title: str = Field(min_length=1)
    url: str
    source_name: str = Field(min_length=1)  # human label, e.g. "Hacker News", "arXiv cs.AI"
    source_type: SourceType
    content: str = ""  # description/abstract snippet, already plain text (collector strips HTML)
    author: str | None = None
    published_at: datetime | None = None  # None = source gave no date; Module 7 decides policy
    tags: tuple[str, ...] = ()
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ---- Filled by the Gemini stage (Module 8) ----
    relevance_score: int | None = Field(default=None, ge=0, le=10)
    ai_summary: str | None = None

    # ------------------------------------------------------------------
    # Derivation & normalisation (deterministic, no I/O)
    # ------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def _derive_id(cls, data: Any) -> Any:
        """If `id` is absent, derive it from the URL. Explicit ids (e.g. from SQLite) are kept."""
        if isinstance(data, dict) and not data.get("id") and isinstance(data.get("url"), str):
            data = {**data, "id": _make_id(canonicalize_url(data["url"]))}
        return data

    @field_validator("url")
    @classmethod
    def _canonical_url(cls, v: str) -> str:
        return canonicalize_url(v)

    @field_validator("title")
    @classmethod
    def _clean_title(cls, v: str) -> str:
        return _WHITESPACE.sub(" ", v)[:TITLE_MAX].rstrip()

    @field_validator("content")
    @classmethod
    def _clean_content(cls, v: str) -> str:
        return _WHITESPACE.sub(" ", v).strip()[:CONTENT_MAX]

    @field_validator("ai_summary")
    @classmethod
    def _clean_summary(cls, v: str | None) -> str | None:
        # Newlines kept: the WhatsApp formatter (Module 10) may want line breaks.
        return v.strip()[:SUMMARY_MAX] if v is not None else v

    @field_validator("published_at", "collected_at")
    @classmethod
    def _to_utc(cls, v: datetime | None) -> datetime | None:
        """Feeds mix naive and offset datetimes; comparing them raises TypeError.
        Normalise everything to aware UTC (naive input is assumed to be UTC)."""
        if v is None:
            return v
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)

    @field_validator("tags", mode="before")
    @classmethod
    def _clean_tags(cls, v: Any) -> Any:
        if v is None:
            return ()
        if isinstance(v, str):
            v = [v]
        cleaned: list[str] = []
        for tag in v:
            t = _WHITESPACE.sub(" ", str(tag)).strip().lower()
            if t and t not in cleaned:  # order-preserving de-duplication
                cleaned.append(t)
        return tuple(cleaned[:TAGS_MAX])

    # ------------------------------------------------------------------
    # Derived, non-stored helpers
    # ------------------------------------------------------------------
    @property
    def title_key(self) -> str:
        """Normalised title (casefolded, punctuation-free). Lets Module 7 catch the SAME
        story published under different URLs by different sources."""
        return _NON_WORD.sub(" ", self.title.casefold()).strip()

    @property
    def is_analyzed(self) -> bool:
        return self.relevance_score is not None and self.ai_summary is not None

    def with_analysis(self, relevance_score: int, ai_summary: str) -> TrendItem:
        """Return a NEW item carrying Gemini's output (self is never mutated).
        Goes through full validation so an out-of-range score cannot slip in."""
        return TrendItem.model_validate(
            {**self.model_dump(), "relevance_score": relevance_score, "ai_summary": ai_summary}
        )
