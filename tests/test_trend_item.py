"""
FILE: tests/test_trend_item.py
MODULE 3 tests — run with:  pytest tests/test_trend_item.py
Pure unit tests: no network, no database, no API keys.
"""
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.models import SourceType, TrendItem, canonicalize_url


def make(**overrides) -> TrendItem:
    """Factory with valid defaults; override only what a test cares about."""
    base = dict(
        title="  New   model released ",
        url="https://Example.com/post/?utm_source=x&b=2&a=1#frag",
        source_name="Example RSS",
        source_type=SourceType.RSS,
    )
    return TrendItem(**{**base, **overrides})


# ---------- URL canonicalisation & identity ----------
def test_canonicalize_url():
    assert (
        canonicalize_url("HTTPS://Example.com/post/?utm_source=x&b=2&a=1#frag")
        == "https://example.com/post?a=1&b=2"
    )
    assert canonicalize_url("https://example.com") == "https://example.com/"


def test_canonicalize_is_idempotent():
    once = canonicalize_url("https://Example.com/a/?b=1&utm_medium=z")
    assert canonicalize_url(once) == once


def test_id_ignores_tracking_scheme_www_and_fragment():
    a = make(url="https://www.example.com/post?utm_campaign=z")
    b = make(url="http://example.com/post/#top")
    assert a.id == b.id
    assert len(a.id) == 16


def test_id_differs_for_different_pages():
    assert make(url="https://example.com/a").id != make(url="https://example.com/b").id
    assert make(url="https://example.com/a?id=1").id != make(url="https://example.com/a?id=2").id


def test_explicit_id_is_preserved():
    assert make(id="abc123").id == "abc123"


@pytest.mark.parametrize("bad", ["ftp://example.com/a", "not a url", "/relative/path", ""])
def test_invalid_url_rejected(bad):
    with pytest.raises(ValidationError):
        make(url=bad)


# ---------- Text normalisation ----------
def test_title_whitespace_collapsed_and_truncated():
    assert make().title == "New model released"
    assert len(make(title="x" * 1000).title) == 300


def test_empty_title_or_source_rejected():
    with pytest.raises(ValidationError):
        make(title="   ")
    with pytest.raises(ValidationError):
        make(source_name="")


def test_title_key_matches_same_story_across_formatting():
    a = make(title="OpenAI Releases GPT-6!")
    b = make(title="openai releases   gpt 6")
    assert a.title_key == b.title_key == "openai releases gpt 6"


def test_tags_normalised_lowercase_unique_capped():
    assert make(tags=["AI ", "ai", " LLM", ""]).tags == ("ai", "llm")
    assert len(make(tags=[f"t{i}" for i in range(50)]).tags) == 10


# ---------- Datetimes ----------
def test_naive_datetime_assumed_utc():
    item = make(published_at=datetime(2026, 9, 19, 12, 0))
    assert item.published_at == datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def test_aware_datetime_converted_to_utc():
    eat = timezone(timedelta(hours=3))  # East Africa Time
    item = make(published_at=datetime(2026, 9, 19, 12, 0, tzinfo=eat))
    assert item.published_at.hour == 9 and item.published_at.utcoffset() == timedelta(0)


def test_collected_at_defaults_to_aware_utc_now():
    assert make().collected_at.tzinfo is not None


# ---------- Immutability & strictness ----------
def test_item_is_frozen():
    with pytest.raises(ValidationError):
        make().title = "changed"


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        make(titel="typo")


# ---------- AI analysis ----------
def test_with_analysis_returns_new_item_and_leaves_original_untouched():
    original = make()
    analysed = original.with_analysis(8, "  Short summary.  ")
    assert not original.is_analyzed
    assert analysed.is_analyzed
    assert (analysed.relevance_score, analysed.ai_summary) == (8, "Short summary.")
    assert analysed.id == original.id


@pytest.mark.parametrize("score", [-1, 11])
def test_score_out_of_range_rejected(score):
    with pytest.raises(ValidationError):
        make().with_analysis(score, "x")


# ---------- Serialisation round-trip (SQLite / JSON safety) ----------
def test_json_roundtrip_is_lossless():
    item = make(published_at=datetime(2026, 9, 19, 12, 0), tags=["ai"]).with_analysis(7, "S")
    assert TrendItem.model_validate_json(item.model_dump_json()) == item
