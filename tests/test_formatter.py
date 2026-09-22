"""
FILE: tests/test_formatter.py
MODULE 10 tests — run with:  pytest tests/test_formatter.py
Pure text tests: no network, no OpenWA, no API key.
"""
import re
from datetime import datetime, timedelta, timezone

import pytest

from app.messaging.formatter import (
    SOURCE_MAX,
    SUMMARY_MAX,
    TITLE_MAX,
    FormattedMessage,
    format_item,
    format_messages,
    sanitize_text,
    truncate,
)
from app.models import SourceType, TrendItem

PUBLISHED = datetime(2026, 9, 19, 8, 30, tzinfo=timezone.utc)
EDGE_UNDERSCORE = re.compile(r"(?:(?<!\w)_|_(?!\w))")


def make(n: int = 1, summary: str = "A concise, useful summary.", score: int = 8, **kw) -> TrendItem:
    fields = dict(
        title=f"Notable AI story number {n}", url=f"https://example.com/{n}", source_name="Demo Feed",
        source_type=SourceType.RSS, published_at=PUBLISHED,
    )
    return TrendItem(**{**fields, **kw}).with_analysis(score, summary)


# =====================================================================
# Layout
# =====================================================================
def test_golden_layout():
    msg = format_item(make(1))
    assert msg.item_id == make(1).id
    assert msg.text == (
        "📰 *Notable AI story number 1*\n"
        "\n"
        "A concise, useful summary.\n"
        "\n"
        "📌 Demo Feed · 19 Sep 2026 · relevance 8/10\n"
        "🔗 https://example.com/1"
    )


@pytest.mark.parametrize(
    "source_type, emoji", [(SourceType.RSS, "📰"), (SourceType.ARXIV, "📄"), (SourceType.GITHUB, "💻")]
)
def test_emoji_reflects_source_type(source_type, emoji):
    assert format_item(make(source_type=source_type)).text.startswith(emoji)


def test_missing_publication_date_omits_the_date_part():
    text = format_item(make(published_at=None)).text
    assert "📌 Demo Feed · relevance 8/10" in text


def test_date_uses_the_given_timezone_and_a_fixed_english_month_table():
    late = make(published_at=datetime(2026, 9, 19, 22, 30, tzinfo=timezone.utc))
    eat = timezone(timedelta(hours=3))  # East Africa Time
    assert "19 Sep 2026" in format_item(late).text
    assert "20 Sep 2026" in format_item(late, tz=eat).text  # crosses midnight in EAT


# =====================================================================
# Safety gate
# =====================================================================
def test_unanalysed_items_are_refused():
    raw = TrendItem(title="No analysis yet here", url="https://x.test/1", source_name="S", source_type=SourceType.RSS)
    with pytest.raises(ValueError, match="no AI analysis"):
        format_item(raw)


def test_summary_that_is_empty_after_sanitising_is_refused():
    with pytest.raises(ValueError, match="empty summary"):
        format_item(make(summary="\u202e \u202c"))  # only bidi controls + space


# =====================================================================
# Sanitising untrusted text
# =====================================================================
def test_whatsapp_markup_in_scraped_text_cannot_format_the_message():
    evil = make(
        title="*Free* ~money~ `now` _click_ here",
        summary="Use *bold* and ~strike~ and `code` and _italic_ freely",
    )
    text = format_item(evil).text
    assert text.count("*") == 2  # ONLY the wrapper around the title
    assert "~" not in text and "`" not in text
    assert not EDGE_UNDERSCORE.search(text.replace(evil.url, ""))  # nothing can open/close italics


def test_intra_word_underscores_survive_but_word_edge_ones_do_not():
    assert sanitize_text("use torch_compile_mode today") == "use torch_compile_mode today"
    assert sanitize_text("_italic_ and snake_case") == "\uff3fitalic\uff3f and snake_case"


def test_url_line_is_never_altered():
    item = make(url="https://example.com/a_b_c/_x_?q=1&r=2")
    assert "a_b_c/_x_" in item.url  # underscores survive canonicalisation...
    assert format_item(item).text.endswith(f"🔗 {item.url}")  # ...and the formatter leaves the line untouched


def test_control_characters_and_bidi_overrides_are_removed():
    dirty = "Safe\u202e txt\u2066\x00\x07 ok\u2069"
    assert sanitize_text(dirty) == "Safe txt ok"


def test_newlines_and_tabs_are_collapsed_so_scraped_text_cannot_add_lines_or_blocks():
    assert sanitize_text("line one\n\n> quote\t- item") == "line one > quote - item"
    assert sanitize_text(">starts like a quote") == "\u203astarts like a quote"
    lines = format_item(make(summary="one\n\ntwo\n* three")).text.split("\n")
    assert lines.count("") == 2  # only the two structural blank lines


def test_sanitize_is_idempotent_and_keeps_ordinary_unicode():
    text = "Dataset _v2_ for Luganda – Ateso ✓ *test* `x`"
    once = sanitize_text(text)
    assert sanitize_text(once) == once
    assert "Luganda – Ateso ✓" in once


# =====================================================================
# Size bounds
# =====================================================================
def test_truncate_prefers_word_boundaries_and_marks_the_cut():
    out = truncate("alpha beta gamma delta epsilon", 18)
    assert out == "alpha beta gamma…" and len(out) <= 18
    assert truncate("short", 18) == "short"
    assert truncate("x" * 50, 10) == "x" * 9 + "…"  # no space to break on -> hard cut


def test_long_fields_are_capped_in_the_message():
    msg = format_item(make(title="word " * 200, summary="sentence " * 300, source_name="S" * 500)).text
    lines = msg.split("\n")
    title_line, summary_line, meta_line = lines[0], lines[2], lines[4]
    assert len(title_line) <= TITLE_MAX + 4 and title_line.endswith("…*")
    assert len(summary_line) <= SUMMARY_MAX and summary_line.endswith("…")
    assert len(meta_line.split(" · ")[0]) <= SOURCE_MAX + 2


def test_worst_case_message_stays_small():
    msg = format_item(make(title="w" * 300, summary="s " * 900, source_name="n" * 300, url="https://e.com/" + "p" * 300))
    assert len(msg.text) < 1400


# =====================================================================
# Batches
# =====================================================================
def test_format_messages_keeps_order_and_ids():
    items = [make(3), make(1), make(2)]
    messages = format_messages(items)
    assert [m.item_id for m in messages] == [i.id for i in items]
    assert all(isinstance(m, FormattedMessage) for m in messages)
    assert format_messages([]) == []
    assert format_messages(iter(items))[0].item_id == items[0].id  # accepts generators
