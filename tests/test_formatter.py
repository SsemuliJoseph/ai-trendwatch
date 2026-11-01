"""
FILE: tests/test_formatter.py
MODULE 10 (v3) tests — run with:  pytest tests/test_formatter.py
Pure text tests: no network, no OpenWA, no API key. `now` is injected so dates are deterministic.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.messaging.formatter import (
    BULLET_MAX,
    MAX_BULLETS,
    SOURCE_MAX,
    TITLE_MAX,
    FormattedMessage,
    format_item,
    format_messages,
    sanitize_text,
    split_sentences,
    to_points,
    truncate,
)
from app.models import SourceType, TrendItem

PUBLISHED = datetime(2026, 9, 19, 8, 30, tzinfo=timezone.utc)
NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)  # same day as PUBLISHED -> "Today"
BULLETS = "OpenAI releases a new model today\nIt is faster and cheaper to run\nStudents get a free tier"


def make(n: int = 1, summary: str = BULLETS, score: int = 8, **kw) -> TrendItem:
    fields = dict(
        title=f"Notable AI story number {n}", url=f"https://example.com/{n}", source_name="Demo Feed",
        source_type=SourceType.RSS, published_at=PUBLISHED,
    )
    return TrendItem(**{**fields, **kw}).with_analysis(score, summary)


def render(item: TrendItem, **kw) -> str:
    return format_item(item, now=kw.pop("now", NOW), **kw).text


# =====================================================================
# Layout
# =====================================================================
def test_golden_layout():
    msg = format_item(make(1), now=NOW)
    assert msg.item_id == make(1).id
    assert msg.text == (
        "📰 *Notable AI story number 1*\n"
        "\n"
        "• OpenAI releases a new model today\n"
        "• It is faster and cheaper to run\n"
        "• Students get a free tier\n"
        "\n"
        "_Demo Feed · Today_\n"
        "Read more:\n"
        "https://example.com/1"
    )


def test_first_line_is_topic_emoji_plus_bold_headline():
    assert render(make()).split("\n")[0] == "📰 *Notable AI story number 1*"  # what the lock screen shows


@pytest.mark.parametrize("score", [0, 6, 8, 10])
def test_the_relevance_score_is_never_shown(score):
    text = render(make(score=score))
    assert "elevance" not in text and f"{score}/10" not in text and "🔥" not in text and "⭐" not in text


def test_link_is_last_and_alone_on_its_line():
    lines = render(make()).split("\n")
    assert lines[-2:] == ["Read more:", "https://example.com/1"]


def test_missing_publication_date_leaves_just_the_source():
    assert "\n_Demo Feed_\n" in render(make(published_at=None))


@pytest.mark.parametrize(
    "source_type, emoji", [(SourceType.RSS, "📰"), (SourceType.ARXIV, "🔬"), (SourceType.GITHUB, "🛠️")]
)
def test_emoji_reflects_source_type_when_no_keyword_rule_applies(source_type, emoji):
    assert render(make(source_type=source_type)).startswith(emoji + " *")


# =====================================================================
# Topic emoji (RSS titles only; conservative by design)
# =====================================================================
@pytest.mark.parametrize(
    "title, emoji",
    [
        ("Introducing GPT-6.1 Terra", "🧠"),
        ("Anthropic launches Claude Opus 5.5 with stricter safeguards for cybersecurity", "🧠"),  # launch beats safety
        ("Mistral releases weights for its trillion-parameter AI model", "🧠"),
        ("How we detect frontier model misalignment", "🛡️"),
        ("Meta patches Muse exploit that let attackers control the AI agent", "🛡️"),
        ("Deepgram acquires Pipecat-maker Daily", "💼"),
        ("Ema raises $77M as AI eats enterprise software", "💼"),
        ("Ex-Palantir execs launch Dream with $17M to build AI world models", "💼"),  # money, not a model launch
        ("Sam Altman's remarks at the United Nations Security Council", "📰"),  # ambiguous word: no tag
        ("Microsoft's AI future has fewer jobs", "📰"),
    ],
)
def test_rss_topic_rules(title, emoji):
    assert render(make(title=title)).startswith(f"{emoji} *")


def test_keyword_rules_do_not_apply_to_papers():
    paper = make(title="We introduce a new model for discrete diffusion", source_type=SourceType.ARXIV)
    assert render(paper).startswith("🔬 *")


# =====================================================================
# Freshness and dates
# =====================================================================
@pytest.mark.parametrize(
    "delta, label",
    [
        (timedelta(hours=-3), "Today"),  # feed clock slightly ahead: still fresh
        (timedelta(hours=3), "Today"),
        (timedelta(days=1), "Yesterday"),
        (timedelta(days=3), "3 days ago"),
        (timedelta(days=6), "6 days ago"),
        (timedelta(days=7), "12 Sep"),  # same year: no year shown
    ],
)
def test_freshness_labels(delta, label):
    assert f"_Demo Feed · {label}_" in render(make(published_at=NOW - delta))


def test_other_year_keeps_the_year():
    item = make(published_at=datetime(2025, 12, 30, 10, 0, tzinfo=timezone.utc))
    assert "_Demo Feed · 30 Dec 2025_" in render(item)


def test_dates_use_the_given_timezone():
    late = make(published_at=datetime(2026, 9, 19, 22, 30, tzinfo=timezone.utc))
    eat = timezone(timedelta(hours=3))  # East Africa Time: 01:30 on the 20th
    now = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
    assert "_Demo Feed · 19 Sep_" in render(late, now=now)
    assert "_Demo Feed · 20 Sep_" in render(late, tz=eat, now=now)  # crosses midnight in EAT


# =====================================================================
# Bullets
# =====================================================================
def test_newline_separated_summary_becomes_bullets():
    assert to_points(BULLETS) == ["OpenAI releases a new model today", "It is faster and cheaper to run", "Students get a free tier"]


def test_old_paragraph_summaries_become_one_bullet_per_sentence():
    old = "Something notable happened in AI today. This matters because students should know."
    assert to_points(old) == ["Something notable happened in AI today.", "This matters because students should know."]
    text = render(make(summary=old))
    assert text.count("•") == 2


def test_a_single_sentence_is_a_single_bullet():
    assert to_points("One sentence only, nothing more to add.") == ["One sentence only, nothing more to add."]


@pytest.mark.parametrize("marker", ["- ", "* ", "• ", "· ", "– ", "1. ", "2) ", "  - "])
def test_list_markers_the_model_adds_anyway_are_stripped(marker):
    assert to_points(f"{marker}First point here\n{marker}Second point here") == ["First point here", "Second point here"]


def test_blank_lines_are_ignored_and_bullets_are_capped():
    summary = "one point here\n\n\ntwo point here\nthree point here\nfour point here\nfive point here"
    assert to_points(summary) == ["one point here", "two point here", "three point here"][:MAX_BULLETS]


def test_each_bullet_is_length_capped():
    (point,) = to_points("word " * 100)
    assert len(point) <= BULLET_MAX and point.endswith("…")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("A single sentence with no boundary at all in it", ["A single sentence with no boundary at all in it"]),
        ("Ends with a decimal like v2.5 and keeps going without a break", ["Ends with a decimal like v2.5 and keeps going without a break"]),
        ("The U.S. government announced new chip rules for exporters", ["The U.S. government announced new chip rules for exporters"]),
        ("Dr. Smith presented the result at the conference in Kampala", ["Dr. Smith presented the result at the conference in Kampala"]),
        ("Models such as e.g. GPT variants were compared on coding tasks", ["Models such as e.g. GPT variants were compared on coding tasks"]),
        ("Wow. This is big.", ["Wow. This is big."]),  # fragments are glued back together
        (
            "Researchers at Acme Inc. released a new dataset today. It covers four languages.",
            ["Researchers at Acme Inc. released a new dataset today.", "It covers four languages."],
        ),
    ],
)
def test_sentence_splitting(text, expected):
    assert split_sentences(text) == expected


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
        summary="Use *bold* and ~strike~\n`code` and _italic_ freely\n* starred _bullet_",
        source_name="*evil* _feed_",
    )
    text = render(evil)
    assert text.count("*") == 2  # ONLY the wrapper around the title
    assert "~" not in text and "`" not in text
    body = text.replace(evil.url, "")
    assert body.count("_") == 2  # only the opening and closing italic markers on the source line


def test_intra_word_underscores_survive_but_word_edge_ones_do_not():
    assert sanitize_text("use torch_compile_mode today") == "use torch_compile_mode today"
    assert sanitize_text("_italic_ and snake_case") == "\uff3fitalic\uff3f and snake_case"


def test_url_line_is_never_altered():
    item = make(url="https://example.com/a_b_c/_x_?q=1&r=2")
    assert render(item).endswith("\n" + item.url)


def test_control_characters_and_bidi_overrides_are_removed():
    assert sanitize_text("Safe\u202e txt\u2066\x00\x07 ok\u2069") == "Safe txt ok"


def test_newlines_inside_a_bullet_source_cannot_forge_extra_structure():
    assert sanitize_text("line one\n\n> quote\t- item") == "line one > quote - item"
    assert sanitize_text(">starts like a quote") == "\u203astarts like a quote"
    lines = render(make(title="a title\n\n> with newlines\n* and a star")).split("\n")
    assert lines[0].count("*") == 2 and lines.count("") == 2  # only the two structural blank lines


def test_sanitize_is_idempotent_and_keeps_ordinary_unicode():
    once = sanitize_text("Dataset _v2_ for Luganda – Ateso ✓ *test* `x`")
    assert sanitize_text(once) == once and "Luganda – Ateso ✓" in once


# =====================================================================
# Size bounds
# =====================================================================
def test_truncate_prefers_word_boundaries_and_marks_the_cut():
    assert truncate("alpha beta gamma delta epsilon", 18) == "alpha beta gamma…"
    assert truncate("short", 18) == "short"
    assert truncate("x" * 50, 10) == "x" * 9 + "…"  # no space to break on -> hard cut


def test_long_fields_are_capped_in_the_message():
    text = render(make(title="word " * 200, summary="sentence " * 300, source_name="S" * 500))
    lines = text.split("\n")
    assert len(lines[0]) <= TITLE_MAX + 6 and lines[0].endswith("…*")  # emoji + space + two asterisks
    assert all(len(line) <= BULLET_MAX + 2 for line in lines if line.startswith("•"))
    assert len(lines[-3]) <= SOURCE_MAX + 2 + len(" · Today")  # italic source line


def test_worst_case_message_fits_an_image_caption_with_a_long_url():
    item = make(title="w" * 300, summary="s " * 900, source_name="n" * 300, url="https://e.com/" + "p" * 300)
    assert len(render(item)) < 1000  # MAX_CAPTION_CHARS in the gateway


def test_a_realistic_worst_case_still_fits_an_image_caption():
    summary = "\n".join(["b" * 170] * 3)  # three maximum-length bullets
    item = make(title="t " * 60, summary=summary, source_name="n" * 60, url="https://e.com/" + "p" * 136)
    assert len(render(item)) < 1000


# =====================================================================
# Batches
# =====================================================================
def test_format_messages_keeps_order_and_ids():
    items = [make(3), make(1), make(2)]
    messages = format_messages(items, now=NOW)
    assert [m.item_id for m in messages] == [i.id for i in items]
    assert all(isinstance(m, FormattedMessage) for m in messages)
    assert format_messages([]) == []
    assert format_messages(iter(items), now=NOW)[0].item_id == items[0].id  # accepts generators
