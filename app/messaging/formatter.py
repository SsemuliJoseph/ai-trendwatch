"""
FILE: app/messaging/formatter.py
MODULE 10 — WhatsApp message formatter (pure text; no network, no OpenWA).

ARCHITECTURE NOTES
  * Layer rule: imports only `app.models`. It produces TEXT; the OpenWA adapter
    (Module 12) is the only code that will ever touch the gateway.
  * ONE MESSAGE PER ITEM (not one big digest). Each `FormattedMessage` carries its
    `item_id`, so the sender can `mark_sent` item-by-item: if message 3 of 5 fails,
    items 1-2 are recorded as sent and 3-5 stay pending for the next run, with no
    double-posting and no lost items. The per-run cap and delay live in Settings
    (MAX_MESSAGES_PER_RUN, MESSAGE_DELAY_SECONDS).
  * SAFETY GATE: an item without a Gemini summary/score is REFUSED (ValueError):
    nothing unreviewed can reach the class group through this path.
  * UNTRUSTED TEXT: titles/summaries come from the internet. `sanitize_text` therefore
      - strips control characters and bidi overrides (U+202A-202E, U+2066-2069), which can
        visually reorder text to disguise a message;
      - collapses whitespace/newlines, so scraped text can't fake extra lines or list/quote blocks;
      - neutralises WhatsApp markup (* _ ~ `) with look-alike characters. WhatsApp has NO escape
        character, so removal or substitution is the only reliable defence against text that
        would otherwise bold/strike/monospace half the message. Intra-word underscores
        (snake_case) are left alone because they do not trigger italics.
    The URL line is never altered: changing a link would break it.
  * BOUNDED SIZE: title <= 200, summary <= 500 chars, cut at a word boundary with '…'.
  * Deterministic: dates use a fixed English month table (no locale surprises) and an
    explicit timezone argument.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo

from app.models import SourceType, TrendItem

TITLE_MAX = 200
SUMMARY_MAX = 500
SOURCE_MAX = 60

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_SOURCE_EMOJI = {SourceType.RSS: "📰", SourceType.ARXIV: "📄", SourceType.GITHUB: "💻"}

# WhatsApp markup characters -> visually similar characters that carry no meaning.
_MARKUP_LOOKALIKES = str.maketrans({"*": "\u2217", "~": "\u223c", "`": "\u02cb"})
_EDGE_UNDERSCORE = re.compile(r"(?:(?<!\w)_|_(?!\w))")  # underscore at a word edge could open/close italics
_BIDI_CONTROLS = dict.fromkeys([*range(0x202A, 0x202F), *range(0x2066, 0x206A)])
_WHITESPACE = re.compile(r"\s+")
_LEADING_QUOTE = re.compile(r"^>")  # '>' at line start renders as a quote block


@dataclass(frozen=True)
class FormattedMessage:
    item_id: str  # lets the sender mark exactly this item as sent
    text: str


def sanitize_text(text: str) -> str:
    """Untrusted text -> single-line, markup-free, control-free text. Idempotent."""
    text = text.translate(_BIDI_CONTROLS)
    text = _WHITESPACE.sub(" ", text)  # newlines/tabs -> single spaces
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc").strip()
    text = text.translate(_MARKUP_LOOKALIKES)
    text = _EDGE_UNDERSCORE.sub("\uff3f", text)  # fullwidth low line
    return _LEADING_QUOTE.sub("\u203a", text)  # single right-pointing angle quotation mark


def truncate(text: str, limit: int) -> str:
    """Cut to at most `limit` chars, preferring a word boundary, marking the cut with '…'."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space >= limit // 2:  # only back up to a space if it doesn't throw away too much
        cut = cut[:space]
    return cut.rstrip(" ,;:.-") + "\u2026"


def _format_date(moment: datetime, tz: tzinfo) -> str:
    local = moment.astimezone(tz)
    return f"{local.day} {_MONTHS[local.month - 1]} {local.year}"


def format_item(item: TrendItem, tz: tzinfo = timezone.utc) -> FormattedMessage:
    """Render one analysed item. Raises ValueError if it has no AI analysis (never send unreviewed)."""
    if not item.is_analyzed:
        raise ValueError(f"item {item.id} has no AI analysis; refusing to format for delivery")

    title = truncate(sanitize_text(item.title), TITLE_MAX) or "(untitled)"
    summary = truncate(sanitize_text(item.ai_summary or ""), SUMMARY_MAX)
    if not summary:
        raise ValueError(f"item {item.id} has an empty summary after sanitising")

    meta = [truncate(sanitize_text(item.source_name), SOURCE_MAX)]
    if item.published_at is not None:
        meta.append(_format_date(item.published_at, tz))
    meta.append(f"relevance {item.relevance_score}/10")
    meta_line = " \u00b7 ".join(meta)  # built outside the f-string: backslashes in f-string expressions fail before Python 3.12

    text = "\n".join(
        [
            f"{_SOURCE_EMOJI[item.source_type]} *{title}*",  # WhatsApp bold; title has no '*' left
            "",
            summary,
            "",
            f"\U0001f4cc {meta_line}",
            f"\U0001f517 {item.url}",  # deliberately untouched
        ]
    )
    return FormattedMessage(item_id=item.id, text=text)


def format_messages(items: Iterable[TrendItem], tz: tzinfo = timezone.utc) -> list[FormattedMessage]:
    """One message per item, input order preserved."""
    return [format_item(item, tz) for item in items]


if __name__ == "__main__":
    # Preview how a message will look:  python -m app.messaging.formatter
    sample = TrendItem(
        title="Open-source lab releases 7B model that beats *larger* rivals",
        url="https://example.com/open_model_release",
        source_name="Demo feed",
        source_type=SourceType.RSS,
        published_at=datetime(2026, 9, 19, 8, 30, tzinfo=timezone.utc),
    ).with_analysis(9, "A permissively licensed 7B model matches bigger systems on coding benchmarks, with weights released.")
    print(format_item(sample).text)
