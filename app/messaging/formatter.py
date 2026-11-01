"""
FILE: app/messaging/formatter.py
MODULE 10 (v3: editorial layout) — WhatsApp message formatter (pure text; no network, no OpenWA).

LAYOUT (the message text; with an image it becomes the image CAPTION, so the picture sits on top)

    🧠 *Introducing GPT-6.1 Terra*

    • OpenAI releases GPT-6.1 Terra with fast computer use.
    • It targets a broad audience, not just developers.
    • A major step for AI capability and everyday apps.

    _OpenAI News · Yesterday_
    Read more:
    https://openai.com/...

WHY IT LOOKS LIKE THIS
  * HEADLINE FIRST, WITH A TOPIC EMOJI. Phones show the first line in the lock-screen preview.
    The emoji (🧠 model, 🛡️ safety, 💼 industry, 🔬 paper, 🛠️ open source, 📰 news) says what kind of
    story it is before the headline is read. No relevance score is shown: it is an internal ranking
    signal, not something a reader needs.
  * THREE SHORT BULLETS instead of a paragraph: scannable in two seconds, like a news channel post.
  * ONE QUIET SOURCE LINE (source + freshness), then "Read more:" with the link on its own line.

BULLETS: the Gemini prompt now asks for up to three short lines. Items analysed before that change hold
a 1-2 sentence paragraph; `to_points` turns each sentence into a bullet, so old and new items look alike.

ARCHITECTURE NOTES
  * Layer rule: imports only `app.models`. It produces TEXT; the OpenWA adapter is the only code that
    touches the gateway. Topic detection is a small deterministic rule table on the TITLE (no LLM call),
    conservative: no confident match -> a neutral 📰, never a wrong label.
  * ONE MESSAGE PER ITEM. Each `FormattedMessage` carries its `item_id`, so the sender can `mark_sent`
    item-by-item: if message 3 of 5 fails, items 1-2 stay sent and 3-5 stay pending for the next run.
  * SAFETY GATE: an item without a Gemini summary/score is REFUSED (ValueError).
  * UNTRUSTED TEXT: titles/summaries come from the internet. `sanitize_text` therefore
      - strips control characters and bidi overrides (U+202A-202E, U+2066-2069);
      - collapses whitespace/newlines, so scraped text can't fake extra lines or list/quote blocks;
      - neutralises WhatsApp markup (* _ ~ `) with look-alike characters. WhatsApp has NO escape
        character, so substitution is the only reliable defence. Intra-word underscores
        (snake_case) are left alone because they do not trigger italics.
    All markup in the message (bold title, italic source line, bullets) is added by THIS module AFTER
    sanitising, so the only `*` and `_` markers are ours. The URL line is never altered.
  * BOUNDED SIZE: title <= 200, three bullets <= 170 each, so a message stays far below the 1024-char
    image-caption limit (only an absurdly long URL can exceed it; the sender then falls back to text).
  * Deterministic: a fixed English month table, an explicit timezone and an injectable `now`.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo

from app.models import SourceType, TrendItem

TITLE_MAX = 200
SOURCE_MAX = 60
MAX_BULLETS = 3
BULLET_MAX = 170  # new Gemini lines are <= 90; the headroom keeps older paragraph-style sentences whole
MIN_SENTENCE = 12  # don't split off a fragment shorter than this (e.g. "Wow. This is big.")
BULLET = "\u2022"

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# ---- Topic tags --------------------------------------------------------------------------
# Defaults by where the item came from. A paper is a paper and a repo is a repo, so keyword rules
# are applied to RSS only: "we introduce a model" in an arXiv title is not a product launch.
_SOURCE_TAG = {
    SourceType.RSS: ("\U0001f4f0", "News"),  # 📰
    SourceType.ARXIV: ("\U0001f52c", "Research"),  # 🔬
    SourceType.GITHUB: ("\U0001f6e0\ufe0f", "Open source"),  # 🛠️
}
_LAUNCH = re.compile(r"\b(introduc\w*|launch\w*|releas\w*|unveil\w*|announc\w*|debut\w*)\b", re.I)
# Named model families only. A bare "model(s)" is too loose: "launch ... AI world models" is a startup story.
# Missing a real launch just falls back to the neutral "News" tag, which is the safe failure.
_MODEL = re.compile(r"\b(gpt[\w.-]*|claude|gemini|llama[\w.-]*|mistral|qwen[\w.-]*|deepseek|grok)\b", re.I)
_SAFETY = re.compile(
    r"\b(safety|safeguards?|alignment|misalignment|guardrails?|jailbreaks?|vuln\w*|exploits?|flaws?|"
    r"patch(?:es|ed)?|hack\w*|compromis\w*|malware|0-day|zero-day|cyber\w*|breach\w*|privacy|monitor\w*)\b",
    re.I,
)
# Money in a headline ("$17M", "$1.25B") is the strongest cheap signal of a business story.
_BUSINESS = re.compile(
    r"(?:\b(?:raises?|funding|acquir\w*|acquisition|valuation|valued|ipo|lawsuit|sues?|regulat\w*|partnerships?)\b|\$\d)",
    re.I,
)
# Ordered: first rule whose patterns ALL match wins. "Launch + model" comes before "safety" so that
# "Claude Opus 5.5 ships with stricter cyber safeguards" is tagged as a model release, not a safety story.
_RSS_RULES = (
    ("\U0001f9e0", "New model", (_LAUNCH, _MODEL)),  # 🧠
    ("\U0001f6e1\ufe0f", "Safety & security", (_SAFETY,)),  # 🛡️
    ("\U0001f4bc", "Industry", (_BUSINESS,)),  # 💼
)

# ---- Sanitising ----------------------------------------------------------------------------
# WhatsApp markup characters -> visually similar characters that carry no meaning.
_MARKUP_LOOKALIKES = str.maketrans({"*": "\u2217", "~": "\u223c", "`": "\u02cb"})
_EDGE_UNDERSCORE = re.compile(r"(?:(?<!\w)_|_(?!\w))")  # underscore at a word edge could open/close italics
_BIDI_CONTROLS = dict.fromkeys([*range(0x202A, 0x202F), *range(0x2066, 0x206A)])
_WHITESPACE = re.compile(r"\s+")
_LEADING_QUOTE = re.compile(r"^>")  # '>' at line start renders as a quote block
_LIST_PREFIX = re.compile(r"^\s*(?:[-*\u2022\u00b7\u2013\u2014]+|\d{1,2}[.)])\s+")  # "- ", "* ", "• ", "1. ", "2) "

# ---- Sentence splitting --------------------------------------------------------------------
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"\u201c\u2018'(])")
_ABBREVIATIONS = frozenset(
    {"e.g.", "i.e.", "vs.", "etc.", "approx.", "dr.", "mr.", "mrs.", "ms.", "prof.", "inc.", "ltd.", "co.",
     "no.", "fig.", "st."}
)
_INITIALS = re.compile(r"^(?:[a-z]\.)+$", re.I)  # "J." or "U.S." — not the end of a sentence


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


def split_sentences(text: str) -> list[str]:
    """Sentences of a one-line paragraph. Abbreviations ("e.g.", "Inc.") and initials ("U.S.") don't end one;
    a fragment shorter than MIN_SENTENCE is glued to its neighbour instead of becoming its own bullet."""
    sentences, start = [], 0
    for match in _SENTENCE_BREAK.finditer(text):
        last_word = text[start: match.start()].rsplit(" ", 1)[-1]
        if last_word.lower() in _ABBREVIATIONS or _INITIALS.match(last_word):
            continue
        sentences.append(text[start: match.start()])
        start = match.end()
    sentences.append(text[start:])
    merged: list[str] = []
    for sentence in sentences:
        if merged and (len(sentence) < MIN_SENTENCE or len(merged[-1]) < MIN_SENTENCE):
            merged[-1] = f"{merged[-1]} {sentence}"
        else:
            merged.append(sentence)
    return merged


def to_points(summary: str) -> list[str]:
    """Up to MAX_BULLETS clean one-line bullets from a summary (newline-separated lines, or a paragraph)."""
    lines = summary.splitlines() if "\n" in summary.strip() else split_sentences(sanitize_text(summary))
    points: list[str] = []
    for line in lines:
        clean = sanitize_text(_LIST_PREFIX.sub("", line.strip()))  # strip the list marker BEFORE sanitising
        if clean:
            points.append(truncate(clean, BULLET_MAX))
    return points[:MAX_BULLETS]


def _topic_tag(item: TrendItem, clean_title: str) -> tuple[str, str]:
    """(emoji, label). Conservative on purpose: unsure -> the neutral source default. Only the emoji is shown."""
    if item.source_type is SourceType.RSS:
        for emoji, label, patterns in _RSS_RULES:
            if all(p.search(clean_title) for p in patterns):
                return emoji, label
    return _SOURCE_TAG[item.source_type]


def _freshness(moment: datetime, now: datetime, tz: tzinfo) -> str:
    """'Today' / 'Yesterday' / '3 days ago', then a plain date. Freshness is a reason to open a message."""
    local, today = moment.astimezone(tz), now.astimezone(tz)
    days = (today.date() - local.date()).days
    if days <= 0:  # 0 = today; negative = a feed with a slightly-ahead clock, still "fresh"
        return "Today"
    if days == 1:
        return "Yesterday"
    if days < 7:
        return f"{days} days ago"
    # Same-year dates drop the year to keep the tag line short; other years keep it for clarity.
    return f"{local.day} {_MONTHS[local.month - 1]}" + ("" if local.year == today.year else f" {local.year}")


def format_item(item: TrendItem, tz: tzinfo = timezone.utc, now: datetime | None = None) -> FormattedMessage:
    """Render one analysed item. Raises ValueError if it has no AI analysis (never send unreviewed)."""
    if not item.is_analyzed:
        raise ValueError(f"item {item.id} has no AI analysis; refusing to format for delivery")
    now = now or datetime.now(timezone.utc)

    title = truncate(sanitize_text(item.title), TITLE_MAX) or "(untitled)"
    points = to_points(item.ai_summary or "")
    if not points:
        raise ValueError(f"item {item.id} has an empty summary after sanitising")

    emoji, _label = _topic_tag(item, title)
    source = [truncate(sanitize_text(item.source_name), SOURCE_MAX)]
    if item.published_at is not None:
        source.append(_freshness(item.published_at, now, tz))
    source_line = " \u00b7 ".join(source)  # built outside the f-string: backslashes there fail before Python 3.12

    lines = [f"{emoji} *{title}*", ""]  # WhatsApp bold; the title has no '*' left after sanitising
    lines += [f"{BULLET} {point}" for point in points]
    lines += ["", f"_{source_line}_", "Read more:", item.url]  # URL last, alone on its line, never altered
    return FormattedMessage(item_id=item.id, text="\n".join(lines))


def format_messages(
    items: Iterable[TrendItem], tz: tzinfo = timezone.utc, now: datetime | None = None
) -> list[FormattedMessage]:
    """One message per item, input order preserved."""
    return [format_item(item, tz, now) for item in items]


if __name__ == "__main__":
    # Preview how the messages will look:  python -m app.messaging.formatter
    from datetime import timedelta

    moment = datetime.now(timezone.utc)
    samples = [
        ("Introducing GPT-6.1 Terra", "OpenAI News", SourceType.RSS, moment - timedelta(hours=3),
         "OpenAI releases GPT-6.1 Terra with fast computer use\nIt targets a broad audience, not just developers\n"
         "A major step for AI capability and everyday apps"),
        ("An ODE-based Sampler for Discrete Diffusion Models", "arXiv cs.LG", SourceType.ARXIV, moment - timedelta(days=3),
         "The paper proposes a faster sampler for discrete diffusion models. It could cut generation cost for text and code."),
    ]
    for title, source, kind, published, summary in samples:
        item = TrendItem(
            title=title, url="https://example.com/story", source_name=source, source_type=kind, published_at=published
        ).with_analysis(8, summary)
        print(format_item(item).text + "\n" + "-" * 50)
