"""
FILE: app/media/card.py
A generated title card, used when an article publishes no preview image (all arXiv papers, and sites
that block bots), so every message in the group still opens with a picture.

DESIGN: a calm dark gradient in a colour that depends on the kind of item, a small label chip, the
headline in large type, and the source plus "AI TrendWatch" at the bottom. Text only: no emoji (they
need a colour-emoji font that may not be installed) and no stock imagery (nothing is invented).

The headline is untrusted internet text, but it is only ever DRAWN, never parsed or executed, so no
markup sanitising is needed here. Missing glyphs (e.g. CJK) render as boxes at worst.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from app.media import Image
from app.models import SourceType, TrendItem

SIZE = (1200, 630)  # 1.9:1, the shape WhatsApp and social cards crop best
MAX_TITLE_LINES = 4
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Debian/Ubuntu (fonts-dejavu-core)
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)
# source type -> (label, gradient top, gradient bottom, accent)
_THEMES = {
    SourceType.RSS: ("NEWS", (18, 32, 64), (10, 14, 28), (96, 165, 250)),
    SourceType.ARXIV: ("RESEARCH", (38, 24, 66), (14, 10, 30), (167, 139, 250)),
    SourceType.GITHUB: ("OPEN SOURCE", (14, 52, 44), (8, 18, 20), (52, 211, 153)),
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)  # Pillow >= 10.1 ships a scalable built-in font


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _fit_title(draw: ImageDraw.ImageDraw, title: str, max_width: int):
    """Largest type size (72 -> 40) at which the headline fits in MAX_TITLE_LINES; ellipsis if it never does."""
    for size in range(72, 39, -4):
        font = _font(size)
        lines = _wrap(draw, title, font, max_width)
        if len(lines) <= MAX_TITLE_LINES:
            return font, lines, size
    font = _font(40)
    lines = _wrap(draw, title, font, max_width)[:MAX_TITLE_LINES]
    lines[-1] = lines[-1].rstrip(" ,;:.-") + "\u2026"
    return font, lines, 40


def render_card(item: TrendItem) -> Image:
    label, top, bottom, accent = _THEMES[item.source_type]
    width, height = SIZE
    canvas = PILImage.new("RGB", SIZE)
    draw = ImageDraw.Draw(canvas)
    for y in range(height):  # vertical gradient, one line at a time
        t = y / (height - 1)
        draw.line([(0, y), (width, y)], fill=tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))
    draw.rectangle([0, 0, 14, height], fill=accent)  # accent bar

    margin = 84
    chip_font = _font(26)
    chip_w = int(draw.textlength(label, font=chip_font)) + 44
    draw.rounded_rectangle([margin, 64, margin + chip_w, 112], radius=24, fill=accent)
    draw.text((margin + 22, 88), label, font=chip_font, fill=(10, 14, 28), anchor="lm")

    title_font, lines, size = _fit_title(draw, " ".join(item.title.split()), width - 2 * margin)
    line_height = int(size * 1.22)
    y = 160 + (MAX_TITLE_LINES - len(lines)) * line_height // 3  # nudge short headlines toward the middle
    for line in lines:
        draw.text((margin, y), line, font=title_font, fill=(245, 247, 250))
        y += line_height

    footer_font = _font(28)
    draw.text((margin, height - 56), item.source_name[:60], font=footer_font, fill=(190, 198, 214), anchor="lm")
    draw.text((width - margin, height - 56), "AI TrendWatch", font=footer_font, fill=accent, anchor="rm")

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return Image(data=buffer.getvalue(), mimetype="image/png", origin="card")
