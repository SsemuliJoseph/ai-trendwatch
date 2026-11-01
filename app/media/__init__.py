"""
Images for WhatsApp messages: the article's own preview image, or a generated title card.
ALLOWED: models.  FORBIDDEN: ai, database, collectors, processing, messaging, agents.
Knows nothing about WhatsApp or OpenWA: it only turns a TrendItem into image BYTES (or None).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Image:
    data: bytes
    mimetype: str  # image/jpeg | image/png | image/webp
    origin: str  # "article" (the page's og:image) or "card" (generated): for logs and previews only
