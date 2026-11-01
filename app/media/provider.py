"""
FILE: app/media/provider.py
ImageProvider: the ONE callable the delivery stage knows about.  item -> Image | None.

Order: the article's own preview image -> a generated card -> nothing (the message then goes out
as plain text, exactly as before).  It never raises: images are decoration, and a failure here
must never stop or duplicate a delivery.
"""
from __future__ import annotations

import logging

from app.media import Image
from app.media.card import render_card
from app.media.fetcher import ArticleImageFetcher
from app.models import TrendItem

logger = logging.getLogger(__name__)


class ImageProvider:
    def __init__(self, fetcher: ArticleImageFetcher | None = None, *, use_cards: bool = True) -> None:
        self._fetcher = fetcher or ArticleImageFetcher()
        self._use_cards = use_cards

    def __call__(self, item: TrendItem) -> Image | None:
        image = self._fetcher.fetch(item)
        if image is not None:
            return image
        if not self._use_cards:
            return None
        try:
            return render_card(item)
        except Exception as exc:  # e.g. Pillow/font problem: fall back to a text-only message
            logger.warning("title card failed for %s (%s: %s); sending text only", item.id, type(exc).__name__, exc)
            return None
