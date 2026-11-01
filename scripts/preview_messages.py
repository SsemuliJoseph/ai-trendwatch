#!/usr/bin/env python3
"""
FILE: scripts/preview_messages.py
Preview how stored items will look in WhatsApp. Sends NOTHING and needs no .env, key or gateway.

    python scripts/preview_messages.py                         # last 5 items already sent: the text part
    python scripts/preview_messages.py --images preview        # ...and save the picture each one would carry
    python scripts/preview_messages.py --status analyzed       # items waiting for a future run
    python scripts/preview_messages.py --limit 10 --tz 3       # East Africa Time (UTC+3, the default)

With --images the article pages ARE fetched (read-only, at most one page + one image per item) to find their
own preview picture; items whose site publishes none, or blocks bots, get the generated title card instead.

DESIGN NOTES
  * Operations tooling: lives in scripts/, may import `app.*`, is imported by nothing in `app/`.
  * It renders through the REAL formatter and the REAL image provider, so the preview is what the pipeline sends.
  * The database is only read (list_by_status); no item changes status.
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets `python scripts/...` find the `app` package

from app.database.repository import ItemStatus, TrendRepository  # noqa: E402
from app.media.provider import ImageProvider  # noqa: E402
from app.messaging.formatter import format_item  # noqa: E402

_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}


def main(argv: list[str] | None = None, *, provider: ImageProvider | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview WhatsApp messages from stored items (sends nothing).")
    parser.add_argument("--db", default="data/trendwatch.db", help="SQLite file (default: data/trendwatch.db)")
    parser.add_argument("--status", choices=[s.value for s in ItemStatus], default=ItemStatus.SENT.value)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--tz", type=float, default=3.0, help="display timezone as hours from UTC (default 3 = EAT)")
    parser.add_argument("--images", metavar="DIR", help="also save the picture each message would carry into DIR")
    args = parser.parse_args(argv)

    if not Path(args.db).exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        return 1

    tz = timezone(timedelta(hours=args.tz))
    with TrendRepository(args.db) as repository:
        items = repository.list_by_status(ItemStatus(args.status), limit=args.limit)
    if not items:
        print(f"no items with status {args.status!r}")
        return 0

    out_dir = Path(args.images) if args.images else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        provider = provider or ImageProvider()

    for number, item in enumerate(items, start=1):
        text = format_item(item, tz).text
        print(text)
        note = f"{len(text)} chars"
        if out_dir:
            image = provider(item)
            if image is None:
                note += " | no image (text-only message)"
            else:
                path = out_dir / f"{number:02d}-{image.origin}.{_EXTENSIONS[image.mimetype]}"
                path.write_bytes(image.data)
                note += f" | image: {image.origin}, {len(image.data) // 1024} KB -> {path}"
        print(f"\n[{note}]\n" + "=" * 52 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
