"""
FILE: app/processing/deduplication.py
MODULE 7b — Deduplication engine.

ARCHITECTURE NOTES
  * TWO IDENTITY SIGNALS (both defined in Module 3):
        item.id         same article, even via tracking links / http-vs-https / www
        item.title_key  same STORY republished by another source under a different URL
  * DEPENDENCY INVERSION: the engine depends on the tiny `SeenStore` Protocol below,
    not on SQLite. `TrendRepository` (Module 6) satisfies it structurally — no
    inheritance, no import — and tests can use a 6-line fake. Rule of thumb: policy
    (this file) depends on abstractions; details (SQLite) depend on nothing above them.
  * READ-ONLY: `filter_new()` never writes. The caller persists survivors with
    `repo.add_new(result.unique)`, whose atomic INSERT OR IGNORE remains the final
    race-proof guard. Keeping "decide" and "write" apart makes each testable alone.
  * FALSE-POSITIVE GUARDS on title matching (a wrongly dropped item is silent data loss):
        - only titles with >= `min_title_words` words are title-deduplicated, because generic
          titles ("Weekly roundup") legitimately recur;
        - matches only look back `title_window_days`, so a title reused months later passes.
  * "First occurrence wins" inside a batch, in the order the caller supplies. Pass items
    newest-first (collectors already sort that way) to keep the freshest copy.
  * Deterministic code only — not an "LLM/agent".
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Protocol

from app.models import TrendItem


class SeenStore(Protocol):
    """What the engine needs to know about history. TrendRepository already provides this."""

    def existing_ids(self, ids: Iterable[str]) -> set[str]: ...

    def existing_title_keys(self, keys: Iterable[str], since: datetime | None = None) -> set[str]: ...


class DuplicateReason(str, Enum):
    DUPLICATE_IN_BATCH = "duplicate_in_batch"  # same id twice in this run
    SAME_TITLE_IN_BATCH = "same_title_in_batch"  # same story, different URL, this run
    ALREADY_SEEN = "already_seen"  # id stored earlier (in ANY status: sent, rejected, ...)
    SAME_TITLE_SEEN = "same_title_seen"  # same story stored earlier within the window


@dataclass(frozen=True)
class Duplicate:
    item: TrendItem
    reason: DuplicateReason


@dataclass(frozen=True)
class DedupResult:
    unique: list[TrendItem] = field(default_factory=list)  # input order preserved
    duplicates: list[Duplicate] = field(default_factory=list)

    def reason_counts(self) -> dict[str, int]:
        return dict(Counter(d.reason.value for d in self.duplicates))


class Deduplicator:
    def __init__(self, store: SeenStore, *, title_window_days: int = 14, min_title_words: int = 3) -> None:
        if title_window_days < 1 or min_title_words < 1:
            raise ValueError("title_window_days and min_title_words must be >= 1")
        self._store = store
        self._window = timedelta(days=title_window_days)
        self._min_words = min_title_words

    def _title_matchable(self, item: TrendItem) -> bool:
        return len(item.title_key.split()) >= self._min_words

    def filter_new(self, items: Iterable[TrendItem], now: datetime | None = None) -> DedupResult:
        """Return items never seen before; everything dropped is reported with a reason."""
        now = now or datetime.now(timezone.utc)
        duplicates: list[Duplicate] = []

        # Phase 1 — inside the batch (no database needed).
        batch_ids: set[str] = set()
        batch_titles: set[str] = set()
        candidates: list[TrendItem] = []
        for item in items:
            if item.id in batch_ids:
                duplicates.append(Duplicate(item, DuplicateReason.DUPLICATE_IN_BATCH))
            elif self._title_matchable(item) and item.title_key in batch_titles:
                duplicates.append(Duplicate(item, DuplicateReason.SAME_TITLE_IN_BATCH))
            else:
                batch_ids.add(item.id)
                if self._title_matchable(item):
                    batch_titles.add(item.title_key)
                candidates.append(item)

        # Phase 2 — against history: two batched queries, not one query per item.
        known_ids = self._store.existing_ids(c.id for c in candidates)
        known_titles = self._store.existing_title_keys(
            (c.title_key for c in candidates if self._title_matchable(c)), since=now - self._window
        )
        unique: list[TrendItem] = []
        for item in candidates:
            if item.id in known_ids:
                duplicates.append(Duplicate(item, DuplicateReason.ALREADY_SEEN))
            elif self._title_matchable(item) and item.title_key in known_titles:
                duplicates.append(Duplicate(item, DuplicateReason.SAME_TITLE_SEEN))
            else:
                unique.append(item)
        return DedupResult(unique=unique, duplicates=duplicates)
