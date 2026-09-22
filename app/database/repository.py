"""
FILE: app/database/repository.py
MODULE 6 — Repository over SQLite.

ARCHITECTURE NOTES
  * REPOSITORY PATTERN: the rest of the app calls intention-revealing methods
    (add_new, pending_delivery, mark_sent) and never sees SQL. Swapping SQLite for
    another store later would touch only this package.
  * Layer rule: imports ONLY `app.models` + `app.database.schema`. It receives
    the DB path as an argument (agents/ reads `Settings.database_path`), and it knows
    nothing of Gemini, OpenWA or the collectors.
  * DEDUP PRIMITIVES, not policy: this layer answers "which of these ids / title
    keys do I already know?". WHAT counts as a duplicate is Module 7's decision.
  * `add_new` uses INSERT OR IGNORE on the PRIMARY KEY, so "is it new?" and
    "store it" are ONE atomic statement: no check-then-insert race, and
    duplicates inside one batch are handled too.
  * GUARDED TRANSITIONS: every status update has a `WHERE status IN (...)` guard
    and returns bool. An item can't jump NEW->SENT or be re-analysed after SENT,
    which protects the "never message twice / never unreviewed" safety goals.
  * CLAIM PROTOCOL (Module 15): delivery is  claim_for_delivery -> send -> mark_sent.
    The claim is one atomic UPDATE, so two processes can never both send the same item, and a
    crash between "sent" and "mark_sent" leaves a SENDING row that `recover_interrupted_deliveries`
    turns into REJECTED (outcome unknown, never re-sent) instead of a duplicate post.
  * CONCURRENCY: one connection (`check_same_thread=False`) serialised by an
    RLock, because LangGraph may run nodes on worker threads. WAL journal mode
    lets a second process (e.g. a `sqlite3` CLI inspection) read while we write.
  * SQL uses `?` placeholders for every value. The only string-built SQL is the
    placeholder list of `IN (?,?,?)`, which contains no user data.

USAGE
    with TrendRepository(settings.database_path) as repo:
        fresh = repo.add_new(items)
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.database.schema import DatabaseError, ItemStatus, migrate
from app.models import TrendItem

__all__ = ["TrendRepository", "ItemStatus", "DatabaseError"]

_IN_CHUNK = 500  # stay far below SQLite's bound-variable limit (999 on old builds)

_INSERT_SQL = """
INSERT OR IGNORE INTO trend_items
    (id, title, url, source_name, source_type, content, author, published_at,
     tags, collected_at, title_key, relevance_score, ai_summary, status)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# ----------------------------------------------------------------------
# Row <-> TrendItem mapping (the ONLY place that knows the column layout)
# ----------------------------------------------------------------------
def _dt_to_str(value: datetime | None) -> str | None:
    """Fixed-width UTC ISO text (…T10:00:00.000000+00:00) -> lexicographic order == time order."""
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _item_params(item: TrendItem) -> tuple[Any, ...]:
    status = ItemStatus.ANALYZED if item.is_analyzed else ItemStatus.NEW
    return (
        item.id, item.title, item.url, item.source_name, item.source_type.value, item.content,
        item.author, _dt_to_str(item.published_at), json.dumps(list(item.tags), ensure_ascii=False),
        _dt_to_str(item.collected_at), item.title_key, item.relevance_score, item.ai_summary, status.value,
    )


def _row_to_item(row: sqlite3.Row) -> TrendItem:
    # model_validate re-runs normalisation, so a corrupted row fails loudly here
    # instead of leaking bad data into the pipeline.
    return TrendItem.model_validate(
        {
            "id": row["id"], "title": row["title"], "url": row["url"],
            "source_name": row["source_name"], "source_type": row["source_type"],
            "content": row["content"], "author": row["author"], "published_at": row["published_at"],
            "tags": json.loads(row["tags"]), "collected_at": row["collected_at"],
            "relevance_score": row["relevance_score"], "ai_summary": row["ai_summary"],
        }
    )


def _chunks(values: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


# ----------------------------------------------------------------------
# Repository
# ----------------------------------------------------------------------
class TrendRepository:
    """SQLite-backed store for TrendItems and their lifecycle status."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._lock = threading.RLock()
        path = str(db_path)
        if path != ":memory:":
            Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
            self._conn.execute("PRAGMA busy_timeout=5000")  # wait up to 5 s on a locked DB
            self._conn.execute("PRAGMA foreign_keys=ON")  # no FKs yet; cheap insurance for later tables
            migrate(self._conn)
        except Exception:
            self._conn.close()
            raise

    # ---- lifecycle ----
    def close(self) -> None:
        with self._lock:
            self._conn.close()  # idempotent

    def __enter__(self) -> TrendRepository:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- writes ----
    def add_new(self, items: Iterable[TrendItem]) -> list[TrendItem]:
        """Insert items whose id is unknown; return ONLY those actually inserted (input order).

        Already-known ids (any status, including SENT/REJECTED) are silently ignored,
        as are repeats inside `items` itself. All-or-nothing: one transaction.
        """
        inserted: list[TrendItem] = []
        with self._lock, self._conn:  # `with conn` = commit on success, rollback on error
            for item in items:
                if self._conn.execute(_INSERT_SQL, _item_params(item)).rowcount == 1:
                    inserted.append(item)
        return inserted

    def update_analysis(self, item: TrendItem) -> bool:
        """Persist Gemini's score + summary (NEW/ANALYZED -> ANALYZED). False if unknown or already terminal."""
        if not item.is_analyzed:
            raise ValueError("item has no analysis; call item.with_analysis(...) first")
        return self._transition(
            "UPDATE trend_items SET relevance_score=?, ai_summary=?, status=? "
            "WHERE id=? AND status IN ('new', 'analyzed')",
            (item.relevance_score, item.ai_summary, ItemStatus.ANALYZED.value, item.id),
        )

    def mark_rejected(self, item_id: str, reason: str | None = None) -> bool:
        """NEW/ANALYZED/SENDING -> REJECTED (kept forever so it is never reprocessed). Analysis is preserved."""
        return self._transition(
            "UPDATE trend_items SET status=?, status_note=?, claimed_at=NULL "
            "WHERE id=? AND status IN ('new', 'analyzed', 'sending')",
            (ItemStatus.REJECTED.value, reason, item_id),
        )

    def mark_sent(self, item_id: str, when: datetime | None = None) -> bool:
        """ANALYZED or SENDING -> SENT. Only analysed items can be sent, and only once."""
        return self._transition(
            "UPDATE trend_items SET status=?, sent_at=?, claimed_at=NULL "
            "WHERE id=? AND status IN ('analyzed', 'sending')",
            (ItemStatus.SENT.value, _dt_to_str(when or datetime.now(timezone.utc)), item_id),
        )

    def claim_for_delivery(self, item_id: str, when: datetime | None = None) -> bool:
        """ANALYZED -> SENDING, atomically. True = THIS caller owns the send; False = someone else does / not eligible."""
        return self._transition(
            "UPDATE trend_items SET status=?, claimed_at=? WHERE id=? AND status='analyzed'",
            (ItemStatus.SENDING.value, _dt_to_str(when or datetime.now(timezone.utc)), item_id),
        )

    def release_claim(self, item_id: str) -> bool:
        """SENDING -> ANALYZED. ONLY for cases where the message provably was NOT sent (request rejected/never left)."""
        return self._transition(
            "UPDATE trend_items SET status=?, claimed_at=NULL WHERE id=? AND status='sending'",
            (ItemStatus.ANALYZED.value, item_id),
        )

    def recover_interrupted_deliveries(
        self, older_than: timedelta = timedelta(minutes=10), now: datetime | None = None
    ) -> list[str]:
        """Claims left behind by a crashed run -> REJECTED ('outcome unknown, not re-sent'). Returns their ids.

        `older_than` protects a healthy run that is still sending right now (claims are only minutes old).
        """
        cutoff = _dt_to_str((now or datetime.now(timezone.utc)) - older_than)
        with self._lock, self._conn:
            ids = [
                r["id"]
                for r in self._conn.execute(
                    "SELECT id FROM trend_items WHERE status='sending' AND claimed_at <= ?", (cutoff,)
                )
            ]
            for item_id in ids:
                self._conn.execute(
                    "UPDATE trend_items SET status=?, status_note=?, claimed_at=NULL WHERE id=?",
                    (ItemStatus.REJECTED.value, "delivery interrupted: outcome unknown, not re-sent to avoid a duplicate post", item_id),
                )
        return ids

    def _transition(self, sql: str, params: tuple[Any, ...]) -> bool:
        with self._lock, self._conn:
            return self._conn.execute(sql, params).rowcount == 1

    # ---- dedup primitives (policy lives in Module 7) ----
    def existing_ids(self, ids: Iterable[str]) -> set[str]:
        """Subset of `ids` already stored, in ANY status."""
        unique = list(dict.fromkeys(ids))
        found: set[str] = set()
        with self._lock:
            for chunk in _chunks(unique, _IN_CHUNK):
                rows = self._conn.execute(
                    f"SELECT id FROM trend_items WHERE id IN ({_placeholders(len(chunk))})", chunk
                )
                found.update(r["id"] for r in rows)
        return found

    def existing_title_keys(self, keys: Iterable[str], since: datetime | None = None) -> set[str]:
        """Subset of normalised titles already stored (optionally only rows collected at/after `since`).

        Empty keys (titles made only of symbols) are ignored: they would match everything.
        """
        unique = [k for k in dict.fromkeys(keys) if k]
        found: set[str] = set()
        since_sql, since_params = ("", [])
        if since is not None:
            since_sql, since_params = (" AND collected_at >= ?", [_dt_to_str(since)])
        with self._lock:
            for chunk in _chunks(unique, _IN_CHUNK):
                rows = self._conn.execute(
                    f"SELECT DISTINCT title_key FROM trend_items "
                    f"WHERE title_key IN ({_placeholders(len(chunk))}){since_sql}",
                    [*chunk, *since_params],
                )
                found.update(r["title_key"] for r in rows)
        return found

    # ---- reads ----
    def get(self, item_id: str) -> TrendItem | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM trend_items WHERE id=?", (item_id,)).fetchone()
        return _row_to_item(row) if row else None

    def list_by_status(self, status: ItemStatus, limit: int | None = None) -> list[TrendItem]:
        """Items in `status`, most recently collected first."""
        sql = "SELECT * FROM trend_items WHERE status=? ORDER BY collected_at DESC, id"
        params: list[Any] = [status.value]
        if limit is not None:
            sql, params = sql + " LIMIT ?", [*params, limit]
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_item(r) for r in rows]

    def pending_delivery(self, min_score: int, limit: int) -> list[TrendItem]:
        """Best ANALYZED items scoring >= min_score: highest score first, then newest. Capped by `limit`.

        `limit` is the per-run message cap (MAX_MESSAGES_PER_RUN), enforced where items are chosen.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM trend_items WHERE status='analyzed' AND relevance_score >= ? "
                "ORDER BY relevance_score DESC, COALESCE(published_at, collected_at) DESC, id LIMIT ?",
                (min_score, limit),
            ).fetchall()
        return [_row_to_item(r) for r in rows]

    def count_by_status(self) -> dict[str, int]:
        counts = {s.value: 0 for s in ItemStatus}  # zeros included -> stable shape for logs/tests
        with self._lock:
            for row in self._conn.execute("SELECT status, COUNT(*) AS n FROM trend_items GROUP BY status"):
                counts[row["status"]] = row["n"]
        return counts


if __name__ == "__main__":
    # Inspect (or initialise) a database:  python -m app.database.repository [path]
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/trendwatch.db"
    with TrendRepository(target) as repo:
        print(f"{target}: {json.dumps(repo.count_by_status())}")
