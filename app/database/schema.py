"""
FILE: app/database/schema.py
MODULE 6 — SQLite schema, item lifecycle and migrations.

ARCHITECTURE NOTES
  * Pure persistence vocabulary: no imports from other `app.*` packages.
  * ITEM LIFECYCLE (the reason we store EVERY seen item, not just the sent ones):

        NEW ──analysis──► ANALYZED ──claim──► SENDING ──confirmed──► SENT
         │                   │  ▲                │
         │                   │  └── released ────┤ (request provably not sent)
         └────── rejected ───┴───── rejected ────┘   REJECTED rows are KEPT

    SENDING is the crash-safety state (Module 15): an item is CLAIMED before its message is sent.
    If the process dies between "sent" and "marked sent", the row is still SENDING on the next run
    and is treated as delivery-UNKNOWN (rejected, never re-sent): at-most-once, no duplicate posts.

    Remembering rejected/irrelevant items means the next run never re-spends
    scarce Gemini free-tier quota on them, and SENT rows guarantee an item is
    never posted to the class group twice.
  * MIGRATIONS via `PRAGMA user_version`: an integer stored inside the DB file.
    MIGRATIONS[i] upgrades version i -> i+1. Rules: append only, NEVER edit an
    applied migration (real databases already ran it), hence the status literals
    are hard-coded in the DDL instead of being generated from `ItemStatus`.
  * Timestamps are stored as fixed-width ISO-8601 UTC text, which sorts
    chronologically with plain string comparison.
  * `tags` is a JSON array in a TEXT column: the app never queries by tag, so a
    join table would be needless complexity for this MVP.
"""
from __future__ import annotations

import sqlite3
from enum import Enum


class DatabaseError(RuntimeError):
    """Database is unusable (e.g. created by a newer version of the app)."""


class ItemStatus(str, Enum):
    NEW = "new"  # stored, awaiting Gemini analysis
    ANALYZED = "analyzed"  # score + summary saved, awaiting delivery decision
    SENDING = "sending"  # claimed for delivery; outcome not yet known
    SENT = "sent"  # delivered to WhatsApp (terminal)
    REJECTED = "rejected"  # dropped (irrelevant / invalid); kept so it is never reprocessed (terminal)


MIGRATIONS: tuple[str, ...] = (
    # ---- v1: initial schema ----
    """
    CREATE TABLE trend_items (
        id              TEXT PRIMARY KEY,
        title           TEXT NOT NULL,
        url             TEXT NOT NULL,
        source_name     TEXT NOT NULL,
        source_type     TEXT NOT NULL,
        content         TEXT NOT NULL DEFAULT '',
        author          TEXT,
        published_at    TEXT,
        tags            TEXT NOT NULL DEFAULT '[]',
        collected_at    TEXT NOT NULL,
        title_key       TEXT NOT NULL,
        relevance_score INTEGER CHECK (relevance_score IS NULL OR relevance_score BETWEEN 0 AND 10),
        ai_summary      TEXT,
        status          TEXT NOT NULL DEFAULT 'new'
                        CHECK (status IN ('new', 'analyzed', 'sent', 'rejected')),
        status_note     TEXT,
        sent_at         TEXT
    );
    CREATE INDEX idx_items_status    ON trend_items (status);
    CREATE INDEX idx_items_title_key ON trend_items (title_key);
    CREATE INDEX idx_items_collected ON trend_items (collected_at);
    """,
    # ---- v2 (Module 15): add the 'sending' status + claimed_at.
    # SQLite cannot alter a CHECK constraint, so the table is rebuilt (copy -> drop -> rename).
    """
    CREATE TABLE trend_items_v2 (
        id              TEXT PRIMARY KEY,
        title           TEXT NOT NULL,
        url             TEXT NOT NULL,
        source_name     TEXT NOT NULL,
        source_type     TEXT NOT NULL,
        content         TEXT NOT NULL DEFAULT '',
        author          TEXT,
        published_at    TEXT,
        tags            TEXT NOT NULL DEFAULT '[]',
        collected_at    TEXT NOT NULL,
        title_key       TEXT NOT NULL,
        relevance_score INTEGER CHECK (relevance_score IS NULL OR relevance_score BETWEEN 0 AND 10),
        ai_summary      TEXT,
        status          TEXT NOT NULL DEFAULT 'new'
                        CHECK (status IN ('new', 'analyzed', 'sending', 'sent', 'rejected')),
        status_note     TEXT,
        sent_at         TEXT,
        claimed_at      TEXT
    );
    INSERT INTO trend_items_v2
        (id, title, url, source_name, source_type, content, author, published_at, tags,
         collected_at, title_key, relevance_score, ai_summary, status, status_note, sent_at)
    SELECT id, title, url, source_name, source_type, content, author, published_at, tags,
           collected_at, title_key, relevance_score, ai_summary, status, status_note, sent_at
    FROM trend_items;
    DROP TABLE trend_items;
    ALTER TABLE trend_items_v2 RENAME TO trend_items;
    CREATE INDEX idx_items_status    ON trend_items (status);
    CREATE INDEX idx_items_title_key ON trend_items (title_key);
    CREATE INDEX idx_items_collected ON trend_items (collected_at);
    """,
)
SCHEMA_VERSION = len(MIGRATIONS)


def migrate(conn: sqlite3.Connection) -> int:
    """Bring `conn`'s database up to SCHEMA_VERSION. Each step is one atomic transaction.

    Returns the resulting version. Raises DatabaseError if the file is NEWER than this code.
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > SCHEMA_VERSION:
        raise DatabaseError(
            f"database schema v{current} is newer than this application supports (v{SCHEMA_VERSION})"
        )
    for version in range(current, SCHEMA_VERSION):
        # executescript() commits any open transaction first, so the BEGIN/COMMIT inside
        # the script is what makes "DDL + version bump" all-or-nothing.
        script = f"BEGIN;\n{MIGRATIONS[version]}\nPRAGMA user_version = {version + 1};\nCOMMIT;"
        try:
            conn.executescript(script)
        except sqlite3.Error:
            conn.rollback()
            raise
    return SCHEMA_VERSION
