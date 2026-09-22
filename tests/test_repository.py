"""
FILE: tests/test_repository.py
MODULE 6 tests — run with:  pytest tests/test_repository.py
Uses in-memory SQLite (and tmp_path files where persistence matters): no network, no leftovers.
"""
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from app.database.repository import DatabaseError, ItemStatus, TrendRepository
from app.database.schema import SCHEMA_VERSION
from app.models import SourceType, TrendItem

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def make(n: int = 1, **kw) -> TrendItem:
    fields = dict(
        title=f"Item {n}", url=f"https://example.com/{n}", source_name="Src",
        source_type=SourceType.RSS, collected_at=NOW,
    )
    return TrendItem(**{**fields, **kw})


@pytest.fixture
def repo():
    with TrendRepository(":memory:") as r:
        yield r


# ---------- setup / schema ----------
def test_creates_missing_parent_dirs_sets_wal_and_schema_version(tmp_path):
    path = tmp_path / "nested" / "dir" / "t.db"
    with TrendRepository(path) as r:
        assert r._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert r._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert path.exists()


def test_reopening_preserves_data_and_does_not_remigrate(tmp_path):
    path = tmp_path / "t.db"
    with TrendRepository(path) as r:
        r.add_new([make(1)])
    with TrendRepository(path) as r:
        assert r.get(make(1).id) is not None


def test_database_from_newer_app_version_is_refused(tmp_path):
    path = tmp_path / "future.db"
    raw = sqlite3.connect(path)
    raw.execute("PRAGMA user_version = 99")
    raw.close()
    with pytest.raises(DatabaseError, match="newer"):
        TrendRepository(path)


def test_schema_rejects_invalid_status_and_score(repo):
    base = "INSERT INTO trend_items (id,title,url,source_name,source_type,collected_at,title_key,{}) VALUES ('x','t','u','s','rss','c','k',{})"
    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute(base.format("status", "'bogus'"))
    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute(base.format("relevance_score", "11"))


def test_use_after_close_raises():
    r = TrendRepository(":memory:")
    r.close()
    with pytest.raises(sqlite3.ProgrammingError):
        r.count_by_status()


# ---------- add_new / round-trip ----------
def test_add_new_returns_only_newly_inserted_items(repo):
    a, b = make(1), make(2)
    assert repo.add_new([a, b]) == [a, b]
    assert repo.add_new([a, make(3)]) == [make(3)]  # `a` already known
    assert repo.count_by_status()["new"] == 3


def test_duplicates_inside_one_batch_are_inserted_once(repo):
    same = [make(1), make(1), make(1)]
    assert len(repo.add_new(same)) == 1
    assert repo.count_by_status()["new"] == 1


def test_get_roundtrip_is_lossless(repo):
    item = make(
        1, content="Body", author="Ada", tags=["ai", "llm"],
        published_at=datetime(2026, 9, 18, 8, 30, 15, 123456, tzinfo=timezone.utc),
    ).with_analysis(8, "Two\nline summary")
    repo.add_new([item])
    assert repo.get(item.id) == item


def test_get_unknown_returns_none(repo):
    assert repo.get("nope") is None


def test_pre_analysed_item_is_stored_as_analyzed(repo):
    repo.add_new([make(1).with_analysis(5, "s")])
    assert repo.count_by_status()["analyzed"] == 1


# ---------- dedup primitives ----------
def test_existing_ids_sees_all_statuses(repo):
    items = [make(i) for i in range(1, 5)]
    repo.add_new(items)
    repo.update_analysis(items[1].with_analysis(9, "s"))
    repo.mark_sent(items[1].id)
    repo.mark_rejected(items[2].id)
    known = repo.existing_ids([i.id for i in items] + ["unknown"])
    assert known == {i.id for i in items}  # NEW, SENT, REJECTED all count as "seen"


def test_existing_ids_handles_more_than_one_chunk(repo):
    items = [make(i) for i in range(1200)]
    repo.add_new(items)
    assert len(repo.existing_ids(i.id for i in items)) == 1200


def test_existing_title_keys_with_since_window_and_empty_keys(repo):
    old = make(1, title="Old Story!", collected_at=NOW - timedelta(days=30))
    new = make(2, title="Fresh story", collected_at=NOW)
    repo.add_new([old, new])
    keys = [old.title_key, new.title_key, ""]
    assert repo.existing_title_keys(keys) == {"old story", "fresh story"}
    assert repo.existing_title_keys(keys, since=NOW - timedelta(days=7)) == {"fresh story"}


# ---------- lifecycle transitions ----------
def test_update_analysis_moves_new_to_analyzed(repo):
    item = make(1)
    repo.add_new([item])
    assert repo.update_analysis(item.with_analysis(7, "Summary")) is True
    stored = repo.get(item.id)
    assert (stored.relevance_score, stored.ai_summary) == (7, "Summary")
    assert repo.count_by_status()["analyzed"] == 1


def test_update_analysis_requires_analysis_and_known_item(repo):
    with pytest.raises(ValueError):
        repo.update_analysis(make(1))
    assert repo.update_analysis(make(9).with_analysis(5, "s")) is False  # not stored


def test_mark_sent_only_from_analyzed_and_only_once(repo):
    item = make(1)
    repo.add_new([item])
    assert repo.mark_sent(item.id) is False  # NEW can't be sent
    repo.update_analysis(item.with_analysis(9, "s"))
    assert repo.mark_sent(item.id, when=NOW) is True
    assert repo.mark_sent(item.id) is False  # never twice
    assert repo.update_analysis(item.with_analysis(1, "again")) is False  # terminal: no re-analysis
    assert repo.get(item.id).relevance_score == 9
    assert repo._conn.execute("SELECT sent_at FROM trend_items").fetchone()[0].startswith("2026-09-19T12:00:00")


def test_mark_rejected_keeps_row_and_analysis_and_is_terminal(repo):
    item = make(1).with_analysis(2, "meh")
    repo.add_new([item])
    assert repo.mark_rejected(item.id, "score 2 < threshold") is True
    assert repo.get(item.id).relevance_score == 2
    assert repo.mark_rejected(item.id) is False
    assert repo.mark_sent(item.id) is False
    assert repo.existing_ids([item.id]) == {item.id}


# ---------- delivery queries ----------
def test_pending_delivery_filters_orders_and_limits(repo):
    a9 = make(1, published_at=NOW - timedelta(hours=5)).with_analysis(9, "s")
    b9 = make(2, published_at=NOW - timedelta(hours=1)).with_analysis(9, "s")  # newer -> before a9
    c7 = make(3).with_analysis(7, "s")
    d3 = make(4).with_analysis(3, "s")  # below threshold
    e10 = make(5).with_analysis(10, "s")
    sent = make(6).with_analysis(10, "s")
    rejected = make(7).with_analysis(10, "s")
    unanalysed = make(8)
    repo.add_new([a9, b9, c7, d3, e10, sent, rejected, unanalysed])
    repo.mark_sent(sent.id)
    repo.mark_rejected(rejected.id)

    got = repo.pending_delivery(min_score=6, limit=10)
    assert [i.id for i in got] == [e10.id, b9.id, a9.id, c7.id]
    assert [i.id for i in repo.pending_delivery(min_score=6, limit=2)] == [e10.id, b9.id]


def test_list_by_status_newest_collected_first_with_limit(repo):
    items = [make(i, collected_at=NOW + timedelta(minutes=i)) for i in range(1, 4)]
    repo.add_new(items)
    assert [i.title for i in repo.list_by_status(ItemStatus.NEW)] == ["Item 3", "Item 2", "Item 1"]
    assert len(repo.list_by_status(ItemStatus.NEW, limit=2)) == 2
    assert repo.list_by_status(ItemStatus.SENT) == []


def test_count_by_status_always_lists_every_status(repo):
    assert repo.count_by_status() == {"new": 0, "analyzed": 0, "sending": 0, "sent": 0, "rejected": 0}


# ---------- safety ----------
def test_sql_metacharacters_in_data_are_stored_literally(repo):
    evil = make(1, title="Robert'); DROP TABLE trend_items;--", tags=["a'b", 'c"d'])
    repo.add_new([evil])
    assert repo.get(evil.id) == evil
    assert repo.count_by_status()["new"] == 1


def test_concurrent_writers_from_threads(repo):
    items = [make(i) for i in range(200)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda chunk: repo.add_new(chunk), [items[i::8] for i in range(8)]))
    assert sum(len(r) for r in results) == 200
    assert repo.count_by_status()["new"] == 200


# =====================================================================
# Module 15: claim protocol (crash-safe, at-most-once delivery)
# =====================================================================
def analysed(repo, n=1):
    item = make(n).with_analysis(8, "s")
    repo.add_new([item])
    return item


def status_of(repo, item_id):
    return repo._conn.execute("SELECT status FROM trend_items WHERE id=?", (item_id,)).fetchone()[0]


def test_claim_is_exclusive_and_only_for_analysed_items(repo):
    item = analysed(repo)
    assert repo.claim_for_delivery(item.id) is True
    assert repo.claim_for_delivery(item.id) is False  # already owned by someone
    assert status_of(repo, item.id) == "sending"

    fresh = make(2)
    repo.add_new([fresh])
    assert repo.claim_for_delivery(fresh.id) is False  # NEW: not reviewed yet
    assert repo.claim_for_delivery("unknown") is False


def test_a_claimed_item_disappears_from_pending_delivery(repo):
    a, b = analysed(repo, 1), analysed(repo, 2)
    repo.claim_for_delivery(a.id)
    assert [i.id for i in repo.pending_delivery(min_score=0, limit=10)] == [b.id]


def test_release_claim_returns_the_item_to_the_queue(repo):
    item = analysed(repo)
    assert repo.release_claim(item.id) is False  # nothing to release yet
    repo.claim_for_delivery(item.id)
    assert repo.release_claim(item.id) is True
    assert [i.id for i in repo.pending_delivery(0, 10)] == [item.id]
    assert repo._conn.execute("SELECT claimed_at FROM trend_items").fetchone()[0] is None


def test_mark_sent_and_mark_rejected_accept_a_claimed_item(repo):
    a, b = analysed(repo, 1), analysed(repo, 2)
    repo.claim_for_delivery(a.id)
    repo.claim_for_delivery(b.id)
    assert repo.mark_sent(a.id, when=NOW) is True
    assert repo.mark_rejected(b.id, "delivery unknown") is True
    assert repo.count_by_status() == {"new": 0, "analyzed": 0, "sending": 0, "sent": 1, "rejected": 1}
    assert repo._conn.execute("SELECT COUNT(*) FROM trend_items WHERE claimed_at IS NOT NULL").fetchone()[0] == 0


def test_a_claimed_item_cannot_be_re_analysed(repo):
    item = analysed(repo)
    repo.claim_for_delivery(item.id)
    assert repo.update_analysis(item.with_analysis(1, "changed")) is False


def test_only_stale_claims_are_recovered_as_interrupted(repo):
    old, recent, idle = analysed(repo, 1), analysed(repo, 2), analysed(repo, 3)
    repo.claim_for_delivery(old.id, when=NOW - timedelta(hours=1))
    repo.claim_for_delivery(recent.id, when=NOW - timedelta(minutes=1))

    recovered = repo.recover_interrupted_deliveries(older_than=timedelta(minutes=10), now=NOW)

    assert recovered == [old.id]
    assert status_of(repo, old.id) == "rejected" and status_of(repo, recent.id) == "sending"
    assert status_of(repo, idle.id) == "analyzed"  # never touched
    note = repo._conn.execute("SELECT status_note FROM trend_items WHERE id=?", (old.id,)).fetchone()[0]
    assert "interrupted" in note and "not re-sent" in note
    assert repo.recover_interrupted_deliveries(older_than=timedelta(minutes=10), now=NOW) == []  # idempotent


def test_exactly_one_of_many_concurrent_claimers_wins(repo):
    item = analysed(repo)
    with ThreadPoolExecutor(max_workers=8) as pool:
        wins = list(pool.map(lambda _: repo.claim_for_delivery(item.id), range(16)))
    assert wins.count(True) == 1


# =====================================================================
# Module 15: migration v1 -> v2 (table rebuild)
# =====================================================================
def test_a_v1_database_is_upgraded_in_place_without_losing_rows(tmp_path):
    from app.database.schema import MIGRATIONS

    path = tmp_path / "old.db"
    raw = sqlite3.connect(path)
    raw.executescript(MIGRATIONS[0] + "PRAGMA user_version = 1;")  # exactly what a Module 6 install looks like
    rows = [("a", "new"), ("b", "analyzed"), ("c", "sent"), ("d", "rejected")]
    for item_id, status in rows:
        raw.execute(
            "INSERT INTO trend_items (id,title,url,source_name,source_type,collected_at,title_key,status,status_note) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (item_id, f"Title {item_id}", f"https://x.test/{item_id}", "S", "rss", "2026-09-19T00:00:00.000000+00:00",
             f"title {item_id}", status, "kept" if status == "rejected" else None),
        )
    raw.commit()
    raw.close()

    with TrendRepository(path) as repo:
        assert repo._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 2
        assert repo.count_by_status() == {"new": 1, "analyzed": 1, "sending": 0, "sent": 1, "rejected": 1}
        assert repo.get("a").title == "Title a"
        assert repo._conn.execute("SELECT status_note FROM trend_items WHERE id='d'").fetchone()[0] == "kept"
        columns = [r[1] for r in repo._conn.execute("PRAGMA table_info(trend_items)")]
        assert "claimed_at" in columns
        indexes = {r[1] for r in repo._conn.execute("PRAGMA index_list(trend_items)")}
        assert {"idx_items_status", "idx_items_title_key", "idx_items_collected"} <= indexes
        repo._conn.execute("UPDATE trend_items SET status='sending' WHERE id='b'")  # the new status is now legal
        repo._conn.commit()  # (raw statements are not auto-committed; without this the reopen check would see nothing)
        with pytest.raises(sqlite3.IntegrityError):
            repo._conn.execute("UPDATE trend_items SET status='bogus' WHERE id='b'")  # ...and the CHECK still guards

    with TrendRepository(path) as repo:  # reopening a v2 file changes nothing
        assert repo.count_by_status()["sending"] == 1
