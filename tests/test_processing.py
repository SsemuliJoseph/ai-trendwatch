"""
FILE: tests/test_processing.py
MODULE 7 tests — run with:  pytest tests/test_processing.py
Validation is tested with a fixed clock; dedup with a fake store (proves the Protocol
decoupling) and once end-to-end against the real in-memory TrendRepository.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.database.repository import TrendRepository
from app.models import SourceType, TrendItem
from app.processing.deduplication import Deduplicator, DuplicateReason
from app.processing.validation import RejectReason, ValidationRules, check_item, validate_items

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def make(n: int = 1, **kw) -> TrendItem:
    fields = dict(
        title=f"Something notable happened number {n}",
        url=f"https://example.com/{n}",
        source_name="Src",
        source_type=SourceType.RSS,
        published_at=NOW - timedelta(hours=2),
        collected_at=NOW,
    )
    return TrendItem(**{**fields, **kw})


# =====================================================================
# Validation
# =====================================================================
RULES = ValidationRules()


def test_good_item_passes():
    assert check_item(make(), RULES, NOW) is None


def test_title_too_short():
    assert check_item(make(title="Short"), RULES, NOW) is RejectReason.TITLE_TOO_SHORT
    assert check_item(make(title="Exactly 10"), RULES, NOW) is None  # boundary: len == 10 passes


def test_age_boundaries():
    assert check_item(make(published_at=NOW - timedelta(days=7)), RULES, NOW) is None
    assert check_item(make(published_at=NOW - timedelta(days=7, seconds=1)), RULES, NOW) is RejectReason.TOO_OLD


def test_future_dates_tolerated_within_window_but_rejected_beyond():
    assert check_item(make(published_at=NOW + timedelta(hours=24)), RULES, NOW) is None
    assert check_item(make(published_at=NOW + timedelta(hours=25)), RULES, NOW) is RejectReason.FUTURE_DATE


def test_undated_policy_is_configurable():
    undated = make(published_at=None)
    assert check_item(undated, RULES, NOW) is None
    strict = ValidationRules(allow_undated=False)
    assert check_item(undated, strict, NOW) is RejectReason.MISSING_DATE


def test_blocked_domain_matches_host_and_subdomains_but_not_lookalikes():
    rules = ValidationRules(blocked_domains=frozenset({"spam.example"}))
    assert check_item(make(url="https://spam.example/a"), rules, NOW) is RejectReason.BLOCKED_DOMAIN
    assert check_item(make(url="https://blog.spam.example/a"), rules, NOW) is RejectReason.BLOCKED_DOMAIN
    assert check_item(make(url="https://notspam.example/a"), rules, NOW) is None
    assert check_item(make(url="https://spam.example.org/a"), rules, NOW) is None


def test_first_failing_rule_wins():
    both = make(title="Short", published_at=NOW - timedelta(days=30))  # too short AND too old
    assert check_item(both, RULES, NOW) is RejectReason.TITLE_TOO_SHORT
    blocked = make(title="Short", url="https://spam.example/x")
    rules = ValidationRules(blocked_domains=frozenset({"spam.example"}))
    assert check_item(blocked, rules, NOW) is RejectReason.BLOCKED_DOMAIN


def test_validate_items_splits_preserves_order_and_counts_reasons():
    a, b, c, d = make(1), make(2, published_at=NOW - timedelta(days=9)), make(3), make(4, title="Tiny")
    result = validate_items([a, b, c, d], RULES, now=NOW)
    assert result.valid == [a, c]
    assert [(r.item, r.reason) for r in result.rejected] == [
        (b, RejectReason.TOO_OLD),
        (d, RejectReason.TITLE_TOO_SHORT),
    ]
    assert result.reason_counts() == {"too_old": 1, "title_too_short": 1}


def test_validate_items_accepts_generators_and_empty_input():
    assert validate_items(iter([]), RULES, now=NOW).valid == []
    assert validate_items((make(i) for i in range(3)), RULES, now=NOW).valid == [make(i) for i in range(3)]


# =====================================================================
# Deduplication (fake store)
# =====================================================================
class FakeStore:
    """Structural SeenStore: no inheritance, no SQLite."""

    def __init__(self, ids=(), titles=()):
        self.ids, self.titles = set(ids), set(titles)
        self.since_seen = None
        self.calls = 0

    def existing_ids(self, ids):
        self.calls += 1
        return self.ids & set(ids)

    def existing_title_keys(self, keys, since=None):
        self.calls += 1
        self.since_seen = since
        return self.titles & set(keys)


def reasons(result):
    return [(d.item, d.reason) for d in result.duplicates]


def test_everything_unique_against_empty_history():
    items = [make(i) for i in range(3)]
    result = Deduplicator(FakeStore()).filter_new(items, now=NOW)
    assert result.unique == items and result.duplicates == []


def test_same_id_twice_in_batch_keeps_first():
    a1, a2 = make(1), make(1, title="A different title for the very same url")
    result = Deduplicator(FakeStore()).filter_new([a1, a2], now=NOW)
    assert result.unique == [a1]
    assert reasons(result) == [(a2, DuplicateReason.DUPLICATE_IN_BATCH)]


def test_same_story_different_urls_in_batch():
    a = make(1, title="OpenAI Releases GPT-6 Today!")
    b = make(2, title="openai releases   gpt 6 today")
    result = Deduplicator(FakeStore()).filter_new([a, b], now=NOW)
    assert result.unique == [a]
    assert reasons(result) == [(b, DuplicateReason.SAME_TITLE_IN_BATCH)]


def test_already_seen_id_beats_title_match():
    item = make(1)
    store = FakeStore(ids={item.id}, titles={item.title_key})
    result = Deduplicator(store).filter_new([item], now=NOW)
    assert reasons(result) == [(item, DuplicateReason.ALREADY_SEEN)]


def test_same_story_seen_earlier_under_another_url():
    item = make(1)
    result = Deduplicator(FakeStore(titles={item.title_key})).filter_new([item], now=NOW)
    assert result.unique == [] and reasons(result) == [(item, DuplicateReason.SAME_TITLE_SEEN)]


def test_generic_short_titles_are_never_title_deduplicated():
    a, b = make(1, title="Weekly roundup"), make(2, title="Weekly roundup")  # 2 words < 3
    result = Deduplicator(FakeStore(titles={a.title_key})).filter_new([a, b], now=NOW)
    assert result.unique == [a, b] and result.duplicates == []


def test_min_title_words_is_configurable():
    a, b = make(1, title="Weekly roundup"), make(2, title="Weekly roundup")
    result = Deduplicator(FakeStore(), min_title_words=2).filter_new([a, b], now=NOW)
    assert reasons(result) == [(b, DuplicateReason.SAME_TITLE_IN_BATCH)]


def test_title_lookback_window_is_passed_to_the_store():
    store = FakeStore()
    Deduplicator(store, title_window_days=10).filter_new([make(1)], now=NOW)
    assert store.since_seen == NOW - timedelta(days=10)


def test_history_is_queried_in_two_batched_calls_regardless_of_batch_size():
    store = FakeStore()
    Deduplicator(store).filter_new([make(i) for i in range(50)], now=NOW)
    assert store.calls == 2  # existing_ids + existing_title_keys, not 100 queries


def test_result_order_and_reason_counts():
    a, b, c = make(1), make(2), make(3)
    store = FakeStore(ids={b.id})
    result = Deduplicator(store).filter_new([a, b, c, make(1)], now=NOW)
    assert result.unique == [a, c]
    assert result.reason_counts() == {"duplicate_in_batch": 1, "already_seen": 1}


@pytest.mark.parametrize("kw", [dict(title_window_days=0), dict(min_title_words=0)])
def test_invalid_parameters_rejected(kw):
    with pytest.raises(ValueError):
        Deduplicator(FakeStore(), **kw)


# =====================================================================
# Integration: validate -> dedup -> persist, against the real repository
# =====================================================================
def test_pipeline_across_runs_with_real_repository():
    with TrendRepository(":memory:") as repo:
        dedup = Deduplicator(repo)  # TrendRepository satisfies SeenStore structurally

        def run(items):
            valid = validate_items(items, RULES, now=NOW).valid
            result = dedup.filter_new(valid, now=NOW)
            return repo.add_new(result.unique), result

        original = make(1, title="Anthropic ships a brand new model")
        repost = make(2, title="ANTHROPIC ships a brand-new model!", url="https://other.example/x")
        stale = make(3, published_at=NOW - timedelta(days=30))

        # Run 1: repost dropped as same story, stale dropped by validation.
        stored, result = run([original, repost, stale])
        assert stored == [original]
        assert reasons(result) == [(repost, DuplicateReason.SAME_TITLE_IN_BATCH)]

        # Run 2 (next hour, same feeds): nothing new. The pair is still in one batch, so the
        # repost is caught by the in-batch pass before history is even consulted.
        stored, result = run([original, repost])
        assert stored == []
        assert result.reason_counts() == {"already_seen": 1, "same_title_in_batch": 1}

        # The repost arriving ALONE later (other source publishes next day) hits history instead.
        stored, result = run([repost])
        assert stored == []
        assert reasons(result) == [(repost, DuplicateReason.SAME_TITLE_SEEN)]

        # Sent/rejected items still count as "seen" (Module 6 keeps them).
        repo.update_analysis(original.with_analysis(9, "s"))
        repo.mark_sent(original.id)
        stored, _ = run([original])
        assert stored == []


def test_old_history_outside_the_window_does_not_block_a_reused_title():
    with TrendRepository(":memory:") as repo:
        old = make(1, title="State of AI report is out", collected_at=NOW - timedelta(days=30))
        repo.add_new([old])
        fresh = make(2, title="State of AI report is out")  # same title, new URL, 30 days later
        result = Deduplicator(repo, title_window_days=14).filter_new([fresh], now=NOW)
        assert result.unique == [fresh]
