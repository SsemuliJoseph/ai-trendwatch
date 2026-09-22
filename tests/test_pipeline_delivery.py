"""
FILE: tests/test_pipeline_delivery.py
MODULE 14 tests — run with:  pytest tests/test_pipeline_delivery.py
The full graph runs for real (LangGraph + real repository + real validation/dedup/formatter).
Only the outside world is faked: collectors, the analyzer and the gateway.
"""
import logging
from datetime import datetime, timedelta, timezone

import pytest

from app.agents.workflow import TrendNodes, WorkflowConfig, WorkflowDeps, build_workflow, run_once
from app.ai.gemini_service import AnalysisRun
from app.database.repository import ItemStatus, TrendRepository
from app.messaging.formatter import format_item
from app.messaging.openwa_client import (
    DryRunClient,
    OpenWAAuthError,
    OpenWADeliveryUnknownError,
    OpenWARejectedError,
    OpenWASessionError,
    OpenWAUnavailableError,
    SendResult,
)
from app.models import SourceType, TrendItem
from app.processing.validation import ValidationRules

GROUP = "120363000000000000@g.us"


def make(n: int, **kw) -> TrendItem:
    fields = dict(
        title=f"Notable AI story number {n}", url=f"https://example.com/{n}", source_name="Src",
        source_type=SourceType.RSS, published_at=datetime.now(timezone.utc) - timedelta(hours=1 + n * 0.01),
    )
    return TrendItem(**{**fields, **kw})


class FakeCollector:
    def __init__(self, items=()):
        self._items, self.failures = list(items), {}

    def collect(self):
        return list(self._items)


class FakeAnalyzer:
    def __init__(self, score=8, summaries=None):
        self.score, self.summaries = score, summaries or {}

    def analyze_items(self, items):
        out = []
        for item in items:
            n = int(item.title.rsplit(" ", 1)[1])
            out.append(item.with_analysis(self.score, self.summaries.get(n, f"Summary {n}.")))
        return AnalysisRun(analyzed=out)


class FakeGateway:
    """`script`: one entry per send attempt: None = succeed, or an exception instance to raise."""

    def __init__(self, script=(), ready_error=None, dry=False):
        self.script, self.ready_error, self.dry = list(script), ready_error, dry
        self.sent, self.ready_calls = [], 0

    def ensure_ready(self):
        self.ready_calls += 1
        if self.ready_error:
            raise self.ready_error

    def send_text(self, chat_id, text):
        step = self.script.pop(0) if self.script else None
        if isinstance(step, BaseException):  # includes Crash below
            raise step
        self.sent.append((chat_id, text))
        return SendResult(message_id=f"m{len(self.sent)}", dry_run=self.dry)


@pytest.fixture
def repo():
    with TrendRepository(":memory:") as r:
        yield r


def deps(repo, gateway, n=3, analyzer=None, **kw) -> WorkflowDeps:
    return WorkflowDeps(
        collectors=[FakeCollector(make(i) for i in range(1, n + 1))],
        repository=repo,
        analyzer=analyzer or FakeAnalyzer(),
        config=WorkflowConfig(min_relevance_score=6, max_messages_per_run=5),
        gateway=gateway,
        chat_id=GROUP,
        **kw,
    )


def statuses(repo):
    return {k: v for k, v in repo.count_by_status().items() if v}


def note_of(repo, item_id):
    return repo._conn.execute("SELECT status_note FROM trend_items WHERE id=?", (item_id,)).fetchone()[0]


# =====================================================================
# Happy path & idempotency
# =====================================================================
def test_items_are_formatted_sent_to_the_group_and_marked_sent(repo):
    gateway = FakeGateway()
    state = run_once(deps(repo, gateway))

    assert state["metrics"]["delivered"] == 3 and state["metrics"]["delivery_aborted"] == 0
    assert statuses(repo) == {"sent": 3}
    assert {chat for chat, _ in gateway.sent} == {GROUP}
    expected = {format_item(i).text for i in state["to_deliver"]}
    assert {text for _, text in gateway.sent} == expected  # exactly what the formatter produces
    assert gateway.ready_calls == 1  # readiness is checked once, before the first send


def test_a_second_run_never_reposts_anything(repo):
    gateway = FakeGateway()
    d = deps(repo, gateway)
    run_once(d)
    second = run_once(d)
    assert len(gateway.sent) == 3
    assert second["metrics"]["to_deliver"] == 0 and "delivered" not in second["metrics"]
    assert gateway.ready_calls == 1  # the deliver node was not even entered


def test_the_display_timezone_reaches_the_formatter(repo):
    late = make(1, published_at=datetime(2026, 9, 19, 22, 30, tzinfo=timezone.utc))
    gateway = FakeGateway()
    d = WorkflowDeps(
        collectors=[FakeCollector([late])], repository=repo, analyzer=FakeAnalyzer(),
        rules=ValidationRules(max_age_days=10_000, max_future_hours=10_000),  # the fixed date must not be filtered
        gateway=gateway, chat_id=GROUP, display_tz=timezone(timedelta(hours=3)),
    )
    run_once(d)
    assert "20 Sep 2026" in gateway.sent[0][1]  # 22:30 UTC is already the 20th in East Africa Time


# =====================================================================
# Dry run: rehearsal must not consume the queue
# =====================================================================
def test_dry_run_previews_without_marking_sent_and_a_later_live_run_delivers_the_same_items(repo):
    preview = DryRunClient(max_sends=5)
    state = run_once(deps(repo, preview))
    assert state["metrics"]["previewed"] == 3 and state["metrics"]["delivered"] == 0
    assert statuses(repo) == {"analyzed": 3}  # untouched: still waiting for a real send
    assert len(preview.sent) == 3

    again = DryRunClient(max_sends=5)
    assert run_once(deps(repo, again))["metrics"]["previewed"] == 3  # rehearsals are repeatable

    live = FakeGateway()
    state = run_once(deps(repo, live))
    assert state["metrics"]["delivered"] == 3 and statuses(repo) == {"sent": 3}


# =====================================================================
# Outcome policy
# =====================================================================
def test_fatal_error_stops_the_run_and_unsent_items_wait_for_the_next_one(repo):
    gateway = FakeGateway(script=[None, OpenWAAuthError("bad key")])
    state = run_once(deps(repo, gateway))

    assert state["metrics"]["delivered"] == 1 and state["metrics"]["delivery_aborted"] == 1
    assert statuses(repo) == {"sent": 1, "analyzed": 2}
    assert any("OpenWAAuthError" in e for e in state["errors"])

    healthy = FakeGateway()
    assert run_once(deps(repo, healthy))["metrics"]["delivered"] == 2  # the rest go out later
    assert statuses(repo) == {"sent": 3}


def test_delivery_unknown_marks_that_item_rejected_and_stops_so_it_is_never_reposted(repo):
    gateway = FakeGateway(script=[None, OpenWADeliveryUnknownError("no reply")])
    state = run_once(deps(repo, gateway))

    assert state["metrics"]["delivery_unknown"] == 1 and state["metrics"]["delivery_aborted"] == 1
    assert statuses(repo) == {"sent": 1, "rejected": 1, "analyzed": 1}
    (unknown,) = repo.list_by_status(ItemStatus.REJECTED)
    assert "delivery unknown" in note_of(repo, unknown.id)

    healthy = FakeGateway()
    run_once(deps(repo, healthy))
    assert len(healthy.sent) == 1  # only the untouched item; the uncertain one is NOT re-sent
    assert statuses(repo) == {"sent": 2, "rejected": 1}


def test_a_message_the_gateway_rejects_is_dropped_with_a_note_and_the_rest_continue(repo):
    gateway = FakeGateway(script=[OpenWARejectedError("bad text", status_code=422)])
    state = run_once(deps(repo, gateway))
    assert state["metrics"]["delivery_rejected"] == 1 and state["metrics"]["delivered"] == 2
    assert statuses(repo) == {"sent": 2, "rejected": 1}
    (rejected,) = repo.list_by_status(ItemStatus.REJECTED)
    assert "gateway rejected" in note_of(repo, rejected.id)


def test_one_transient_failure_is_retried_next_run_without_stopping_this_one(repo):
    gateway = FakeGateway(script=[OpenWAUnavailableError("blip")])
    state = run_once(deps(repo, gateway))
    assert state["metrics"]["delivery_failed"] == 1 and state["metrics"]["delivered"] == 2
    assert state["metrics"]["delivery_aborted"] == 0
    assert statuses(repo) == {"sent": 2, "analyzed": 1}  # the failed one stays queued

    assert run_once(deps(repo, FakeGateway()))["metrics"]["delivered"] == 1
    assert statuses(repo) == {"sent": 3}


def test_two_consecutive_transient_failures_abort_the_run(repo):
    gateway = FakeGateway(script=[OpenWAUnavailableError("down"), OpenWAUnavailableError("down")])
    state = run_once(deps(repo, gateway, n=4))
    assert state["metrics"]["delivery_failed"] == 2 and state["metrics"]["delivery_aborted"] == 1
    assert statuses(repo) == {"analyzed": 4}
    assert len(gateway.script) == 0 and gateway.sent == []  # the 3rd and 4th items were never attempted


def test_success_resets_the_consecutive_failure_counter(repo):
    gateway = FakeGateway(script=[OpenWAUnavailableError("x"), None, OpenWAUnavailableError("y"), None])
    state = run_once(deps(repo, gateway, n=4))
    assert state["metrics"]["delivery_aborted"] == 0 and state["metrics"]["delivered"] == 2
    assert statuses(repo) == {"sent": 2, "analyzed": 2}


def test_a_session_that_is_not_ready_prevents_any_attempt(repo):
    gateway = FakeGateway(ready_error=OpenWASessionError("session is 'qr_ready', not 'ready'"))
    state = run_once(deps(repo, gateway))
    assert gateway.sent == [] and state["metrics"]["delivery_aborted"] == 1
    assert "qr_ready" in state["errors"][0] and "not attempted" in state["errors"][0]
    assert statuses(repo) == {"analyzed": 3}


def test_an_invalid_destination_aborts_without_touching_item_status(repo):
    class Strict(FakeGateway):
        def send_text(self, chat_id, text):
            raise ValueError("recipient must be a WhatsApp GROUP id")

    state = run_once(deps(repo, Strict()))
    assert state["metrics"]["delivery_aborted"] == 1 and statuses(repo) == {"analyzed": 3}
    assert "invalid message or destination" in state["errors"][-1]


def test_an_item_that_cannot_be_formatted_is_rejected_and_the_others_still_go_out(repo):
    analyzer = FakeAnalyzer(summaries={2: "\u202e \u202c"})  # only invisible bidi controls: empty once sanitised
    gateway = FakeGateway()
    state = run_once(deps(repo, gateway, analyzer=analyzer))
    assert state["metrics"]["delivery_rejected"] == 1 and state["metrics"]["delivered"] == 2
    assert statuses(repo) == {"sent": 2, "rejected": 1}


# =====================================================================
# Graph shape
# =====================================================================
def test_delivery_is_a_conditional_stage_after_select(repo):
    graph = build_workflow(deps(repo, FakeGateway())).get_graph()
    assert "deliver" in graph.nodes
    edges = {(e.source, e.target) for e in graph.edges}
    assert {("select", "deliver"), ("select", "__end__"), ("deliver", "__end__")} <= edges


def test_without_a_gateway_the_pipeline_stops_after_selection(repo):
    d = deps(repo, gateway=None)
    state = run_once(d)
    assert "delivered" not in state["metrics"] and len(state["to_deliver"]) == 3
    assert statuses(repo) == {"analyzed": 3}


# =====================================================================
# Module 15: claim -> send -> record (crash-safe, at-most-once)
# =====================================================================
class Crash(BaseException):
    """Simulates the process being killed mid-send. Not an Exception: nothing in the code can 'handle' it."""


def age_claims(repo):
    """Pretend a crashed run happened long ago (recovery only touches claims older than 10 minutes)."""
    repo._conn.execute("UPDATE trend_items SET claimed_at='2000-01-01T00:00:00.000000+00:00' WHERE status='sending'")
    repo._conn.commit()


def test_each_item_is_claimed_before_its_send_and_confirmed_after(repo):
    class Spy(FakeGateway):
        during = []

        def send_text(self, chat_id, text):
            self.during.append(repo.count_by_status()["sending"])
            return super().send_text(chat_id, text)

    spy = Spy()
    run_once(deps(repo, spy))
    assert spy.during == [1, 1, 1]  # exactly the item being sent is in the SENDING state
    assert statuses(repo) == {"sent": 3}


def test_a_crash_between_send_and_record_never_causes_a_duplicate_post(repo):
    first = FakeGateway(script=[None, Crash()])
    with pytest.raises(Crash):
        run_once(deps(repo, first))
    assert statuses(repo) == {"sent": 1, "sending": 1, "analyzed": 1}  # the 2nd message's fate is unknown

    # A re-run right away must not touch the fresh claim (it may belong to a live run) ...
    second = FakeGateway()
    run_once(deps(repo, second))
    assert len(second.sent) == 1 and statuses(repo) == {"sent": 2, "sending": 1}

    # ... but once the claim is old, it is resolved as "unknown": rejected, NEVER re-sent.
    age_claims(repo)
    third = FakeGateway()
    state = run_once(deps(repo, third))
    assert third.sent == []
    assert state["metrics"]["interrupted_recovered"] == 1
    assert any("NOT re-sent" in e for e in state["errors"])
    assert statuses(repo) == {"sent": 2, "rejected": 1}

    posted = [text for gw in (first, second, third) for _, text in gw.sent]
    assert len(posted) == len(set(posted)) == 2  # no item was ever posted twice


def test_an_unexpected_exception_keeps_the_claim_because_the_outcome_is_unknown(repo):
    with pytest.raises(RuntimeError, match="bug"):
        run_once(deps(repo, FakeGateway(script=[RuntimeError("bug in the gateway code")])))
    assert statuses(repo) == {"sending": 1, "analyzed": 2}


def test_fatal_and_transient_failures_release_the_claim_because_nothing_was_sent(repo):
    run_once(deps(repo, FakeGateway(script=[OpenWAAuthError("bad key")])))
    assert statuses(repo) == {"analyzed": 3}

    run_once(deps(repo, FakeGateway(script=[OpenWAUnavailableError("x"), OpenWAUnavailableError("y")])))
    assert statuses(repo) == {"analyzed": 3}
    assert repo._conn.execute("SELECT COUNT(*) FROM trend_items WHERE claimed_at IS NOT NULL").fetchone()[0] == 0


def test_an_item_claimed_by_another_run_is_skipped_not_sent_twice(repo):
    items = run_once(deps(repo, gateway=None))["to_deliver"]
    repo.claim_for_delivery(items[0].id)  # a concurrent run got there first

    gateway = FakeGateway()
    out = TrendNodes(deps(repo, gateway)).deliver({"to_deliver": items})
    assert out["metrics"]["delivered"] == 2 and len(gateway.sent) == 2
    assert statuses(repo) == {"sent": 2, "sending": 1}


def test_a_dry_run_never_claims_so_an_interrupted_rehearsal_cannot_lose_an_item(repo):
    class CrashingRehearsal(DryRunClient):
        def send_text(self, chat_id, text):
            if self.sent:
                raise Crash()
            return super().send_text(chat_id, text)

    with pytest.raises(Crash):
        run_once(deps(repo, CrashingRehearsal(max_sends=5)))
    assert statuses(repo) == {"analyzed": 3}


# =====================================================================
# Module 16: delivery audit trail (logger "trendwatch.audit")
# =====================================================================
def audit_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "trendwatch.audit"]


def test_audit_records_an_attempt_then_an_outcome_for_every_item(repo, caplog):
    caplog.set_level(logging.INFO, logger="trendwatch.audit")
    run_once(deps(repo, FakeGateway()))
    events = [m.split(" ", 1)[0] for m in audit_messages(caplog)]
    assert events == ["attempt", "sent"] * 3  # strictly interleaved: the intent line always comes first


def test_the_attempt_line_is_written_before_the_send_happens(repo, caplog):
    caplog.set_level(logging.INFO, logger="trendwatch.audit")
    seen = []

    class Spy(FakeGateway):
        def send_text(self, chat_id, text):
            seen.append([m.split(" ", 1)[0] for m in audit_messages(caplog)])
            return super().send_text(chat_id, text)

    run_once(deps(repo, Spy()))
    assert seen[0] == ["attempt"] and seen[1] == ["attempt", "sent", "attempt"]


def test_audit_distinguishes_unknown_released_and_rejected_outcomes(repo, caplog):
    caplog.set_level(logging.INFO, logger="trendwatch.audit")
    gateway = FakeGateway(script=[OpenWARejectedError("bad", status_code=422), OpenWAUnavailableError("blip"),
                                  OpenWADeliveryUnknownError("no reply")])
    run_once(deps(repo, gateway))
    text = "\n".join(audit_messages(caplog))
    assert "rejected item=" in text and "reason=gateway_rejected" in text
    assert "released item=" in text and "OpenWAUnavailableError" in text and "will be retried" in text
    assert "unknown item=" in text and "CHECK THE GROUP" in text
    assert [r.levelname for r in caplog.records if "unknown item" in r.getMessage()] == ["WARNING"]


def test_a_rehearsal_audits_previews_and_never_claims_a_send(repo, caplog):
    caplog.set_level(logging.INFO, logger="trendwatch.audit")
    run_once(deps(repo, DryRunClient(max_sends=5)))
    events = {m.split(" ", 1)[0] for m in audit_messages(caplog)}
    assert events == {"preview"}
