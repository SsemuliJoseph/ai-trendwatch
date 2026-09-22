"""
FILE: tests/test_workflow.py
MODULE 9 tests — run with:  pytest tests/test_workflow.py
The graph runs for real (LangGraph + real TrendRepository in memory + real validation/dedup).
Only the two outside-world edges are faked: collectors (network) and the analyzer (Gemini).
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.agents.workflow import (
    TrendNodes,
    WorkflowConfig,
    WorkflowDeps,
    build_workflow,
    route_after_dedup,
    run_once,
)
from app.ai.gemini_service import AnalysisRun
from app.config.settings import Settings
from app.database.repository import ItemStatus, TrendRepository
from app.models import SourceType, TrendItem


def make(n: int, days_old: float = 0.1, **kw) -> TrendItem:
    """Validation uses the real clock, so publication dates are relative to now."""
    fields = dict(
        title=f"Notable AI story number {n}",
        url=f"https://example.com/{n}",
        source_name="Src",
        source_type=SourceType.RSS,
        published_at=datetime.now(timezone.utc) - timedelta(days=days_old),
    )
    return TrendItem(**{**fields, **kw})


class FakeCollector:
    def __init__(self, items=(), failures=None, boom=False):
        self._items, self.failures, self._boom = list(items), dict(failures or {}), boom

    def collect(self):
        if self._boom:
            raise RuntimeError("collector bug")
        return list(self._items)


class FakeAnalyzer:
    """Scores by item number (from the title); optionally simulates a quota abort."""

    def __init__(self, scores=None, default=8, abort=False):
        self.scores, self.default, self.abort = scores or {}, default, abort
        self.calls: list[int] = []  # batch size of every analyze_items() call

    def analyze_items(self, items):
        items = list(items)
        self.calls.append(len(items))
        if self.abort:
            return AnalysisRun(unanalyzed=items, errors=["GeminiQuotaError: quota exceeded"], aborted=True)
        analysed = []
        for item in items:
            n = int(item.title.rsplit(" ", 1)[1])
            analysed.append(item.with_analysis(self.scores.get(n, self.default), f"Summary {n}"))
        return AnalysisRun(analyzed=analysed)


@pytest.fixture
def repo():
    with TrendRepository(":memory:") as r:
        yield r


def deps_for(repo, collectors, analyzer, **cfg) -> WorkflowDeps:
    return WorkflowDeps(collectors=collectors, repository=repo, analyzer=analyzer, config=WorkflowConfig(**cfg))


def statuses(repo):
    return {k: v for k, v in repo.count_by_status().items() if v}


# =====================================================================
# Whole-graph behaviour
# =====================================================================
def test_full_pass_scores_triages_and_selects_the_best_within_the_cap(repo):
    scores = {1: 9, 2: 7, 3: 3, 4: 8, 5: 10, 6: 6}
    analyzer = FakeAnalyzer(scores)
    deps = deps_for(repo, [FakeCollector(make(n) for n in range(1, 7))], analyzer, max_messages_per_run=3)

    state = run_once(deps)

    assert state["metrics"] == {
        "collected": 6, "valid": 6, "rejected_validation": 0,
        "duplicates": 0, "new_stored": 6, "backlog": 6,
        "analyzed": 6, "left_unanalyzed": 0, "analysis_aborted": 0,
        "rejected_low_score": 1, "to_deliver": 3,
    }
    assert [i.relevance_score for i in state["to_deliver"]] == [10, 9, 8]  # best first, capped at 3
    assert statuses(repo) == {"analyzed": 5, "rejected": 1}
    assert state["errors"] == []
    assert analyzer.calls == [6]


def test_rerunning_the_same_feeds_is_idempotent_and_skips_the_analysis_branch(repo):
    analyzer = FakeAnalyzer()
    deps = deps_for(repo, [FakeCollector(make(n) for n in range(1, 4))], analyzer)
    first = run_once(deps)
    second = run_once(deps)

    assert second["metrics"]["new_stored"] == 0 and second["metrics"]["duplicates"] == 3
    assert "analyzed" not in second["metrics"]  # conditional edge bypassed `analyze`
    assert analyzer.calls == [3]  # no second API call
    assert [i.id for i in second["to_deliver"]] == [i.id for i in first["to_deliver"]]  # still pending, not duplicated
    assert statuses(repo) == {"analyzed": 3}


def test_stale_items_are_rejected_by_validation_and_never_stored(repo):
    collector = FakeCollector([make(1), make(2, days_old=30)])
    state = run_once(deps_for(repo, [collector], FakeAnalyzer()))
    assert state["metrics"]["rejected_validation"] == 1
    assert sum(repo.count_by_status().values()) == 1


def test_collector_failures_and_crashes_are_reported_while_healthy_collectors_proceed(repo):
    flaky = FakeCollector([make(1)], failures={"Feed X": "HTTPError: 500"})
    crashing = FakeCollector(boom=True)
    healthy = FakeCollector([make(2)])
    state = run_once(deps_for(repo, [flaky, crashing, healthy], FakeAnalyzer()))

    assert state["metrics"]["collected"] == 2
    assert any("Feed X" in e and "HTTPError" in e for e in state["errors"])
    assert any("crashed" in e and "collector bug" in e for e in state["errors"])
    assert statuses(repo) == {"analyzed": 2}


def test_items_from_several_collectors_are_merged_newest_first(repo):
    older = make(1, days_old=2)
    newer = make(2, days_old=0.1)
    state = run_once(deps_for(repo, [FakeCollector([older]), FakeCollector([newer])], FakeAnalyzer()))
    assert [i.id for i in state["collected"]] == [newer.id, older.id]


# =====================================================================
# Backlog, quota and cost control
# =====================================================================
def test_quota_abort_leaves_items_new_and_the_next_run_retries_them_without_new_arrivals(repo):
    items = [make(n) for n in range(1, 4)]

    aborted = run_once(deps_for(repo, [FakeCollector(items)], FakeAnalyzer(abort=True)))
    assert statuses(repo) == {"new": 3}
    assert aborted["metrics"]["analysis_aborted"] == 1 and aborted["metrics"]["analyzed"] == 0
    assert aborted["to_deliver"] == [] and "GeminiQuotaError" in aborted["errors"][0]

    # Next run: feeds return NOTHING new, but the backlog is analysed anyway.
    recovered = run_once(deps_for(repo, [FakeCollector([])], FakeAnalyzer()))
    assert recovered["metrics"]["backlog"] == 3 and recovered["metrics"]["analyzed"] == 3
    assert len(recovered["to_deliver"]) == 3
    assert statuses(repo) == {"analyzed": 3}


def test_max_analyze_per_run_caps_spend_and_the_rest_waits_for_later_runs(repo):
    analyzer = FakeAnalyzer()
    first = deps_for(repo, [FakeCollector(make(n) for n in range(1, 8))], analyzer, max_analyze_per_run=3)
    run_once(first)
    assert analyzer.calls == [3] and statuses(repo) == {"analyzed": 3, "new": 4}

    later = deps_for(repo, [FakeCollector([])], analyzer, max_analyze_per_run=3)
    run_once(later)
    run_once(later)
    assert analyzer.calls == [3, 3, 1] and statuses(repo) == {"analyzed": 7}


def test_low_scores_are_rejected_with_a_note_and_never_analysed_again(repo):
    analyzer = FakeAnalyzer({1: 2})
    deps = deps_for(repo, [FakeCollector([make(1)])], analyzer, min_relevance_score=6)
    run_once(deps)

    (rejected,) = repo.list_by_status(ItemStatus.REJECTED)
    assert rejected.relevance_score == 2
    assert repo._conn.execute("SELECT status_note FROM trend_items").fetchone()[0] == "score 2 < 6"

    run_once(deps)  # same feed again
    assert analyzer.calls == [1]


def test_threshold_is_inclusive(repo):
    state = run_once(deps_for(repo, [FakeCollector([make(1)])], FakeAnalyzer({1: 6}), min_relevance_score=6))
    assert [i.relevance_score for i in state["to_deliver"]] == [6]


# =====================================================================
# Structure & units
# =====================================================================
def test_nodes_work_in_isolation(repo):
    nodes = TrendNodes(deps_for(repo, [], FakeAnalyzer()))
    out = nodes.validate({"collected": [make(1), make(2, days_old=30)]})
    assert len(out["valid"]) == 1 and out["metrics"] == {"valid": 1, "rejected_validation": 1}
    assert nodes.select({})["to_deliver"] == []  # empty database -> nothing to deliver


@pytest.mark.parametrize("state, expected", [({}, "select"), ({"backlog": 0}, "select"), ({"backlog": 4}, "analyze")])
def test_router_only_enters_analysis_when_there_is_a_backlog(state, expected):
    assert route_after_dedup(state) == expected


def test_graph_has_the_documented_nodes_and_edges(repo):
    graph = build_workflow(deps_for(repo, [], FakeAnalyzer())).get_graph()
    assert {"collect", "validate", "deduplicate", "analyze", "triage", "select"} <= set(graph.nodes)
    edges = {(e.source, e.target) for e in graph.edges}
    assert {("__start__", "collect"), ("collect", "validate"), ("validate", "deduplicate"),
            ("deduplicate", "analyze"), ("deduplicate", "select"),
            ("analyze", "triage"), ("triage", "select"), ("select", "__end__")} <= edges


def test_workflow_config_comes_from_settings():
    defaults = WorkflowConfig.from_settings(Settings(_env_file=None))
    assert (defaults.min_relevance_score, defaults.max_analyze_per_run, defaults.max_messages_per_run) == (6, 20, 5)
    custom = WorkflowConfig.from_settings(Settings(_env_file=None, min_relevance_score=8, max_messages_per_run=2))
    assert (custom.min_relevance_score, custom.max_messages_per_run) == (8, 2)
