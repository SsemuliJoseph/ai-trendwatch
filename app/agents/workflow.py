"""
FILE: app/agents/workflow.py   (UPDATED in Module 14: adds the `deliver` stage)
LangGraph workflow orchestration.

ARCHITECTURE NOTES
  * ORCHESTRATION ONLY. Every node is glue that calls code built in earlier modules (collectors,
    validate_items, Deduplicator, TrendRepository, GeminiService, format_item, Gateway). No
    parsing, scoring, SQL or HTTP lives here. `agents/` is the ONE package allowed to import every
    other layer.
  * TERMINOLOGY: LangGraph is the orchestrator. Nodes are deterministic except `analyze`, which
    delegates to the Gemini service (the only LLM step in the system).
  * DEPENDENCY INJECTION: `WorkflowDeps` carries collectors, repository, analyzer, gateway and
    policy. The graph never builds them, so tests inject fakes and main.py builds the real ones.
  * GRAPH:
        START -> collect -> validate -> deduplicate --(backlog > 0)--> analyze -> triage -> select
                                              └--------(backlog == 0)------------------------^
        select --(gateway configured AND items chosen)--> deliver -> END
               └--------------------(otherwise)-----------------------^
  * STATE = what flows between nodes. `metrics` (dict, merged) and `errors` (list, concatenated)
    use REDUCERS so each node reports only its own contribution.
  * BACKLOG-DRIVEN ANALYSIS: `analyze` reads NEW items from SQLite, not just this run's arrivals, so
    leftovers from a quota abort are retried automatically; `max_analyze_per_run` caps API spend.
  * SAFE TO RE-RUN: every step is idempotent (dedup, guarded status transitions).

DELIVERY PROTOCOL (Module 15: claim -> send -> record). Before each real send the item is CLAIMED in
SQLite (ANALYZED -> SENDING, one atomic UPDATE). Two consequences:
  * two overlapping runs can never both send the same item;
  * if the process dies after WhatsApp received the message but before `mark_sent`, the row is still
    SENDING next run and `select` turns it into REJECTED ("interrupted, not re-sent"): a missed item
    is preferred to a duplicate post. Fresh claims (< 10 min) are left alone: they may be live.

DELIVERY POLICY (the `deliver` node; the adapter classifies errors, this node decides what to do):
    success (real)         -> mark_sent, continue
    success (DRY RUN)      -> no claim, no mark: a preview must not consume the queue, otherwise
                              going live later would skip everything that was merely rehearsed
    Fatal (auth/session/   -> release the claim (nothing was sent); stop; retried next run
      rate limit/cap)
    Delivery UNKNOWN       -> the message may be in the group. Mark the item REJECTED with a note
                              (never re-sent: at-most-once beats a duplicate post), then stop
    Rejected by gateway    -> that message is bad: mark REJECTED with a note, continue
    Unavailable (5xx/conn) -> release the claim; item is retried next run; stop after 2 in a row
    session not ready      -> checked ONCE before the first send; nothing is attempted if not ready
    unexpected exception   -> claim deliberately KEPT: the outcome is unknown, so recovery applies
"""
import logging
import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone, tzinfo
from typing import Annotated, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from app.ai.gemini_service import AnalysisRun
from app.config.settings import Settings
from app.database.repository import ItemStatus, TrendRepository
from app.messaging.formatter import format_item
from app.messaging.openwa_client import (
    Gateway,
    OpenWADeliveryUnknownError,
    OpenWAError,
    OpenWAFatalError,
    OpenWARejectedError,
)
from app.models import TrendItem
from app.processing.deduplication import Deduplicator
from app.processing.validation import DEFAULT_RULES, ValidationRules, validate_items

logger = logging.getLogger(__name__)
# Write-ahead audit trail of every delivery (routed to logs/deliveries.log by app/utils/logging_setup.py).
# An "attempt" line is written BEFORE each send; after a crash, the last attempt without an outcome
# names the message whose fate is unknown. Titles are logged with %r so scraped text stays inert.
audit = logging.getLogger("trendwatch.audit")

_EPOCH = datetime.min.replace(tzinfo=timezone.utc)
MAX_CONSECUTIVE_DELIVERY_FAILURES = 2


# ----------------------------------------------------------------------
# Contracts (duck-typed: RSSCollector, ArxivCollector, GeminiService already fit)
# ----------------------------------------------------------------------
class Collector(Protocol):
    failures: dict[str, str]

    def collect(self) -> list[TrendItem]: ...


class Analyzer(Protocol):
    def analyze_items(self, items: Iterable[TrendItem]) -> AnalysisRun: ...


@dataclass(frozen=True)
class WorkflowConfig:
    """Pipeline POLICY (what to keep / how much to spend), separate from wiring."""

    min_relevance_score: int = 6  # items scoring below this are REJECTED after analysis
    max_analyze_per_run: int = 20  # API-spend cap per run
    max_messages_per_run: int = 5  # delivery cap per run

    @classmethod
    def from_settings(cls, settings: Settings) -> "WorkflowConfig":
        return cls(
            min_relevance_score=settings.min_relevance_score,
            max_analyze_per_run=settings.max_analyze_per_run,
            max_messages_per_run=settings.max_messages_per_run,
        )


@dataclass(frozen=True)
class WorkflowDeps:
    collectors: Sequence[Collector]
    repository: TrendRepository
    analyzer: Analyzer
    rules: ValidationRules = DEFAULT_RULES
    config: WorkflowConfig = field(default_factory=WorkflowConfig)
    gateway: Gateway | None = None  # None -> the pipeline stops after `select` (no delivery stage)
    chat_id: str = ""  # destination group (ignored when gateway is None)
    display_tz: tzinfo = timezone.utc  # timezone for the date shown in messages


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------
def _merge_metrics(current: dict[str, int], update: dict[str, int]) -> dict[str, int]:
    return {**current, **update}


class TrendState(TypedDict, total=False):
    collected: list[TrendItem]  # everything collectors returned (newest first)
    valid: list[TrendItem]  # passed validation
    fresh: list[TrendItem]  # actually new: stored in SQLite this run
    backlog: int  # NEW items in SQLite awaiting analysis (this run's + leftovers)
    to_deliver: list[TrendItem]  # best analysed items chosen for delivery
    metrics: Annotated[dict[str, int], _merge_metrics]  # per-stage counters, merged
    errors: Annotated[list[str], operator.add]  # human-readable problems, concatenated


# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------
class TrendNodes:
    """Node implementations bound to their dependencies. Each returns a PARTIAL state update."""

    def __init__(self, deps: WorkflowDeps) -> None:
        self._deps = deps
        self._repo = deps.repository
        self._dedup = Deduplicator(deps.repository)

    def collect(self, state: TrendState) -> dict:
        items: list[TrendItem] = []
        errors: list[str] = []
        for collector in self._deps.collectors:
            name = type(collector).__name__
            try:
                items.extend(collector.collect())
            except Exception as exc:  # noqa: BLE001 - one broken collector must not stop the run
                logger.exception("collector %s crashed", name)
                errors.append(f"{name} crashed: {type(exc).__name__}: {exc}")
            errors.extend(f"{name}[{source}]: {err}" for source, err in collector.failures.items())
        # Newest first, so in-batch dedup keeps the freshest copy of a story (see Deduplicator docs).
        items.sort(key=lambda i: i.published_at or _EPOCH, reverse=True)
        logger.info("collect: %d items, %d error(s)", len(items), len(errors))
        return {"collected": items, "errors": errors, "metrics": {"collected": len(items)}}

    def validate(self, state: TrendState) -> dict:
        result = validate_items(state.get("collected", []), self._deps.rules)
        logger.info("validate: %d valid, rejected=%s", len(result.valid), result.reason_counts())
        return {
            "valid": result.valid,
            "metrics": {"valid": len(result.valid), "rejected_validation": len(result.rejected)},
        }

    def deduplicate(self, state: TrendState) -> dict:
        result = self._dedup.filter_new(state.get("valid", []))
        fresh = self._repo.add_new(result.unique)  # atomic INSERT OR IGNORE = final race-proof guard
        backlog = self._repo.count_by_status()[ItemStatus.NEW.value]
        logger.info("deduplicate: %d new, duplicates=%s, backlog=%d", len(fresh), result.reason_counts(), backlog)
        return {
            "fresh": fresh,
            "backlog": backlog,
            "metrics": {"duplicates": len(result.duplicates), "new_stored": len(fresh), "backlog": backlog},
        }

    def analyze(self, state: TrendState) -> dict:
        batch = self._repo.list_by_status(ItemStatus.NEW, limit=self._deps.config.max_analyze_per_run)
        run = self._deps.analyzer.analyze_items(batch)
        saved = sum(1 for item in run.analyzed if self._repo.update_analysis(item))
        logger.info("analyze: %d saved, %d left NEW, aborted=%s", saved, len(run.unanalyzed), run.aborted)
        return {
            "errors": run.errors,
            "metrics": {
                "analyzed": saved,
                "left_unanalyzed": len(run.unanalyzed),
                "analysis_aborted": int(run.aborted),
            },
        }

    def triage(self, state: TrendState) -> dict:
        """Reject analysed items below the relevance threshold (kept in SQLite, never re-analysed)."""
        minimum = self._deps.config.min_relevance_score
        rejected = 0
        for item in self._repo.list_by_status(ItemStatus.ANALYZED):
            if item.relevance_score is not None and item.relevance_score < minimum:
                rejected += self._repo.mark_rejected(item.id, f"score {item.relevance_score} < {minimum}")
        logger.info("triage: %d rejected below score %d", rejected, minimum)
        return {"metrics": {"rejected_low_score": rejected}}

    def select(self, state: TrendState) -> dict:
        cfg = self._deps.config
        metrics: dict[str, int] = {}
        errors: list[str] = []
        # Claims left behind by a crashed earlier run: outcome unknown -> rejected, NEVER re-sent.
        interrupted = self._repo.recover_interrupted_deliveries()
        if interrupted:
            metrics["interrupted_recovered"] = len(interrupted)
            errors.append(
                f"{len(interrupted)} delivery(ies) from an earlier run were interrupted mid-send and are NOT "
                f"re-sent (outcome unknown; check the group): {', '.join(interrupted)}"
            )
        chosen = self._repo.pending_delivery(cfg.min_relevance_score, cfg.max_messages_per_run)
        metrics["to_deliver"] = len(chosen)
        logger.info("select: %d item(s) chosen for delivery", len(chosen))
        return {"to_deliver": chosen, "metrics": metrics, "errors": errors}

    def deliver(self, state: TrendState) -> dict:
        """Format -> claim -> send -> record, one item at a time (see DELIVERY PROTOCOL in the module docstring)."""
        gateway, items = self._deps.gateway, state.get("to_deliver", [])
        if gateway is None or not items:
            return {}
        rehearsal = bool(getattr(gateway, "dry_run", False))  # DryRunClient: never touch item status

        metrics = {"delivered": 0, "previewed": 0, "delivery_rejected": 0,
                   "delivery_failed": 0, "delivery_unknown": 0, "delivery_aborted": 0}
        errors: list[str] = []

        try:
            gateway.ensure_ready()  # once, before the first send: a dead session fails fast and loudly
        except OpenWAError as exc:
            metrics["delivery_aborted"] = 1
            return {"errors": [f"delivery not attempted: {exc}"], "metrics": metrics}

        consecutive_failures = 0
        for item in items:
            try:
                message = format_item(item, self._deps.display_tz)
            except ValueError as exc:  # e.g. summary empties out after sanitising
                self._repo.mark_rejected(item.id, f"unformattable: {exc}")
                audit.info("rejected item=%s reason=unformattable", item.id)
                metrics["delivery_rejected"] += 1
                errors.append(f"item {item.id} not sent: {exc}")
                continue

            claimed = False
            if not rehearsal:
                claimed = self._repo.claim_for_delivery(item.id)
                if not claimed:  # another run owns it, or it is no longer eligible
                    logger.info("item %s is already claimed by another run; skipping", item.id)
                    continue

            audit.info("%s item=%s chat=%s title=%r", "preview" if rehearsal else "attempt",
                       item.id, self._deps.chat_id, item.title[:80])
            try:
                result = gateway.send_text(self._deps.chat_id, message.text)
            except OpenWAFatalError as exc:
                self._release(item, claimed)
                audit.info("released item=%s reason=%s (nothing was sent)", item.id, type(exc).__name__)
                errors.append(f"delivery stopped ({type(exc).__name__}): {exc}")
                metrics["delivery_aborted"] = 1
                break
            except OpenWADeliveryUnknownError as exc:
                # The message MAY be in the group. Never re-send: prefer a missed item to a duplicate post.
                self._repo.mark_rejected(item.id, "delivery unknown: not retried, to avoid a duplicate post")
                audit.warning("unknown item=%s (request sent, no reply: CHECK THE GROUP; will not be re-sent)", item.id)
                metrics["delivery_unknown"] += 1
                metrics["delivery_aborted"] = 1
                errors.append(f"item {item.id}: {exc}")
                break
            except OpenWARejectedError as exc:
                self._repo.mark_rejected(item.id, f"gateway rejected: {exc}")
                audit.info("rejected item=%s reason=gateway_rejected", item.id)
                metrics["delivery_rejected"] += 1
                errors.append(f"item {item.id} rejected by the gateway: {exc}")
                consecutive_failures = 0
                continue
            except OpenWAError as exc:  # unavailable / 5xx: item goes back in the queue for the next run
                self._release(item, claimed)
                audit.info("released item=%s reason=%s (will be retried next run)", item.id, type(exc).__name__)
                metrics["delivery_failed"] += 1
                consecutive_failures += 1
                errors.append(f"item {item.id} not sent ({type(exc).__name__}): {exc}")
                if consecutive_failures >= MAX_CONSECUTIVE_DELIVERY_FAILURES:
                    metrics["delivery_aborted"] = 1
                    break
                continue
            except ValueError as exc:  # invalid destination/text: a configuration bug, nothing will work
                self._release(item, claimed)
                audit.info("released item=%s reason=invalid_message_or_destination", item.id)
                errors.append(f"delivery stopped (invalid message or destination): {exc}")
                metrics["delivery_aborted"] = 1
                break
            # NOTE: any OTHER exception propagates on purpose and the claim stays: outcome unknown -> recovery.

            consecutive_failures = 0
            if result.dry_run:
                self._release(item, claimed)  # rehearsal result from a gateway not flagged as dry-run
                metrics["previewed"] += 1  # rehearsal: leave the item ANALYZED for a later live run
            elif self._repo.mark_sent(item.id):
                audit.info("sent item=%s gateway_id=%s", item.id, result.message_id or "n/a")
                metrics["delivered"] += 1
            else:
                logger.warning("item %s was sent but could not be marked sent (status changed?)", item.id)
                metrics["delivered"] += 1

        logger.info("deliver: %s", metrics)
        return {"errors": errors, "metrics": metrics}

    def _release(self, item: TrendItem, claimed: bool) -> None:
        if claimed:
            self._repo.release_claim(item.id)


def route_after_dedup(state: TrendState) -> str:
    """Skip the analysis branch (and any API/limiter work) when nothing awaits analysis."""
    return "analyze" if state.get("backlog", 0) > 0 else "select"


# ----------------------------------------------------------------------
# Graph assembly
# ----------------------------------------------------------------------
def build_workflow(deps: WorkflowDeps):
    """Compile the graph. Returns a LangGraph CompiledStateGraph (use `.invoke({})`)."""
    nodes = TrendNodes(deps)
    graph = StateGraph(TrendState)
    for name in ("collect", "validate", "deduplicate", "analyze", "triage", "select", "deliver"):
        graph.add_node(name, getattr(nodes, name))

    def route_after_select(state: TrendState) -> str:
        return "deliver" if deps.gateway is not None and state.get("to_deliver") else "end"

    graph.add_edge(START, "collect")
    graph.add_edge("collect", "validate")
    graph.add_edge("validate", "deduplicate")
    graph.add_conditional_edges("deduplicate", route_after_dedup, {"analyze": "analyze", "select": "select"})
    graph.add_edge("analyze", "triage")
    graph.add_edge("triage", "select")
    graph.add_conditional_edges("select", route_after_select, {"deliver": "deliver", "end": END})
    graph.add_edge("deliver", END)
    return graph.compile()


def run_once(deps: WorkflowDeps) -> TrendState:
    """Execute one full pass and return the final state (metrics, errors, to_deliver)."""
    return build_workflow(deps).invoke({})


if __name__ == "__main__":
    # Print the workflow as a Mermaid diagram (paste into https://mermaid.live). Needs no keys/network.
    demo_repo = TrendRepository(":memory:")
    demo_deps = WorkflowDeps(collectors=[], repository=demo_repo, analyzer=None)  # type: ignore[arg-type]
    print(build_workflow(demo_deps).get_graph().draw_mermaid())
    demo_repo.close()
