"""
FILE: app/processing/validation.py
MODULE 7a — Validation engine (deterministic quality gate).

ARCHITECTURE NOTES
  * PURE FUNCTIONS, no I/O, no clock reads hidden inside: `now` is injectable, so
    every rule is testable with a fixed date. This is plain rule-based code — NOT
    an "LLM/agent"; only the Gemini stage (Module 8) earns that word.
  * Division of labour with the model layer (Module 3):
        TrendItem       -> STRUCTURAL validity + normalisation (has a title, URL parses ...)
        validation.py   -> QUALITY POLICY (too old? too short? blocked source?)
    Policy changes here without ever touching the schema.
  * ORDER OF CHECKS is fixed (cheapest first) and FIRST FAILURE WINS, so each
    rejected item carries exactly one reason code; that makes logs and stats
    (Module 16) trivial to aggregate.
  * Validation rejects are NOT stored in SQLite: they are cheap to re-check and cost no
    Gemini quota, unlike analysed/irrelevant items (which Module 6 does remember).
  * Rules live in one frozen dataclass so the orchestrator can build them from
    Settings later without this module importing `app.config`.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from urllib.parse import urlsplit

from app.models import TrendItem


class RejectReason(str, Enum):
    BLOCKED_DOMAIN = "blocked_domain"
    TITLE_TOO_SHORT = "title_too_short"
    FUTURE_DATE = "future_date"  # almost always a feed bug / wrong timezone
    TOO_OLD = "too_old"
    MISSING_DATE = "missing_date"  # only when allow_undated=False


@dataclass(frozen=True)
class ValidationRules:
    min_title_length: int = 10  # rejects "Update", "Untitled", "Read more"
    max_age_days: int = 7  # this is a *recent* updates system
    max_future_hours: int = 24  # tolerance for timezone sloppiness before "future" is a bug
    allow_undated: bool = True  # some feeds omit dates; collected_at is "now", so keep them
    blocked_domains: frozenset[str] = frozenset()  # e.g. frozenset({"spam.example"}); subdomains match too


DEFAULT_RULES = ValidationRules()


@dataclass(frozen=True)
class Rejection:
    item: TrendItem
    reason: RejectReason


@dataclass(frozen=True)
class ValidationResult:
    valid: list[TrendItem] = field(default_factory=list)  # input order preserved
    rejected: list[Rejection] = field(default_factory=list)

    def reason_counts(self) -> dict[str, int]:
        """{'too_old': 12, 'title_too_short': 1} — ready for a log line."""
        return dict(Counter(r.reason.value for r in self.rejected))


def _host_is_blocked(host: str, blocked: frozenset[str]) -> bool:
    for domain in blocked:
        d = domain.lower().lstrip(".")
        if d and (host == d or host.endswith("." + d)):
            return True
    return False


def check_item(item: TrendItem, rules: ValidationRules, now: datetime) -> RejectReason | None:
    """Return the FIRST rule the item breaks, or None if it passes."""
    host = (urlsplit(item.url).hostname or "").lower()
    if _host_is_blocked(host, rules.blocked_domains):
        return RejectReason.BLOCKED_DOMAIN

    if len(item.title) < rules.min_title_length:
        return RejectReason.TITLE_TOO_SHORT

    published = item.published_at  # aware UTC, guaranteed by the model
    if published is None:
        return None if rules.allow_undated else RejectReason.MISSING_DATE
    if published > now + timedelta(hours=rules.max_future_hours):
        return RejectReason.FUTURE_DATE
    if published < now - timedelta(days=rules.max_age_days):
        return RejectReason.TOO_OLD
    return None


def validate_items(
    items: Iterable[TrendItem],
    rules: ValidationRules = DEFAULT_RULES,
    now: datetime | None = None,
) -> ValidationResult:
    """Split `items` into valid / rejected(with reason). `now` is read ONCE so a batch is judged consistently."""
    now = now or datetime.now(timezone.utc)
    result = ValidationResult()
    for item in items:
        reason = check_item(item, rules, now)
        if reason is None:
            result.valid.append(item)
        else:
            result.rejected.append(Rejection(item, reason))
    return result
