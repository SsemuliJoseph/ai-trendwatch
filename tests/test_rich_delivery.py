"""
FILE: tests/test_rich_delivery.py
MODULE 20 tests — run with:  pytest tests/test_rich_delivery.py
The real graph (LangGraph + repository + validation + formatter) with a fake gateway and a fake image provider:
what is sent when there IS a picture, and every way it falls back to plain text WITHOUT duplicating or losing a post.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.agents.workflow import WorkflowConfig, WorkflowDeps, run_once
from app.ai.gemini_service import AnalysisRun
from app.database.repository import TrendRepository
from app.media import Image
from app.messaging.openwa_client import (
    MAX_CAPTION_CHARS,
    DryRunClient,
    OpenWADeliveryUnknownError,
    OpenWARejectedError,
    OpenWAUnavailableError,
    SendResult,
)
from app.models import SourceType, TrendItem

GROUP = "120363000000000000@g.us"
PNG = Image(b"\x89PNG\r\n\x1a\n" + b"\x01" * 300, "image/png", "card")
SUMMARY = "First key point here\nSecond key point here\nThird key point here"


def make(n: int, **kw) -> TrendItem:
    fields = dict(
        title=f"Notable AI story number {n}", url=f"https://example.com/{n}", source_name="Src",
        source_type=SourceType.RSS, published_at=datetime.now(timezone.utc) - timedelta(hours=1 + n * 0.01),
    )
    return TrendItem(**{**fields, **kw})


class FakeCollector:
    def __init__(self, items):
        self._items, self.failures = list(items), {}

    def collect(self):
        return list(self._items)


class FakeAnalyzer:
    def analyze_items(self, items):
        return AnalysisRun(analyzed=[i.with_analysis(8, SUMMARY) for i in items])


class Gateway:
    """`image_script` / `text_script`: per-attempt None (succeed) or an exception to raise."""

    def __init__(self, image_script=(), text_script=(), dry=False):
        self.image_script, self.text_script, self.dry = list(image_script), list(text_script), dry
        self.images, self.texts = [], []

    def ensure_ready(self):
        return None

    def send_image(self, chat_id, image, mimetype, caption):
        step = self.image_script.pop(0) if self.image_script else None
        if isinstance(step, BaseException):
            raise step
        self.images.append((chat_id, mimetype, len(image), caption))
        return SendResult(message_id=f"i{len(self.images)}", dry_run=self.dry)

    def send_text(self, chat_id, text):
        step = self.text_script.pop(0) if self.text_script else None
        if isinstance(step, BaseException):
            raise step
        self.texts.append((chat_id, text))
        return SendResult(message_id=f"t{len(self.texts)}", dry_run=self.dry)


@pytest.fixture
def repo():
    with TrendRepository(":memory:") as r:
        yield r


def run(repo, gateway, provider, n=3):
    deps = WorkflowDeps(
        collectors=[FakeCollector(make(i) for i in range(1, n + 1))], repository=repo, analyzer=FakeAnalyzer(),
        config=WorkflowConfig(min_relevance_score=6, max_messages_per_run=5), gateway=gateway, chat_id=GROUP,
        image_provider=provider,
    )
    return run_once(deps)


def sent_count(repo):
    return repo.count_by_status().get("sent", 0)


def test_with_a_picture_each_story_is_ONE_image_message_with_the_text_as_caption(repo):
    gateway = Gateway()
    state = run(repo, gateway, lambda item: PNG)
    assert len(gateway.images) == 3 and gateway.texts == []  # no separate text message
    chat, mimetype, size, caption = gateway.images[0]
    assert (chat, mimetype, size) == (GROUP, "image/png", len(PNG.data))
    assert caption.startswith("📰 *Notable AI story number 1*") and "• First key point here" in caption
    assert caption.endswith("Read more:\nhttps://example.com/1") and "elevance" not in caption
    assert state["metrics"]["delivered"] == 3 and sent_count(repo) == 3


def test_without_a_provider_it_is_text_only_exactly_as_before(repo):
    gateway = Gateway()
    run(repo, gateway, None)
    assert gateway.images == [] and len(gateway.texts) == 3 and sent_count(repo) == 3


def test_a_provider_that_finds_nothing_falls_back_to_text(repo):
    gateway = Gateway()
    run(repo, gateway, lambda item: None)
    assert gateway.images == [] and len(gateway.texts) == 3 and sent_count(repo) == 3


def test_a_provider_that_raises_never_stops_the_story(repo):
    def boom(item):
        raise RuntimeError("provider bug")

    gateway = Gateway()
    state = run(repo, gateway, boom)
    assert len(gateway.texts) == 3 and state["metrics"]["delivered"] == 3


def test_mixed_batch_some_pictures_some_text(repo):
    gateway = Gateway()
    run(repo, gateway, lambda item: PNG if item.url.endswith(("/1", "/3")) else None)
    assert len(gateway.images) == 2 and len(gateway.texts) == 1 and sent_count(repo) == 3


def test_a_caption_too_long_for_an_image_is_sent_as_plain_text(repo):
    gateway = Gateway()
    deps = WorkflowDeps(
        collectors=[FakeCollector([make(1, url="https://example.com/" + "p" * (MAX_CAPTION_CHARS + 50))])],
        repository=repo, analyzer=FakeAnalyzer(), gateway=gateway, chat_id=GROUP, image_provider=lambda item: PNG,
    )
    run_once(deps)
    assert gateway.images == [] and len(gateway.texts) == 1 and sent_count(repo) == 1


def test_a_picture_the_gateway_refuses_is_retried_as_text_with_no_duplicate(repo):
    # 4xx = refused = nothing was posted, so sending the same story as text cannot duplicate it.
    gateway = Gateway(image_script=[OpenWARejectedError("bad media", status_code=400)])
    state = run(repo, gateway, lambda item: PNG)
    assert len(gateway.images) == 2 and len(gateway.texts) == 1  # item 1 fell back; items 2-3 got pictures
    assert gateway.texts[0][1].startswith("📰 *Notable AI story number 1*")
    assert state["metrics"]["delivered"] == 3 and state["metrics"]["delivery_rejected"] == 0 and sent_count(repo) == 3


def test_an_unknown_image_outcome_is_never_resent_as_text(repo):
    # The picture MAY be in the group. A text fallback here would risk a duplicate post.
    gateway = Gateway(image_script=[OpenWADeliveryUnknownError("no reply")])
    state = run(repo, gateway, lambda item: PNG)
    assert gateway.texts == [] and gateway.images == []
    assert state["metrics"]["delivery_unknown"] == 1 and state["metrics"]["delivery_aborted"] == 1
    assert sent_count(repo) == 0 and repo.count_by_status().get("rejected", 0) == 1  # at-most-once


def test_a_gateway_outage_on_an_image_releases_the_item_for_the_next_run(repo):
    gateway = Gateway(image_script=[OpenWAUnavailableError("down")])
    state = run(repo, gateway, lambda item: PNG)
    assert state["metrics"]["delivery_failed"] == 1 and gateway.texts == []  # no text fallback on an outage
    assert repo.count_by_status().get("analyzed", 0) >= 1  # released, not lost
    assert sent_count(repo) == 2  # the other two still went out


def test_dry_run_previews_with_pictures_but_consumes_nothing(repo):
    gateway = DryRunClient(max_sends=5)
    state = run(repo, gateway, lambda item: PNG)
    assert state["metrics"]["previewed"] == 3 and sent_count(repo) == 0
    assert len(gateway.images) == 3 and all(chat == GROUP for chat, _, _ in gateway.images)
    assert gateway.sent[0][1].startswith("📰 *Notable AI story number 1*")
