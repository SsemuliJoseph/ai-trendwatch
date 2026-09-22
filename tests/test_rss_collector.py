"""
FILE: tests/test_rss_collector.py
MODULE 4 tests — run with:  pytest tests/test_rss_collector.py
Fully offline: feeds are inline fixtures and HTTP is mocked with `responses`.
"""
from datetime import datetime, timezone

import pytest
import requests
import responses

from app.collectors.rss import FeedParseError, FeedSource, RSSCollector, parse_feed, strip_html
from app.models import SourceType

SRC = FeedSource("Test Feed", "https://feed.test/rss", ("ai",))

RSS_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>T</title><link>https://feed.test</link><description>d</description>
  <item>
    <title>Older post</title>
    <link>https://feed.test/older</link>
    <description>Old.</description>
    <pubDate>Mon, 15 Sep 2026 08:00:00 GMT</pubDate>
  </item>
  <item>
    <title>  Newest &amp; <b>best</b> post </title>
    <link>https://feed.test/new?utm_source=rss&amp;id=7#top</link>
    <description><![CDATA[<p>Hello <b>world</b> &amp; friends</p><script>evil()</script>]]></description>
    <author>dev@feed.test (Dev)</author>
    <category>LLM</category><category>Open Source</category>
    <pubDate>Fri, 19 Sep 2026 10:00:00 +0300</pubDate>
  </item>
  <item><title>No link here</title><description>skipped</description></item>
  <item><link>https://feed.test/no-title</link><description>skipped</description></item>
  <item><title>Undated</title><link>https://feed.test/undated</link></item>
</channel></rss>"""

ATOM_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>A</title><id>x</id><updated>2026-09-19T10:00:00Z</updated>
  <entry>
    <title>Atom entry</title><id>e1</id>
    <link href="https://feed.test/atom-1"/>
    <updated>2026-09-19T10:00:00Z</updated>
    <summary>Atom summary</summary>
    <author><name>Ada</name></author>
  </entry>
</feed>"""

HTML_PAGE = "<html><body><h1>503 Service Unavailable</h1></body></html>"


# ---------- strip_html ----------
def test_strip_html_removes_tags_scripts_and_decodes_entities():
    html = "<p>Hello <b>world</b> &amp; friends</p><script>evil()</script><style>p{}</style><p>Bye</p>"
    assert strip_html(html) == "Hello world & friends Bye"


def test_strip_html_handles_none_and_plain_text():
    assert strip_html(None) == ""
    assert strip_html("just text") == "just text"


# ---------- parse_feed ----------
def test_parse_rss_extracts_fields_and_cleans_data():
    items = parse_feed(RSS_XML, SRC)
    top = items[0]
    assert top.title == "Newest & best post"
    assert top.url == "https://feed.test/new?id=7"  # utm_* and #fragment removed by the model
    assert top.content == "Hello world & friends"  # HTML + <script> stripped
    assert top.source_name == "Test Feed" and top.source_type is SourceType.RSS
    assert top.tags == ("ai", "llm", "open source")  # source tags first, then entry categories
    assert top.author  # feedparser exposes the RSS <author>
    assert top.published_at == datetime(2026, 9, 19, 7, 0, tzinfo=timezone.utc)  # +03:00 -> UTC


def test_entries_without_link_or_title_are_skipped():
    titles = [i.title for i in parse_feed(RSS_XML, SRC)]
    assert "No link here" not in titles
    assert len(titles) == 3  # newest, older, undated


def test_items_sorted_newest_first_with_undated_last():
    assert [i.title for i in parse_feed(RSS_XML, SRC)] == ["Newest & best post", "Older post", "Undated"]


def test_limit_keeps_the_newest_items():
    assert [i.title for i in parse_feed(RSS_XML, SRC, limit=1)] == ["Newest & best post"]


def test_parse_atom():
    (item,) = parse_feed(ATOM_XML, SRC)
    assert (item.title, item.url, item.author, item.content) == (
        "Atom entry", "https://feed.test/atom-1", "Ada", "Atom summary",
    )
    assert item.published_at == datetime(2026, 9, 19, 10, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("body", [HTML_PAGE, "Service unavailable", b"", "<rss><oops"])
def test_non_feed_documents_raise_feed_parse_error(body):
    # HTML error pages (served with HTTP 200), plain text and empty bodies are all "not a feed".
    with pytest.raises(FeedParseError):
        parse_feed(body, SRC)


@pytest.mark.parametrize(
    "empty",
    [
        '<?xml version="1.0"?><rss version="2.0"><channel><title>T</title></channel></rss>',
        '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>A</title>'
        "<id>x</id><updated>2026-09-19T10:00:00Z</updated></feed>",
    ],
)
def test_valid_but_empty_feed_returns_empty_list(empty):
    assert parse_feed(empty, SRC) == []


# ---------- RSSCollector (HTTP mocked) ----------
@responses.activate
def test_collector_returns_items_and_sends_user_agent():
    responses.add(responses.GET, SRC.url, body=RSS_XML, content_type="application/rss+xml")
    collector = RSSCollector([SRC], max_items_per_source=10)
    assert len(collector.collect()) == 3
    assert collector.failures == {}
    assert "AI-TrendWatch" in responses.calls[0].request.headers["User-Agent"]


@responses.activate
def test_one_failing_feed_does_not_stop_the_others():
    good = FeedSource("Good", "https://good.test/rss")
    responses.add(responses.GET, "https://down.test/rss", status=500)
    responses.add(responses.GET, "https://slow.test/rss", body=requests.exceptions.ConnectTimeout("timed out"))
    responses.add(responses.GET, "https://html.test/rss", body=HTML_PAGE, content_type="text/html")
    responses.add(responses.GET, good.url, body=ATOM_XML)

    collector = RSSCollector(
        [
            FeedSource("Down", "https://down.test/rss"),
            FeedSource("Slow", "https://slow.test/rss"),
            FeedSource("Html", "https://html.test/rss"),
            good,
        ]
    )
    items = collector.collect()

    assert [i.source_name for i in items] == ["Good"]
    assert set(collector.failures) == {"Down", "Slow", "Html"}
    assert "HTTPError" in collector.failures["Down"]
    assert "ConnectTimeout" in collector.failures["Slow"]


@responses.activate
def test_failures_are_reset_on_each_collect():
    responses.add(responses.GET, SRC.url, status=404)
    collector = RSSCollector([SRC])
    collector.collect()
    assert "Test Feed" in collector.failures
    responses.replace(responses.GET, SRC.url, body=RSS_XML)
    collector.collect()
    assert collector.failures == {}


# =====================================================================
# Module 15: opt-in retries with backoff
# =====================================================================
from app.utils.retry import RetryPolicy  # noqa: E402

RETRY = RetryPolicy(max_attempts=3, base_delay=1, max_delay=30, jitter=0)


def retrying(sleeps):
    return RSSCollector([SRC], retry=RETRY, sleep=sleeps.append)


@responses.activate
def test_transient_failures_are_retried_with_exponential_backoff_then_succeed():
    responses.add(responses.GET, SRC.url, status=503)
    responses.add(responses.GET, SRC.url, status=502)
    responses.add(responses.GET, SRC.url, body=RSS_XML)
    sleeps = []
    collector = retrying(sleeps)
    assert len(collector.collect()) == 3
    assert sleeps == [1, 2] and len(responses.calls) == 3 and collector.failures == {}


@responses.activate
def test_retrying_gives_up_after_max_attempts_and_records_the_failure():
    responses.add(responses.GET, SRC.url, status=503)
    sleeps = []
    collector = retrying(sleeps)
    assert collector.collect() == []
    assert len(responses.calls) == 3 and sleeps == [1, 2]
    assert "HTTPError" in collector.failures["Test Feed"]


@responses.activate
def test_permanent_errors_fail_immediately_without_retrying():
    responses.add(responses.GET, SRC.url, status=404)
    sleeps = []
    retrying(sleeps).collect()
    assert len(responses.calls) == 1 and sleeps == []


@responses.activate
def test_connection_errors_are_retried():
    responses.add(responses.GET, SRC.url, body=requests.exceptions.ConnectionError("reset"))
    responses.add(responses.GET, SRC.url, body=RSS_XML)
    sleeps = []
    assert len(retrying(sleeps).collect()) == 3 and sleeps == [1]


@responses.activate
def test_retry_after_from_the_server_is_honoured_but_a_huge_one_is_not_waited_for():
    responses.add(responses.GET, SRC.url, status=429, headers={"Retry-After": "7"})
    responses.add(responses.GET, SRC.url, body=RSS_XML)
    sleeps = []
    assert len(retrying(sleeps).collect()) == 3 and sleeps == [7.0]

    responses.reset()
    responses.add(responses.GET, SRC.url, status=429, headers={"Retry-After": "999"})
    sleeps = []
    collector = retrying(sleeps)
    collector.collect()
    assert len(responses.calls) == 1 and sleeps == [] and collector.failures


@responses.activate
def test_by_default_nothing_is_retried():
    responses.add(responses.GET, SRC.url, status=503)
    responses.add(responses.GET, SRC.url, body=RSS_XML)
    collector = RSSCollector([SRC])
    assert collector.collect() == [] and len(responses.calls) == 1
