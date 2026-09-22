"""
FILE: tests/test_arxiv_collector.py
MODULE 5 tests — run with:  pytest tests/test_arxiv_collector.py
Fully offline: the Atom responses are inline fixtures and HTTP is mocked with `responses`.
"""
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
import responses

from app.collectors.arxiv import API_URL, ArxivCollector, build_query, parse_arxiv
from app.collectors.rss import FeedParseError
from app.models import SourceType

ATOM_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <title>ArXiv Query</title><id>http://arxiv.org/api/query-id</id><updated>2026-09-19T00:00:00Z</updated>
  <opensearch:totalResults>2</opensearch:totalResults>"""

ENTRY_NEW = """
  <entry>
    <id>http://arxiv.org/abs/2609.01234v2</id>
    <updated>2026-09-19T12:00:00Z</updated>
    <published>2026-09-18T17:59:59Z</published>
    <title>Scaling Laws for
  Tiny   Models</title>
    <summary>  We study scaling where $p&lt;q$ and $a&gt;b$ hold.
  Second line.</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <author><name>Grace Hopper</name></author>
    <author><name>Linus T</name></author>
    <link href="http://arxiv.org/abs/2609.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2609.01234v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>"""

ENTRY_OLD_STYLE = """
  <entry>
    <id>http://arxiv.org/abs/hep-th/9901001v1</id>
    <updated>2026-09-10T00:00:00Z</updated>
    <published>2026-09-10T00:00:00Z</published>
    <title>Old style identifier</title>
    <summary>Abstract.</summary>
    <author><name>Solo Author</name></author>
    <category term="hep-th" scheme="http://arxiv.org/schemas/atom"/>
  </entry>"""

ATOM_TWO = ATOM_HEAD + ENTRY_OLD_STYLE + ENTRY_NEW + "\n</feed>"  # deliberately oldest-first
ATOM_EMPTY = ATOM_HEAD + "\n</feed>"
ATOM_ERROR = ATOM_HEAD + """
  <entry>
    <id>http://arxiv.org/api/errors#query_error</id>
    <title>Error</title><summary>query syntax error</summary>
    <updated>2026-09-19T00:00:00Z</updated>
  </entry>
</feed>"""
HTML_PAGE = "<html><body><h1>503 Service Unavailable</h1></body></html>"


# ---------- build_query ----------
def test_build_query_joins_categories():
    assert build_query(["cs.AI", "cs.CL"]) == "cat:cs.AI OR cat:cs.CL"
    assert build_query(["hep-th"]) == "cat:hep-th"


@pytest.mark.parametrize("bad", [[], ["cs.AI OR all:*"], ["cs.AI;drop"], [""], ["1cs.AI"]])
def test_build_query_rejects_invalid_categories(bad):
    with pytest.raises(ValueError):
        build_query(bad)


# ---------- parse_arxiv ----------
def test_parse_maps_fields_and_strips_version():
    items = parse_arxiv(ATOM_TWO)
    top = items[0]
    assert top.url == "https://arxiv.org/abs/2609.01234"  # https, no v2
    assert top.title == "Scaling Laws for Tiny Models"  # newline + spaces collapsed
    assert top.source_type is SourceType.ARXIV
    assert top.source_name == "arXiv cs.CL"  # from arxiv:primary_category
    assert top.author == "Ada Lovelace, Alan Turing, Grace Hopper et al."
    assert top.tags == ("arxiv", "research", "cs.cl", "cs.lg")
    assert top.published_at == datetime(2026, 9, 18, 17, 59, 59, tzinfo=timezone.utc)  # v1 date, not `updated`


def test_abstract_with_latex_symbols_is_not_mangled():
    assert parse_arxiv(ATOM_TWO)[0].content == "We study scaling where $p<q$ and $a>b$ hold. Second line."


def test_versions_of_same_paper_share_one_id():
    v1 = parse_arxiv(ATOM_TWO)[0]
    v3_xml = ATOM_HEAD + ENTRY_NEW.replace("2609.01234v2", "2609.01234v3") + "\n</feed>"
    assert parse_arxiv(v3_xml)[0].id == v1.id


def test_old_style_id_keeps_its_slash_and_single_author_has_no_et_al():
    old = [i for i in parse_arxiv(ATOM_TWO) if "hep-th" in i.url][0]
    assert old.url == "https://arxiv.org/abs/hep-th/9901001"
    assert old.author == "Solo Author"
    assert old.source_name == "arXiv hep-th"  # falls back to first category when no primary tag


def test_sorted_newest_first_and_limit():
    assert [i.title for i in parse_arxiv(ATOM_TWO)] == ["Scaling Laws for Tiny Models", "Old style identifier"]
    assert [i.title for i in parse_arxiv(ATOM_TWO, limit=1)] == ["Scaling Laws for Tiny Models"]


def test_zero_results_is_an_empty_list_not_an_error():
    assert parse_arxiv(ATOM_EMPTY) == []


def test_in_band_api_error_raises():
    with pytest.raises(FeedParseError, match="query syntax error"):
        parse_arxiv(ATOM_ERROR)


@pytest.mark.parametrize("body", [HTML_PAGE, "Service unavailable", b""])
def test_non_feed_bodies_raise(body):
    with pytest.raises(FeedParseError):
        parse_arxiv(body)


# ---------- ArxivCollector (HTTP mocked) ----------
@responses.activate
def test_collector_sends_expected_query_and_returns_items():
    responses.add(responses.GET, API_URL, body=ATOM_TWO, content_type="application/atom+xml")
    collector = ArxivCollector(["cs.AI", "cs.LG"], max_items=5)
    assert len(collector.collect()) == 2
    assert collector.failures == {}

    request = responses.calls[0].request
    params = parse_qs(urlsplit(request.url).query)
    assert params["search_query"] == ["cat:cs.AI OR cat:cs.LG"]
    assert params["sortBy"] == ["submittedDate"] and params["sortOrder"] == ["descending"]
    assert params["max_results"] == ["5"]
    assert "AI-TrendWatch" in request.headers["User-Agent"]


@responses.activate
def test_max_items_is_clamped_to_api_cap():
    responses.add(responses.GET, API_URL, body=ATOM_EMPTY)
    ArxivCollector(max_items=5000).collect()
    assert parse_qs(urlsplit(responses.calls[0].request.url).query)["max_results"] == ["100"]


@responses.activate
@pytest.mark.parametrize(
    "mock, expected",
    [
        (dict(status=429), "HTTPError"),
        (dict(body=requests.exceptions.ConnectTimeout("timed out")), "ConnectTimeout"),
        (dict(body=ATOM_ERROR), "FeedParseError"),
        (dict(body=HTML_PAGE, content_type="text/html"), "FeedParseError"),
    ],
)
def test_failures_are_recorded_not_raised(mock, expected):
    responses.add(responses.GET, API_URL, **mock)
    collector = ArxivCollector()
    assert collector.collect() == []
    assert expected in collector.failures["arxiv"]


@responses.activate
def test_failures_reset_on_next_successful_collect():
    responses.add(responses.GET, API_URL, status=500)
    collector = ArxivCollector()
    collector.collect()
    assert collector.failures
    responses.replace(responses.GET, API_URL, body=ATOM_TWO)
    assert len(collector.collect()) == 2
    assert collector.failures == {}


def test_bad_category_fails_fast_at_construction():
    with pytest.raises(ValueError):
        ArxivCollector(["cs.AI OR all:*"])


# =====================================================================
# Module 15: opt-in retries (arXiv answers bursts with 429)
# =====================================================================
from app.utils.retry import RetryPolicy  # noqa: E402

RETRY = RetryPolicy(max_attempts=3, base_delay=3, max_delay=30, jitter=0)


@responses.activate
def test_a_429_is_retried_after_the_server_hinted_delay_then_succeeds():
    responses.add(responses.GET, API_URL, status=429, headers={"Retry-After": "5"})
    responses.add(responses.GET, API_URL, body=ATOM_TWO)
    sleeps = []
    collector = ArxivCollector(retry=RETRY, sleep=sleeps.append)
    assert len(collector.collect()) == 2 and sleeps == [5.0] and collector.failures == {}


@responses.activate
def test_server_errors_use_exponential_backoff_and_eventually_give_up():
    responses.add(responses.GET, API_URL, status=503)
    sleeps = []
    collector = ArxivCollector(retry=RETRY, sleep=sleeps.append)
    assert collector.collect() == []
    assert sleeps == [3, 6] and len(responses.calls) == 3 and "HTTPError" in collector.failures["arxiv"]


@responses.activate
def test_in_band_api_errors_are_not_retried():
    responses.add(responses.GET, API_URL, body=ATOM_ERROR)  # a bad query stays bad
    sleeps = []
    ArxivCollector(retry=RETRY, sleep=sleeps.append).collect()
    assert len(responses.calls) == 1 and sleeps == []
