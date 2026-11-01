"""
FILE: tests/test_media.py
MODULE 20 tests — run with:  pytest tests/test_media.py
Article-image fetching (HTTP mocked with `responses`, DNS faked), the generated title card, and the provider.
No real network, no WhatsApp.
"""
import io
import socket

import pytest
import responses
from PIL import Image as PILImage

from app.media import Image
from app.media.card import SIZE, render_card
from app.media.fetcher import (
    MAX_IMAGE_BYTES,
    MIN_IMAGE_BYTES,
    ArticleImageFetcher,
    extract_image_url,
    sniff_image_type,
)
from app.media.provider import ImageProvider
from app.models import SourceType, TrendItem

PAGE = "https://news.example.com/story"
IMG = "https://cdn.example.com/hero.jpg"
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 4000
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 4000
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 4000


def item(url=PAGE, source_type=SourceType.RSS, title="Notable AI story number 1") -> TrendItem:
    return TrendItem(title=title, url=url, source_name="Demo Feed", source_type=source_type)


def public_dns(host, port):  # every name resolves to a public address
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]


def fetcher(**kw) -> ArticleImageFetcher:
    return ArticleImageFetcher(resolver=kw.pop("resolver", public_dns), **kw)


def html(*metas: str) -> str:
    return f"<html><head><title>x</title>{''.join(metas)}</head><body><p>hello</p></body></html>"


OG = f'<meta property="og:image" content="{IMG}">'


# =====================================================================
# Finding the image URL in the page head
# =====================================================================
def test_og_image_is_found_and_made_absolute():
    assert extract_image_url(html(OG), PAGE) == IMG
    assert extract_image_url(html('<meta property="og:image" content="/img/a.jpg">'), PAGE) == "https://news.example.com/img/a.jpg"


def test_preference_order_and_attribute_variants():
    page = html('<meta name="twitter:image" content="https://t.example/t.jpg">', OG)
    assert extract_image_url(page, PAGE) == IMG  # og:image beats twitter:image
    assert extract_image_url(html('<meta name="twitter:image" content="https://t.example/t.jpg">'), PAGE) == "https://t.example/t.jpg"
    secure = html(OG, '<meta property="og:image:secure_url" content="https://s.example/s.jpg">')
    assert extract_image_url(secure, PAGE) == "https://s.example/s.jpg"


def test_no_image_meta_or_empty_content_gives_none():
    assert extract_image_url(html("<meta name='description' content='x'>"), PAGE) is None
    assert extract_image_url(html('<meta property="og:image" content="">'), PAGE) is None
    assert extract_image_url("", PAGE) is None


def test_tags_after_the_head_are_ignored():
    page = f"<html><head><title>x</title></head><body>{OG}</body></html>"
    assert extract_image_url(page, PAGE) is None


# =====================================================================
# Image bytes
# =====================================================================
def test_image_type_comes_from_the_bytes_not_the_header():
    assert sniff_image_type(JPEG) == "image/jpeg"
    assert sniff_image_type(PNG) == "image/png"
    assert sniff_image_type(WEBP) == "image/webp"
    for bad in (b"GIF89a" + b"\x00" * 50, b"<svg xmlns=...>", b"<!doctype html>", b"", b"\x00" * 20):
        assert sniff_image_type(bad) is None


# =====================================================================
# Fetching
# =====================================================================
@responses.activate
def test_happy_path_returns_the_article_image():
    responses.add(responses.GET, PAGE, body=html(OG), content_type="text/html")
    responses.add(responses.GET, IMG, body=JPEG, content_type="image/jpeg")
    image = fetcher().fetch(item())
    assert image == Image(data=JPEG, mimetype="image/jpeg", origin="article")
    assert "AI-TrendWatch" in responses.calls[0].request.headers["User-Agent"]  # honest identification


@responses.activate
def test_a_lying_content_type_does_not_matter_but_non_images_are_refused():
    responses.add(responses.GET, PAGE, body=html(OG), content_type="text/html")
    responses.add(responses.GET, IMG, body=b"<html>not an image</html>" + b" " * 3000, content_type="image/jpeg")
    assert fetcher().fetch(item()) is None


@responses.activate
def test_redirects_are_followed_within_the_limit():
    responses.add(responses.GET, "https://short.example/s", status=301, headers={"Location": PAGE})
    responses.add(responses.GET, PAGE, body=html(OG))
    responses.add(responses.GET, IMG, body=PNG)
    assert fetcher().fetch(item(url="https://short.example/s")).mimetype == "image/png"


@responses.activate
def test_redirect_loops_are_cut_off():
    responses.add(responses.GET, PAGE, status=302, headers={"Location": PAGE})
    assert fetcher().fetch(item()) is None
    assert len(responses.calls) == 4  # first request + 3 redirects, then give up


@responses.activate
def test_a_redirect_into_a_private_address_is_refused():
    def dns(host, port):
        ip = "10.0.0.5" if host == "internal.example" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    responses.add(responses.GET, PAGE, status=302, headers={"Location": "https://internal.example/admin"})
    assert fetcher(resolver=dns).fetch(item()) is None
    assert [c.request.url for c in responses.calls] == [PAGE]  # the internal host was never contacted


@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.1.2.3", "192.168.0.9", "172.16.0.1", "169.254.169.254", "0.0.0.0", "::1", "fe80::1", "fc00::1"]
)
@responses.activate
def test_private_loopback_and_metadata_addresses_are_never_fetched(address):
    def dns(host, port):
        return [(socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    assert fetcher(resolver=dns).fetch(item()) is None
    assert len(responses.calls) == 0


@pytest.mark.parametrize("url", ["ftp://x.example/a", "file:///etc/passwd", "https://user:pw@news.example.com/s", "http:///nohost"])
@responses.activate
def test_only_plain_http_urls_without_credentials_are_fetched(url):
    # TrendItem validates URLs too; model_construct skips that so the FETCHER's own guard is what is tested.
    raw = TrendItem.model_construct(title="t", url=url, source_name="S", source_type=SourceType.RSS)
    assert fetcher().fetch(raw) is None
    assert len(responses.calls) == 0


def test_a_name_that_does_not_resolve_is_refused():
    def dns(host, port):
        raise socket.gaierror("no such host")

    assert fetcher(resolver=dns).fetch(item()) is None


@responses.activate
def test_a_mixed_answer_with_any_private_address_is_refused():
    def dns(host, port):  # rebinding-style: one public, one loopback
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    assert fetcher(resolver=dns).fetch(item()) is None


@responses.activate
def test_oversized_and_tiny_images_are_rejected():
    responses.add(responses.GET, PAGE, body=html(OG))
    responses.add(responses.GET, IMG, body=JPEG[:4] + b"\x00" * (MAX_IMAGE_BYTES + 10))
    assert fetcher().fetch(item()) is None
    responses.replace(responses.GET, IMG, body=JPEG[:4] + b"\x00" * (MIN_IMAGE_BYTES - 100))  # a tracking pixel
    assert fetcher().fetch(item()) is None


@responses.activate
def test_a_huge_page_is_read_only_up_to_the_cap_and_the_head_still_works():
    responses.add(responses.GET, PAGE, body=html(OG) + "x" * 2_000_000)
    responses.add(responses.GET, IMG, body=JPEG)
    assert fetcher().fetch(item()).origin == "article"


@pytest.mark.parametrize("status", [403, 404, 429, 500])
@responses.activate
def test_sites_that_refuse_us_are_not_worked_around(status):
    responses.add(responses.GET, PAGE, status=status)
    assert fetcher().fetch(item()) is None
    assert len(responses.calls) == 1  # one honest attempt, no retry with a disguise


@responses.activate
def test_network_errors_never_raise():
    import requests

    responses.add(responses.GET, PAGE, body=requests.ConnectionError("boom"))
    assert fetcher().fetch(item()) is None


@responses.activate
def test_arxiv_items_are_skipped_without_any_request():
    assert fetcher().fetch(item(url="https://arxiv.org/abs/2610.00001", source_type=SourceType.ARXIV)) is None
    assert len(responses.calls) == 0


# =====================================================================
# Title card
# =====================================================================
@pytest.mark.parametrize("source_type", list(SourceType))
def test_card_is_a_valid_png_of_the_expected_size(source_type):
    card = render_card(item(source_type=source_type))
    assert card.mimetype == "image/png" and card.origin == "card"
    assert sniff_image_type(card.data) == "image/png" and len(card.data) < MAX_IMAGE_BYTES
    assert PILImage.open(io.BytesIO(card.data)).size == SIZE


@pytest.mark.parametrize(
    "title",
    [
        "x",
        "Word " * 80,  # overflows four lines: must be truncated, not crash
        "Supercalifragilisticexpialidocious" * 8,  # one unbreakable word wider than the card
        "Luganda: Okwebaza ennyo ✓ – “quotes” and ü ñ",
        "日本語のタイトル",  # glyphs the font may lack: boxes at worst, never an exception
    ],
)
def test_card_survives_awkward_titles(title):
    assert sniff_image_type(render_card(item(title=title or "x")).data) == "image/png"


# =====================================================================
# Provider: article image -> card -> None
# =====================================================================
class StubFetcher:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def fetch(self, _item):
        self.calls += 1
        return self.result


def test_provider_prefers_the_article_image():
    article = Image(JPEG, "image/jpeg", "article")
    assert ImageProvider(StubFetcher(article))(item()) is article


def test_provider_falls_back_to_a_card_when_there_is_no_article_image():
    assert ImageProvider(StubFetcher(None))(item()).origin == "card"


def test_provider_can_be_limited_to_article_images_only():
    assert ImageProvider(StubFetcher(None), use_cards=False)(item()) is None


def test_provider_returns_none_rather_than_raising_when_the_card_fails(monkeypatch):
    def boom(_item):
        raise RuntimeError("font exploded")

    monkeypatch.setattr("app.media.provider.render_card", boom)
    assert ImageProvider(StubFetcher(None))(item()) is None
