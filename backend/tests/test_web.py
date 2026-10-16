"""Tests for web search + page reading (ROADMAP section E).

Pure helpers (HTML extraction, Bing RSS parsing, URL resolution) are tested
offline; ``read_pages`` is only exercised for the skip path so no network runs.

Run:  backend/venv/bin/python backend/tests/test_web.py
"""
import asyncio
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.services import web

ARTICLE_HTML = """
<html><head><title>Spain calls snap election</title>
<meta name="description" content="Pedro Sánchez dissolved parliament.">
<script>var ignored = "SCRIPT_NOISE";</script>
<style>.x{color:red}</style>
</head><body>
<nav>MENU LINKS</nav>
<article>
<h1>Spain calls snap election</h1>
<p>Prime Minister Pedro Sánchez dissolved parliament after losing the housing vote.</p>
<p>Feijóo's PP said it would win a majority.</p>
</article>
<footer>Copyright notice</footer>
</body></html>
"""


def test_extract_readable_title_and_body():
    title, text = web.extract_readable(ARTICLE_HTML, max_chars=1000)
    assert title == "Spain calls snap election"
    assert "Pedro Sánchez" in text
    assert "housing vote" in text
    assert "Feijóo" in text


def test_extract_readable_drops_noise():
    _, text = web.extract_readable(ARTICLE_HTML, max_chars=1000)
    assert "SCRIPT_NOISE" not in text
    assert "MENU LINKS" not in text
    assert "Copyright notice" not in text
    assert ".x{color" not in text


def test_extract_readable_truncates():
    _, text = web.extract_readable(ARTICLE_HTML, max_chars=40)
    assert len(text) <= 40


def test_extract_readable_falls_back_to_meta_description():
    tiny = (
        "<html><head><title>T</title>"
        "<meta name='description' content='Pedro calls snap election.'></head>"
        "<body><p>x</p></body></html>"
    )
    _, text = web.extract_readable(tiny)
    assert "Pedro calls snap election." in text


def test_extract_readable_empty():
    assert web.extract_readable("") == ("", "")
    assert web.extract_readable("<html><body></body></html>")[0] == ""


def test_extract_publisher_url_from_apiclick():
    wrapped = "http://www.bing.com/news/apiclick.aspx?ref=FexRss&url=https%3a%2f%2fwww.elpais.com%2fx%3fy&amp;c=446&amp;mkt=es-es"
    assert web.extract_publisher_url(wrapped) == "https://www.elpais.com/x?y"


def test_extract_publisher_url_passthrough():
    direct = "https://en.m.wikipedia.org/wiki/Spain"
    assert web.extract_publisher_url(direct) == direct
    assert web.extract_publisher_url("") == ""


def test_domain():
    assert web.domain("https://www.elpais.com/x") == "elpais.com"
    assert web.domain("https://news.google.com/rss") == "news.google.com"


def test_is_fetchable():
    assert web.is_fetchable("https://www.reuters.com/a") is True
    assert web.is_fetchable("https://news.google.com/rss/articles/abc") is False
    assert web.is_fetchable("") is False
    assert web.is_fetchable("ftp://x") is False


def test_parse_bing_rss_news():
    xml = (
        "<rss><channel><item>"
        "<title>Housing crisis sparks protests</title>"
        "<link>http://www.bing.com/news/apiclick.aspx?ref=FexRss"
        "&amp;url=https%3a%2f%2fwww.business-standard.com%2fworld-news%2fa.html&amp;c=1</link>"
        "<description>Spaniards march over rents.</description>"
        "</item><item><title>No fetchable link</title>"
        "<link>https://news.google.com/rss/articles/xyz</link>"
        "<description>ignored</description></item></channel></rss>"
    )
    out = web.parse_bing_rss(xml, label="NEWS")
    assert len(out) == 1
    assert out[0]["title"] == "Housing crisis sparks protests"
    assert out[0]["url"] == "https://www.business-standard.com/world-news/a.html"
    assert out[0]["source"] == "NEWS · business-standard.com"
    assert out[0]["snippet"] == "Spaniards march over rents."


def test_parse_bing_rss_web_label_and_malformed():
    xml = (
        "<rss><channel><item><title>Spain</title>"
        "<link>https://en.m.wikipedia.org/wiki/Spain</link>"
        "<description>Country page</description></item></channel></rss>"
    )
    assert web.parse_bing_rss(xml, label="WEB")[0]["source"] == "WEB · en.m.wikipedia.org"
    assert web.parse_bing_rss("not xml") == []
    assert web.parse_bing_rss("") == []


def test_read_pages_skips_unfetchable_without_network():
    out = asyncio.run(web.read_pages(
        [{"url": "https://news.google.com/rss/articles/x"}, {"url": ""}],
        limit=5,
    ))
    assert out == []


if __name__ == "__main__":
    tests = [
        (name, fn)
        for name, fn in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
        if name.startswith("test_")
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
    sys.exit(1 if failed else 0)
