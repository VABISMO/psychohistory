"""Web search + page reading (ROADMAP section E).

The retrieval stack used to show only headlines and metadata. This module lets
it actually **open the sources**: it searches Bing (RSS, keyless), resolves the
real publisher URL, downloads the page and extracts its readable text so the
forecast can name the actors and facts that appear in the articles.

Everything is best-effort: any failure returns ``[]`` / ``None`` and never
breaks a prediction. The parsing helpers are pure so they can be unit-tested
offline (no network).
"""
from __future__ import annotations

import asyncio
import html as _html
import re
import time
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import httpx

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120 Safari/537.36 PsychohistoryResearch/1.0"
    ),
    "Accept-Language": "en,es;q=0.8",
}
_TIMEOUT = httpx.Timeout(12.0, connect=8.0)
_BING_NEWS = "https://www.bing.com/news/search"
_BING_WEB = "https://www.bing.com/search"

# Page cache: url -> (expires_at, {title, url, text} | None)
_CACHE: Dict[str, Tuple[float, Optional[dict]]] = {}
_CACHE_TTL = 600.0


# ---------------------------------------------------------------------------
# Pure helpers (offline-testable)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside",
             "form", "svg", "template", "iframe", "figure"}
    _BLOCK = {"p", "div", "article", "section", "h1", "h2", "h3", "h4", "li",
              "br", "tr", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: List[str] = []
        self.parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._BLOCK and self._skip_depth == 0:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self._BLOCK and self._skip_depth == 0:
            self.parts.append("\n")

    def handle_data(self, data):
        text = (data or "").strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        elif self._skip_depth == 0:
            self.parts.append(text)


_META_RE = re.compile(
    r'<meta[^>]+(?:name|property)=["\'](?:description|og:description|og:title|twitter:description)["\']'
    r'[^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def extract_readable(html_text: str, max_chars: int = 1500) -> Tuple[str, str]:
    """Return ``(title, text)`` extracted from raw HTML.

    Drops scripts/styles/navigation and collapses whitespace. Falls back to the
    page's meta description when the visible body text is too thin (common on
    JS-rendered news sites).
    """
    if not html_text:
        return "", ""
    parser = _TextExtractor()
    try:
        parser.feed(html_text)
    except Exception:
        pass
    title = re.sub(r"\s+", " ", " ".join(parser.title_parts)).strip()
    body = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    if len(body) < 200:
        meta = _META_RE.search(html_text)
        if meta:
            extra = _html.unescape(meta.group(1)).strip()
            body = f"{body} {extra}".strip() if body else extra
    return title, body[:max_chars]


def extract_publisher_url(link: str) -> str:
    """Resolve a Bing RSS link to the real publisher URL.

    Bing News wraps links as ``.../apiclick.aspx?...&url=<encoded>``; web results
    are already direct. Returns a plain URL (or "" when nothing usable).
    """
    if not link:
        return ""
    link = _html.unescape(link).strip()
    if "url=" in link and ("bing.com" in link):
        value = parse_qs(urlparse(link).query).get("url", [""])[0]
        if value:
            return value.strip()
    return link


def domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def is_fetchable(url: str) -> bool:
    """True for http(s) pages we should try to download (not Google aggregates)."""
    if not url or not url.lower().startswith(("http://", "https://")):
        return False
    return "news.google.com" not in url


def parse_bing_rss(xml_text: str, label: str = "WEB", limit: int = 8) -> List[dict]:
    """Parse a Bing RSS feed into source records with real publisher URLs."""
    out: List[dict] = []
    if not xml_text:
        return out
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for item in root.findall(".//item")[:limit]:
        title = (item.findtext("title") or "").strip()
        link = extract_publisher_url(item.findtext("link") or "")
        desc = (item.findtext("description") or "").strip()
        if not title or not is_fetchable(link):
            continue
        desc = re.sub(r"<[^>]+>", " ", desc)
        desc = re.sub(r"\s+", " ", _html.unescape(desc)).strip()
        dom = domain(link)
        out.append({
            "title": title[:200],
            "url": link,
            "snippet": desc[:300],
            "source": f"{label} · {dom}" if dom else label,
        })
    return out


# ---------------------------------------------------------------------------
# Online fetchers
# ---------------------------------------------------------------------------

async def _bing(query: str, base: str, label: str, limit: int) -> List[dict]:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True) as client:
            resp = await client.get(base, params={"q": query, "format": "RSS"})
            resp.raise_for_status()
        return parse_bing_rss(resp.text, label=label, limit=limit)
    except Exception as e:
        print(f"[WEB] Bing {label} failed: {type(e).__name__}")
        return []


async def bing_news(query: str, limit: int = 6) -> List[dict]:
    return await _bing(query, _BING_NEWS, "NEWS", limit)


async def bing_web(query: str, limit: int = 6) -> List[dict]:
    return await _bing(query, _BING_WEB, "WEB", limit)


async def fetch_page(url: str, max_chars: int = 1500) -> Optional[dict]:
    """Download a page and return ``{title, url, text}`` (cached, best-effort)."""
    now = time.time()
    cached = _CACHE.get(url)
    if cached and cached[0] > now:
        return cached[1]
    result: Optional[dict] = None
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True
        ) as client:
            resp = await client.get(url)
            if resp.status_code == 200 and "html" in resp.headers.get("content-type", ""):
                title, text = extract_readable(resp.text, max_chars=max_chars)
                if text:
                    result = {"title": title, "url": url, "text": text}
    except Exception as e:
        print(f"[WEB] fetch failed ({type(e).__name__}): {url[:70]}")
    _CACHE[url] = (now + _CACHE_TTL, result)
    return result


async def read_pages(items: List[dict], limit: int = 5, max_chars: int = 1500) -> List[dict]:
    """Fetch the readable text of up to ``limit`` items (concurrently)."""
    urls: List[str] = []
    for it in items:
        url = (it.get("url") or "").strip()
        if url and is_fetchable(url) and url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    if not urls:
        return []
    pages = await asyncio.gather(*[fetch_page(u, max_chars=max_chars) for u in urls])
    out = [p for p in pages if p]
    print(f"[WEB] Read {len(out)}/{len(urls)} pages")
    return out
