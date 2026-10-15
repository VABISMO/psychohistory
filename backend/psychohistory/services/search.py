import asyncio
import re
import xml.etree.ElementTree as ET
from typing import List
from urllib.parse import quote, quote_plus

import httpx

CURATED_SOURCES = [
    "World Bank - World Development Indicators (Gini, GDP, governance)",
    "Freedom House - Freedom in the World Report (freedom indices)",
    "Pew Research Center - Social Trust & Polarization surveys",
    "Our World in Data - long-run social and economic indicators",
    "Transparency International - Corruption Perceptions Index",
    "V-Dem Institute - Democracy indices",
    "arXiv / JSTOR - social complexity, networks and resilience",
    "Historical archives - regime transitions, collapses, revolutions",
]

_HEADERS = {
    "User-Agent": (
        "PsychohistoryResearch/1.0 (https://github.com/psychohistory; "
        "psychohistory@users.noreply.github.com) python-httpx"
    )
}


async def _google_news_items(query: str, limit: int = 8) -> List[dict]:
    url = (
        "https://news.google.com/rss/search?"
        f"q={quote_plus(query)}&hl=es&gl=ES&ceid=ES:es"
    )
    try:
        async with httpx.AsyncClient(timeout=12.0, headers=_HEADERS) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        root = ET.fromstring(resp.text)
        out = []
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            source_el = item.find("source")
            source = (source_el.text if source_el is not None and source_el.text else "").strip()
            if not title:
                continue
            out.append({
                "title": title,
                "url": link,
                "snippet": " · ".join(x for x in [source, pub] if x),
                "source": "NEWS" + (f" · {source}" if source else ""),
            })
        return out
    except Exception as e:
        print(f"[SEARCH] Google News failed: {type(e).__name__}")
        return []


async def _google_news(query: str, limit: int = 8) -> List[str]:
    out = []
    for it in await _google_news_items(query, limit):
        extra = it["snippet"]
        out.append(f"NEWS: {it['title']}" + (f" ({extra})" if extra else ""))
    return out


async def _wikipedia_items(query: str, limit: int = 3, lang: str = "es") -> List[dict]:
    base = f"https://{lang}.wikipedia.org"
    try:
        async with httpx.AsyncClient(timeout=12.0, headers=_HEADERS) as client:
            s = await client.get(
                f"{base}/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": limit,
                },
            )
            s.raise_for_status()
            titles = [x["title"] for x in s.json().get("query", {}).get("search", [])]

            out = []
            for title in titles:
                summary = await client.get(
                    f"{base}/api/rest_v1/page/summary/{quote(title, safe='')}"
                )
                if summary.status_code == 200:
                    data = summary.json()
                    extract = (data.get("extract") or "").strip()
                    if extract:
                        page_url = (
                            data.get("content_urls", {}).get("desktop", {}).get("page")
                            or f"{base}/wiki/{quote(title, safe='')}"
                        )
                        out.append({
                            "title": f"{title} ({lang.upper()})",
                            "url": page_url,
                            "snippet": extract[:320],
                            "source": f"WIKIPEDIA · {lang.upper()}",
                        })
            return out
    except Exception as e:
        print(f"[SEARCH] Wikipedia failed: {type(e).__name__}")
        return []


async def _wikipedia(query: str, limit: int = 3, lang: str = "es") -> List[str]:
    out = []
    for it in await _wikipedia_items(query, limit, lang):
        title = it["title"].split(" (")[0]
        out.append(f"WIKIPEDIA ({title}): {it['snippet']}")
    return out


async def retrieve_context(prompt: str, max_results: int = 10) -> List[str]:
    """
    Retrieve real context from the internet: recent news + encyclopedia + curated
    macro sources. Everything is injected into the LLM to ground the predictions.
    """
    print(f"[SEARCH] Retrieving context for: '{prompt}'")

    news_task = _google_news(prompt)
    wiki_es_task = _wikipedia(prompt, lang="es")
    wiki_en_task = _wikipedia(prompt, lang="en")
    news, wiki_es, wiki_en = await asyncio.gather(news_task, wiki_es_task, wiki_en_task)
    wiki = wiki_es + wiki_en

    # Rerank + dedup before injecting, and chunk any oversized line.
    news = dedup_and_rerank(news, prompt, limit=8)
    wiki = dedup_and_rerank(wiki, prompt, limit=6)

    context: List[str] = []
    context.append(f"USER SCENARIO: {prompt}")
    if news:
        context.append(f"--- RECENT NEWS ({len(news)}) ---")
        for line in news:
            context.extend(chunk_text(line))
    if wiki:
        context.append("--- ENCYCLOPEDIC CONTEXT ---")
        for line in wiki:
            context.extend(chunk_text(line))
    if not news and not wiki:
        context.append("(No live web access; use general knowledge)")

    context.append("--- AVAILABLE DATA SOURCES ---")
    context.append("; ".join(CURATED_SOURCES))

    print(f"[SEARCH] Context: {len(news)} news, {len(wiki)} wiki entries")
    return context[: max_results + 4]


def _dedupe_sources(items: List[dict]) -> List[dict]:
    """Deduplicate structured sources by URL, then by normalized title."""
    seen = set()
    out = []
    for it in items:
        url = (it.get("url") or "").rstrip("/").lower()
        key = url or re.sub(r"\W+", " ", (it.get("title") or "").lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


async def retrieve_sources(prompt: str, max_results: int = 12) -> List[dict]:
    """Retrieve structured evidence sources (title, url, snippet, source).

    Same live sources as :func:`retrieve_context` but returned as records so the
    API can surface per-event citations (ROADMAP section E). Deduped by URL.
    """
    news_items, wiki_es_items, wiki_en_items = await asyncio.gather(
        _google_news_items(prompt),
        _wikipedia_items(prompt, limit=3, lang="es"),
        _wikipedia_items(prompt, limit=3, lang="en"),
    )
    items = _dedupe_sources(news_items + wiki_es_items + wiki_en_items)
    # Rank by keyword overlap with the scenario and keep the top results.
    qwords = set(_keywords(prompt))
    items.sort(
        key=lambda it: len(qwords & set(_keywords(it.get("title", "") + " " + it.get("snippet", "")))),
        reverse=True,
    )
    print(f"[SEARCH] Retrieved {len(items)} structured sources")
    return items[:max_results]


# Generic, domain-agnostic "bound-seeking" terms. They are combined with the
# USER'S OWN scenario, so nothing is hardcoded per domain or use-case: the
# queries always adapt to whatever is being predicted.
_BOUND_TERMS = [
    "total seats majority limit",
    "maximum record highest ever",
    "legal cap constitutional limit",
    "capacidad total limite legal",
    "maximo record historico",
]


def _constraint_queries(prompt: str, k: int = 3) -> List[str]:
    """Derive bound-seeking queries from the user's own scenario."""
    return [f"{prompt} {term}" for term in _BOUND_TERMS[:k]]


def _detemporalize(text: str) -> str:
    """Strip temporal qualifiers (years, months, ordinal days) from a scenario.

    Generic and domain-agnostic: it lets a future/hypothetical scenario match the
    institutional Wikipedia article that states the real total or capacity.
    """
    text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
    text = re.sub(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d{1,2}(?:st|nd|rd|th)\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _has_number(line: str) -> bool:
    return bool(re.search(r"\d[\d\.,]*", line))


_CAPACITY_WORDS = (
    "seat", "seats", "escaño", "escaños", "majority", "mayoria", "mayoría",
    "total", "member", "members", "miembros", "capacity", "capacidad",
    "limit", "límite", "tope", "record", "récord", "maximum", "máximo",
    "percent", "quota", "cuota", "bench", "esquinas",
)


def _score_fact(fact: str, scenario: str) -> float:
    """Rank a retrieved line by on-topic overlap + presence of capacity/limit words."""
    f = fact.lower()
    words = [w for w in re.findall(r"\w+", scenario.lower()) if len(w) > 3]
    score = float(sum(1 for w in words if w in f))
    score += 2.0 * sum(1 for w in _CAPACITY_WORDS if w in f)
    return score


# ---------------------------------------------------------------------------
# Retrieval hygiene (ROADMAP section F): dedup, rerank and chunk evidence.
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "will", "are", "was",
    "what", "when", "who", "has", "have", "into", "about", "over", "than",
    "then", "they", "their", "there", "these", "those", "would", "could",
    "should", "there", "which", "while", "your", "para", "como", "sobre",
    "entre", "donde", "cuando", "porque", "este", "esta", "esos", "esas",
}


def _keywords(text: str) -> List[str]:
    return [
        w for w in re.findall(r"\w+", text.lower())
        if len(w) > 3 and w not in _STOPWORDS
    ]


def _normalize_key(text: str) -> str:
    """A coarse key for near-duplicate detection (prefix + first words)."""
    words = re.findall(r"\w+", text.lower().replace("'", "").replace("’", ""))
    return " ".join(words[:12])


def chunk_text(text: str, max_len: int = 320) -> List[str]:
    """Split a long source line into sentence-aware chunks of ~max_len chars."""
    text = text.strip()
    if len(text) <= max_len:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + 1 + len(s) > max_len:
            chunks.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        chunks.append(buf)
    return chunks


def dedup_and_rerank(items: List[str], query: str, limit: int | None = None) -> List[str]:
    """Drop near-duplicates and order remaining lines by keyword overlap & length.

    Rank = on-topic keyword overlap with the query, small bonus for carrying a
    number, minus a small penalty for overly long lines. Stable and dependency-free.
    """
    if not items:
        return []
    qwords = set(_keywords(query))
    seen = set()
    scored = []
    for idx, raw in enumerate(items):
        line = raw.strip()
        if not line:
            continue
        key = _normalize_key(line)
        if key in seen:
            continue
        seen.add(key)
        words = set(_keywords(line))
        overlap = len(qwords & words)
        has_num = 1.0 if _has_number(line) else 0.0
        length_penalty = 0.001 * max(0, len(line) - 300)
        score = overlap + 0.5 * has_num - length_penalty
        scored.append((score, -idx, line))
    scored.sort(reverse=True)
    ranked = [line for _, _, line in scored]
    return ranked[:limit] if limit is not None else ranked


async def retrieve_constraints(prompt: str) -> List[str]:
    """Fetch real-world maximums / limits / records for whatever the scenario predicts.

    Runs live web queries built from the scenario plus generic bound-seeking
    terms (Spanish and English Wikipedia + news), then keeps only the retrieved
    lines that carry a concrete number (totals, seat counts, caps, historical
    records, percentages). Every numeric bound comes from the web — none is
    hardcoded.
    """
    print(f"[SEARCH] Looking up real maximums/limits for: '{prompt}'")

    tasks = []
    # Direct lookups of the scenario itself (best chance of hitting the
    # institutional article that states the total/capacity).
    tasks.append(_wikipedia(prompt, limit=2, lang="es"))
    tasks.append(_wikipedia(prompt, limit=2, lang="en"))
    # De-temporalized lookups: a future scenario matches the standing article
    # that states the real total/capacity (e.g. the chamber's seat count).
    bare = _detemporalize(prompt)
    if bare:
        tasks.append(_wikipedia(bare, limit=2, lang="es"))
        tasks.append(_wikipedia(bare, limit=2, lang="en"))
    # Bound-seeking queries derived from the user's own scenario.
    for q in _constraint_queries(prompt, k=2):
        tasks.append(_wikipedia(q, limit=1, lang="es"))
        tasks.append(_wikipedia(q, limit=1, lang="en"))
        tasks.append(_google_news(q, limit=3))

    results = await asyncio.gather(*tasks)

    facts: List[str] = []
    seen = set()
    for group in results:
        for line in group:
            if not line or not _has_number(line):
                continue
            key = line[:80].lower()
            if key in seen:
                continue
            seen.add(key)
            facts.append(line[:300])

    # Prioritize the on-topic institutional fact that states the real total/cap.
    facts.sort(key=lambda x: _score_fact(x, prompt), reverse=True)
    facts = [f for f in facts if _score_fact(f, prompt) > 0]

    print(f"[SEARCH] Found {len(facts)} real numeric limit/record facts")
    return facts[:6]
