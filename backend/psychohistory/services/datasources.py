"""Additional keyless data providers (ROADMAP section E).

Beyond Google News + Wikipedia, predictions are grounded on more structured
sources. Every provider here is **best-effort**: a failure, timeout or
rate-limit returns an empty list and never breaks a prediction. The online
fetchers are thin wrappers around small, pure ``parse_*`` helpers so the parsing
and selection logic can be unit-tested offline.

Sources
-------
- **Wikidata** (keyless): entities, people and organisations for the scenario.
- **GDELT** (keyless): global news coverage. Public endpoint is aggressively
  rate-limited (HTTP 429), so it is treated as a bonus source.
- **World Bank** (keyless): country macro indicators (population, GDP, inflation).
- **IMF** (keyless): World Economic Outlook real GDP growth.
- **PredictIt** (keyless): prediction-market odds, incl. election markets.
- **ACLED** (key-gated): armed-conflict events; enabled only when
  ``ACLED_API_KEY`` and ``ACLED_EMAIL`` are set in the environment.

Polling aggregators and generic election databases expose no keyless public
API; their coverage is partly served here by Wikidata entities and the
PredictIt election markets.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

_ENABLED = os.getenv("PSYCHO_DATASOURCES", "1").lower() not in ("0", "false", "no", "off")

_HEADERS = {
    "User-Agent": (
        "PsychohistoryResearch/1.0 (https://github.com/VABISMO/psychohistory) "
        "python-httpx"
    ),
    "Accept": "application/json",
}
_TIMEOUT = httpx.Timeout(12.0, connect=8.0)

_WIKIDATA = "https://www.wikidata.org/w/api.php"
_GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
_WORLD_BANK = "https://api.worldbank.org/v2"
_IMF = "https://www.imf.org/external/datamapper/api/v1"
_PREDICTIT = "https://www.predictit.org/api/marketdata/all/"
_ACLED = "https://api.acleddata.com/acled/read"

# World Bank indicators pulled for a country matched in the scenario.
_WB_INDICATORS = [
    ("SP.POP.TOTL", "population"),
    ("NY.GDP.MKTP.CD", "GDP (current US$)"),
    ("FP.CPI.TOTL.ZG", "inflation (annual %)"),
]

# Short aliases that the World Bank's long official names would otherwise miss.
_COUNTRY_ALIASES = {
    "us": "USA", "usa": "USA", "u.s.": "USA", "united states": "USA", "america": "USA",
    "uk": "GBR", "u.k.": "GBR", "britain": "GBR", "england": "GBR",
    "uae": "ARE", "south korea": "KOR", "north korea": "PRK", "russia": "RUS",
    "drc": "COD", "ivory coast": "CIV", "czech republic": "CZE", "vatican": "VAT",
}

# ---------------------------------------------------------------------------
# Pure parsers (offline-testable)
# ---------------------------------------------------------------------------

def parse_wikidata(payload: Any) -> List[Dict[str, str]]:
    """Parse a ``wbsearchentities`` response into source records."""
    out: List[Dict[str, str]] = []
    if not isinstance(payload, dict):
        return out
    for item in payload.get("search", []) or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        url = str(item.get("concepturi") or item.get("url") or "").strip()
        if not label or not url:
            continue
        desc = str(item.get("description") or "").strip()
        out.append({
            "title": label[:200],
            "url": url,
            "snippet": (desc or f"Wikidata entity {item.get('id', '')}").strip()[:300],
            "source": "Wikidata",
        })
    return out


_MONTHS = {
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
}
_CAP_STOP = {
    "The", "A", "An", "And", "Or", "For", "To", "Of", "In", "On", "At", "By",
    "With", "Will", "Would", "Could", "Should", "After", "Before", "Next",
    "Last", "This", "That", "New", "General", "Election", "Elections", "Snap",
    "Party", "Parties", "President", "Prime", "Minister", "Government",
    "Country", "Leader", "Year", "Round", "Rumours",
}


def _entity_candidates(query: str, limit: int = 4) -> List[str]:
    """Extract proper-noun phrases from a scenario to query Wikidata by name.

    ``wbsearchentities`` matches entity labels, not free text, so we search the
    capitalised phrases (people, parties, countries) of the scenario instead of
    the whole sentence.
    """
    out: List[str] = []
    for match in re.findall(r"[A-ZÀ-Þ][A-Za-zÀ-ÿ'.-]+(?:\s+[A-ZÀ-Þ][A-Za-zÀ-ÿ'.-]+)*", query or ""):
        words = match.split()
        if words[0] in _MONTHS or words[0] in _CAP_STOP:
            continue
        candidate = " ".join(words[:4]) if len(words) > 4 else match
        if candidate not in out:
            out.append(candidate)
        if len(out) >= limit:
            break
    return out


def parse_gdelt(payload: Any) -> List[Dict[str, str]]:
    """Parse a GDELT ``artlist`` response (tolerates non-JSON error bodies)."""
    out: List[Dict[str, str]] = []
    if not isinstance(payload, dict):
        return out
    for art in payload.get("articles", []) or []:
        if not isinstance(art, dict):
            continue
        title = str(art.get("title") or "").strip()
        url = str(art.get("url") or "").strip()
        if not title or not url:
            continue
        bits = [str(art.get(k) or "").strip() for k in ("domain", "seendate", "language")]
        out.append({
            "title": title[:200],
            "url": url,
            "snippet": " · ".join(b for b in bits if b) or "GDELT global news",
            "source": "GDELT",
        })
    return out


def _fmt_number(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v.is_integer():
        return f"{int(v):,}"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:,.1f}"


def parse_worldbank(payload: Any) -> List[Dict[str, str]]:
    """Turn a World Bank indicator response into one source record per row."""
    out: List[Dict[str, str]] = []
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        return out
    labels = dict(_WB_INDICATORS)
    for row in payload[1]:
        if not isinstance(row, dict):
            continue
        value = row.get("value")
        if value is None:
            continue
        indicator = row.get("indicator") or {}
        key = str(indicator.get("id") or "").strip()
        label = str(indicator.get("value") or labels.get(key) or key)
        country = str((row.get("country") or {}).get("value") or "").strip()
        iso3 = str(row.get("countryiso3code") or "").strip()
        year = str(row.get("date") or "").strip()
        url = f"{_WORLD_BANK}/country/{iso3}/indicator/{key}" if iso3 and key else _WORLD_BANK
        out.append({
            "title": f"{country} · {label}".strip(" ·")[:200],
            "url": url,
            "snippet": f"{_fmt_number(value)} ({year})".strip(),
            "source": "World Bank",
        })
    return out


def parse_imf(payload: Any, iso3: str, country_name: str = "", indicator: str = "NGDP_RPCH") -> List[Dict[str, str]]:
    """Parse an IMF datamapper response and return the latest forecast as a record."""
    try:
        series = payload["values"][indicator][iso3]
    except (KeyError, TypeError):
        return []
    if not isinstance(series, dict):
        return []
    years = [y for y, v in series.items() if v is not None]
    if not years:
        return []
    year = max(years, key=lambda y: int(y) if str(y).isdigit() else -1)
    label = {
        "NGDP_RPCH": "real GDP growth",
        "PCPIPCH": "inflation",
        "LUR": "unemployment",
    }.get(indicator, indicator)
    where = country_name or iso3
    return [{
        "title": f"{where} · {label}"[:200],
        "url": f"{_IMF}/{indicator}/{iso3}",
        "snippet": f"{_fmt_number(series[year])}% ({year})",
        "source": "IMF",
    }]


def render_facts(records: Optional[List[dict]]) -> List[str]:
    """Render source records into compact one-line facts for the LLM prompt."""
    lines: List[str] = []
    for r in records or []:
        title = str(r.get("title") or "").strip()
        snippet = str(r.get("snippet") or "").strip()
        src = str(r.get("source") or "").strip()
        line = f"{title} {snippet}".strip()
        if not line:
            continue
        if src:
            line = f"{src.upper()}: {line}"
        lines.append(line[:220])
    return lines


_TOKEN_STOP = {
    "will", "the", "for", "and", "are", "was", "with", "this", "that", "from",
    "have", "has", "into", "who", "what", "when", "how", "more",
    # generic political words: present in almost every scenario/market, so they
    # carry no discriminating signal on their own.
    "election", "elections", "general", "vote", "votes", "party", "parties",
    "poll", "polls", "seat", "seats",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and not w.isdigit() and w not in _TOKEN_STOP}


def _contract_odds(contracts: List[dict], max_items: int = 2) -> str:
    parts: List[str] = []
    for c in contracts:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or c.get("shortName") or "").strip()
        yes = c.get("bestBuyYesCost")
        if yes is None:
            yes = c.get("lastTradePrice")
        if yes is None or not name:
            continue
        try:
            parts.append(f"{name} → {round(float(yes) * 100)}% YES")
        except (TypeError, ValueError):
            continue
        if len(parts) >= max_items:
            break
    return "; ".join(parts)


def parse_predictit(payload: Any, query: str, limit: int = 3) -> List[Dict[str, str]]:
    """Select prediction markets relevant to the scenario and surface their odds."""
    out: List[Dict[str, str]] = []
    if not isinstance(payload, dict):
        return out
    qk = _tokens(query)
    scored: List[Tuple[int, dict, str]] = []
    for market in payload.get("markets", []) or []:
        if not isinstance(market, dict):
            continue
        if str(market.get("status", "")).lower() not in ("", "open"):
            continue
        name = str(market.get("name") or "")
        short = str(market.get("shortName") or name).strip()
        if not short:
            continue
        overlap = len(qk & _tokens(f"{name} {short}"))
        if overlap == 0:
            continue
        scored.append((overlap, market, short))
    scored.sort(key=lambda t: -t[0])
    for _, market, short in scored[:limit]:
        odds = _contract_odds(market.get("contracts") or [])
        if not odds:
            continue
        out.append({
            "title": short[:200],
            "url": str(market.get("url") or _PREDICTIT),
            "snippet": f"Implied market odds: {odds}",
            "source": "Prediction market · PredictIt",
        })
    return out


def parse_acled(payload: Any) -> List[Dict[str, str]]:
    """Parse an ACLED read response into conflict-event source records."""
    out: List[Dict[str, str]] = []
    if not isinstance(payload, dict):
        return out
    for ev in payload.get("data", []) or []:
        if not isinstance(ev, dict):
            continue
        etype = str(ev.get("event_type") or "Event").strip()
        loc = str(ev.get("location") or ev.get("admin1") or "").strip()
        country = str(ev.get("country") or "").strip()
        date = str(ev.get("event_date") or "").strip()
        notes = str(ev.get("notes") or "").strip()
        fatalities = ev.get("fatalities")
        try:
            famt = int(float(fatalities))
        except (TypeError, ValueError):
            famt = 0
        head = f"{etype} — {loc}, {country}".strip(" —,")
        bits = [date, (f"fatalities {famt}" if famt else ""), notes[:160]]
        out.append({
            "title": head[:200],
            "url": str(ev.get("source") or _ACLED),
            "snippet": " · ".join(b for b in bits if b) or "ACLED conflict event",
            "source": "ACLED",
        })
    return out


# ---------------------------------------------------------------------------
# Country resolution (World Bank country catalogue, cached in memory)
# ---------------------------------------------------------------------------

_COUNTRY_INDEX: Optional[Dict[str, str]] = None
_COUNTRY_NAMES: Dict[str, str] = {}


def resolve_country(query: str, index: Dict[str, str]) -> Optional[str]:
    """Return the ISO3 code of the country named in ``query`` (longest match wins)."""
    if not query or not index:
        return None
    q = query.lower()
    for name in sorted(index, key=len, reverse=True):
        if re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", q):
            return index[name]
    return None


async def _country_index() -> Dict[str, str]:
    global _COUNTRY_INDEX
    if _COUNTRY_INDEX is not None:
        return _COUNTRY_INDEX
    index: Dict[str, str] = dict(_COUNTRY_ALIASES)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True) as client:
            resp = await client.get(f"{_WORLD_BANK}/country", params={"format": "json", "per_page": 400})
            resp.raise_for_status()
            rows = resp.json()[1] or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            iso3 = str(row.get("id") or "")
            if len(iso3) != 3:
                continue
            name = str(row.get("name") or "").strip()
            if name:
                index.setdefault(name.lower(), iso3)
                _COUNTRY_NAMES.setdefault(iso3, name)
    except Exception as e:
        print(f"[DATASOURCES] country catalogue failed: {type(e).__name__}")
    _COUNTRY_INDEX = index
    return index


def _country_name(iso3: str) -> str:
    return _COUNTRY_NAMES.get(iso3, iso3)


# ---------------------------------------------------------------------------
# Online fetchers
# ---------------------------------------------------------------------------

async def _get_json(client: httpx.AsyncClient, url: str, params: Optional[dict] = None) -> Any:
    resp = await client.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


async def _wikidata_search(client: httpx.AsyncClient, term: str, limit: int) -> List[Dict[str, str]]:
    payload = await _get_json(client, _WIKIDATA, {
        "action": "wbsearchentities", "search": term,
        "language": "en", "uselang": "en", "format": "json", "limit": limit,
    })
    return parse_wikidata(payload)


async def fetch_wikidata(query: str, limit: int = 3) -> List[Dict[str, str]]:
    candidates = _entity_candidates(query) or [query.strip()]
    candidates = [c for c in candidates if c][:4]
    if not candidates:
        return []
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
            results = await asyncio.gather(
                *[_wikidata_search(client, term, 2) for term in candidates],
                return_exceptions=True,
            )
    except Exception as e:
        print(f"[DATASOURCES] Wikidata failed: {type(e).__name__}")
        return []
    seen: set = set()
    out: List[Dict[str, str]] = []
    for res in results:
        if isinstance(res, Exception):
            continue
        for item in res:
            key = item["title"].lower()
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out[:limit]


async def fetch_gdelt(query: str, limit: int = 5) -> List[Dict[str, str]]:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
            resp = await client.get(_GDELT, params={
                "query": query, "mode": "artlist", "maxrecords": limit,
                "format": "json", "timespan": "3m", "sort": "hybridrel",
            })
        # The public endpoint returns plain text (not JSON) when rate-limited.
        if resp.status_code != 200 or not resp.text.lstrip().startswith(("{", "[")):
            print(f"[DATASOURCES] GDELT unavailable (HTTP {resp.status_code})")
            return []
        return parse_gdelt(resp.json())
    except Exception as e:
        print(f"[DATASOURCES] GDELT failed: {type(e).__name__}")
        return []


async def fetch_predictit(query: str, limit: int = 3) -> List[Dict[str, str]]:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
            payload = await _get_json(client, _PREDICTIT)
        return parse_predictit(payload, query, limit)
    except Exception as e:
        print(f"[DATASOURCES] PredictIt failed: {type(e).__name__}")
        return []


async def fetch_acled(query: str, iso3: Optional[str] = None, limit: int = 4) -> List[Dict[str, str]]:
    """Fetch ACLED conflict events. Key-gated: returns [] unless configured."""
    if not acled_enabled():
        return []
    params = {
        "key": os.environ.get("ACLED_API_KEY", ""),
        "email": os.environ.get("ACLED_EMAIL", ""),
        "limit": limit,
        "event_date": _recent_date(),
        "event_date_where": ">=",
        "terms": "accept",
    }
    if iso3:
        params["iso"] = iso3
    else:
        params["search"] = query
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
            payload = await _get_json(client, _ACLED, params)
        return parse_acled(payload)
    except Exception as e:
        print(f"[DATASOURCES] ACLED failed: {type(e).__name__}")
        return []


def acled_enabled() -> bool:
    return bool(os.environ.get("ACLED_API_KEY") and os.environ.get("ACLED_EMAIL"))


def _recent_date(days: int = 90) -> str:
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")


async def external_sources(prompt: str, limit: int = 6) -> List[dict]:
    """Gather keyless external citations (Wikidata/GDELT/PredictIt/ACLED)."""
    if not _ENABLED:
        return []
    wikidata, gdelt, predictit = await asyncio.gather(
        fetch_wikidata(prompt, limit=3),
        fetch_gdelt(prompt, limit=5),
        fetch_predictit(prompt, limit=2),
    )
    items = wikidata + gdelt + predictit
    if acled_enabled():
        index = await _country_index()
        iso3 = resolve_country(prompt, index)
        items += await fetch_acled(prompt, iso3=iso3)
    # De-dup by URL/title and keep the most on-topic items.
    seen = set()
    ordered: List[dict] = []
    qk = _tokens(prompt)
    for it in sorted(
        items,
        key=lambda it: len(qk & _tokens(it.get("title", "") + " " + it.get("snippet", ""))),
        reverse=True,
    ):
        key = (it.get("url") or "").rstrip("/").lower() or (it.get("title") or "").lower()
        if key and key not in seen:
            seen.add(key)
            ordered.append(it)
    print(f"[DATASOURCES] External sources: {len(ordered)} (wikidata={len(wikidata)}, gdelt={len(gdelt)}, predictit={len(predictit)})")
    return ordered[:limit]


async def macro_sources(prompt: str, limit: int = 5) -> List[dict]:
    """Fetch World Bank + IMF macro records for a country named in the scenario."""
    if not _ENABLED:
        return []
    index = await _country_index()
    iso3 = resolve_country(prompt, index)
    if not iso3:
        return []
    name = _country_name(iso3)
    groups = await asyncio.gather(
        _worldbank_sources_for(iso3),
        _imf_sources_for(iso3, name),
    )
    merged: List[dict] = []
    for group in groups:
        merged.extend(group)
    print(f"[DATASOURCES] Macro sources for {iso3}: {len(merged)}")
    return merged[:limit]


async def _worldbank_sources_for(iso3: str) -> List[dict]:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True) as client:
            results = await asyncio.gather(
                *[_get_json(client, f"{_WORLD_BANK}/country/{iso3}/indicator/{code}",
                            {"format": "json", "mrnev": 1})
                  for code, _ in _WB_INDICATORS],
                return_exceptions=True,
            )
    except Exception as e:
        print(f"[DATASOURCES] World Bank failed: {type(e).__name__}")
        return []
    out: List[dict] = []
    for res in results:
        if isinstance(res, Exception):
            continue
        out.extend(parse_worldbank(res))
    return out


async def _imf_sources_for(iso3: str, name: str) -> List[dict]:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=_HEADERS) as client:
            payload = await _get_json(client, f"{_IMF}/NGDP_RPCH/{iso3}")
        return parse_imf(payload, iso3, name)
    except Exception as e:
        print(f"[DATASOURCES] IMF failed: {type(e).__name__}")
        return []
