"""Temporal grounding + citation attribution (ROADMAP section E).

Two independent, pure capabilities:

* **Temporal grounding** — resolve natural-language time references
  ("next election", "in 3 years", "next year") to concrete ISO dates/years so
  the predicted timeline is anchored, not vague.
* **Citation attribution** — attach the retrieved web sources that best match
  each event (by keyword overlap) so the UI can show *why* an event exists.

No I/O and no network here; everything is unit-tested. ``today`` is injectable
for deterministic tests.
"""
import calendar
import re
from datetime import date, timedelta
from typing import Dict, List, Optional

from ..models import Citation

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "will", "are", "was",
    "what", "when", "who", "has", "have", "into", "about", "over", "than",
    "then", "they", "their", "there", "these", "those", "would", "could",
    "should", "which", "while", "your", "para", "como", "sobre", "entre",
    "donde", "cuando", "porque", "este", "esta",
}


def _shift_months(d: date, months: int) -> date:
    total = (d.month - 1) + months
    year = d.year + total // 12
    month = total % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _iso(d: date) -> str:
    return d.isoformat()


def resolve_relative_dates(text: str, today: Optional[date] = None) -> List[Dict]:
    """Return every resolved time reference, most-specific-first.

    Each entry: ``{"label", "kind", "year", "iso"}`` where ``kind`` is
    ``"year" | "month" | "day"`` and ``iso`` is ``YYYY`` / ``YYYY-MM`` / ``YYYY-MM-DD``.
    """
    if not text:
        return []
    today = today or date.today()
    low = text.lower()
    out: List[Dict] = []

    def add(label, kind, year: int, iso: str):
        out.append({"label": label, "kind": kind, "year": year, "iso": iso})

    for m in re.finditer(r"\b(?:in|within|en|dentro de)\s+(\d+)\s+(years?|años?|anos?)\b", low):
        n = int(m.group(1))
        add(m.group(0), "year", today.year + n, str(today.year + n))

    for m in re.finditer(r"\b(\d+)\s+(years?|años?|anos?)\s+(?:from now|a partir de ahora|desde ahora)\b", low):
        n = int(m.group(1))
        add(m.group(0), "year", today.year + n, str(today.year + n))

    for m in re.finditer(r"\b(?:in|within|en|dentro de)\s+(\d+)\s+(months?|meses?)\b", low):
        n = int(m.group(1))
        d = _shift_months(today, n)
        add(m.group(0), "month", d.year, d.strftime("%Y-%m"))

    for m in re.finditer(r"\b(?:in|within|en|dentro de)\s+(\d+)\s+(days?|días?|dias?)\b", low):
        n = int(m.group(1))
        d = today + timedelta(days=n)
        add(m.group(0), "day", d.year, _iso(d))

    for pat, months, kind in (
        (r"\bnext year\b|\bel año que viene\b|\bel próximo año\b|\bproximo ano\b", 12, "year"),
        (r"\bthis year\b|\beste año\b|\beste ano\b", 0, "year"),
        (r"\bnext month\b|\bel próximo mes\b|\bproximo mes\b", 1, "month"),
        (r"\bthis month\b|\beste mes\b", 0, "month"),
    ):
        if re.search(pat, low):
            d = _shift_months(today, months)
            add(pat, kind, d.year, str(d.year) if kind == "year" else d.strftime("%Y-%m"))

    return out


def resolve_date(text: str, today: Optional[date] = None) -> Optional[Dict]:
    """Return the single best (earliest) resolved time reference, or ``None``."""
    refs = resolve_relative_dates(text, today=today)
    if not refs:
        return None
    return refs[0]


def extract_iso_dates(text: str) -> List[str]:
    """Find ISO dates (``YYYY-MM-DD``) and year-months (``YYYY-MM``) in text."""
    full = re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text or "")
    ym = [m for m in re.findall(r"\b(\d{4}-\d{2})\b", text or "") if m not in {f[:7] for f in full}]
    return full + ym


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def normalize_citation(
    title: str,
    url: str = "",
    snippet: str = "",
    source: str = "",
    retrieved_at: Optional[str] = None,
) -> Optional[Citation]:
    """Build a :class:`Citation`, or ``None`` if there is nothing to show."""
    title = (title or "").strip()
    snippet = (snippet or "").strip()
    if not title and not snippet:
        return None
    return Citation(
        title=title or (snippet[:80]),
        url=(url or "").strip(),
        snippet=snippet[:300],
        source=(source or "").strip(),
        retrieved_at=retrieved_at or date.today().isoformat(),
    )


def _key(c: Citation) -> str:
    if c.url:
        return c.url.rstrip("/").lower()
    return re.sub(r"\W+", " ", c.title.lower()).strip()


def dedupe_citations(citations: List[Citation]) -> List[Citation]:
    seen = set()
    out: List[Citation] = []
    for c in citations:
        k = _key(c)
        if k and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def _keywords(text: str) -> List[str]:
    return [w for w in re.findall(r"\w+", (text or "").lower()) if len(w) > 3 and w not in _STOPWORDS]


def keyword_overlap(a: str, b: str) -> int:
    return len(set(_keywords(a)) & set(_keywords(b)))


def attribute_citations(
    events: List,
    citations: List[Citation],
    top_k: int = 2,
    min_score: int = 1,
) -> List:
    """Attach the best-matching citations to each event, in place; return events.

    Each event gets up to ``top_k`` sources whose keyword overlap with the event
    text (name + description) is at least ``min_score``.
    """
    if not citations:
        return events
    for ev in events:
        text = f"{getattr(ev, 'name', '')} {getattr(ev, 'description', '')}"
        scored = []
        for c in citations:
            score = keyword_overlap(text, f"{c.title} {c.snippet}")
            if score >= min_score:
                scored.append((score, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        chosen = [c for _, c in scored[:top_k]]
        if chosen:
            try:
                ev.citations = chosen
            except Exception:
                pass
    return events
