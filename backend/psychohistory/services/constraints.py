"""Structured numeric constraint store + validator (ROADMAP section E).

The LLM occasionally emits impossible numbers — e.g. "a party wins 400 seats"
in a 350-seat chamber. This module holds a small, curated store of real totals
and caps for well-known institutions and validates every number against them.

Everything here is pure (no I/O, no network) and therefore unit-tested. The web
search layer still fetches the *current* limits; this store is the offline
back-stop and the enforcement point for the numbers the model emits.
"""
import re
from typing import Dict, List, Optional, Tuple

# kind: "seats" | "members" | "electoral_votes"
CONSTRAINTS: Dict[str, dict] = {
    "spain_congress": {
        "label": "Congress of Deputies (Spain)",
        "kind": "seats",
        "total": 350,
        "aliases": [
            "congress of deputies", "congreso de los diputados", "spanish congress",
            "parliament of spain", "spain's parliament", "cortes generales",
        ],
    },
    "spain_senate": {
        "label": "Senate of Spain",
        "kind": "seats",
        "total": 265,
        "aliases": ["senate of spain", "senado de espana", "spanish senate"],
    },
    "us_house": {
        "label": "United States House of Representatives",
        "kind": "seats",
        "total": 435,
        "aliases": ["house of representatives", "us house", "u.s. house", "the house"],
    },
    "us_senate": {
        "label": "United States Senate",
        "kind": "seats",
        "total": 100,
        "aliases": ["us senate", "u.s. senate", "united states senate"],
    },
    "us_electoral": {
        "label": "United States Electoral College",
        "kind": "electoral_votes",
        "total": 538,
        "aliases": ["electoral college", "electoral votes"],
    },
    "uk_commons": {
        "label": "House of Commons (UK)",
        "kind": "seats",
        "total": 650,
        "aliases": ["house of commons", "uk parliament", "british parliament", "westminster"],
    },
    "eu_parliament": {
        "label": "European Parliament",
        "kind": "seats",
        "total": 720,
        "aliases": ["european parliament", "eu parliament", "europarliament"],
    },
    "germany_bundestag": {
        "label": "Bundestag (Germany)",
        "kind": "seats",
        "total": 630,
        "aliases": ["bundestag", "german parliament"],
    },
    "france_assembly": {
        "label": "National Assembly (France)",
        "kind": "seats",
        "total": 577,
        "aliases": ["national assembly", "assemblee nationale", "french national assembly"],
    },
    "india_lok_sabha": {
        "label": "Lok Sabha (India)",
        "kind": "seats",
        "total": 543,
        "aliases": ["lok sabha", "indian parliament", "indian lower house"],
    },
    "japan_shugiin": {
        "label": "House of Representatives (Japan)",
        "kind": "seats",
        "total": 465,
        "aliases": ["shugiin", "japanese lower house", "house of representatives of japan"],
    },
    "brazil_chamber": {
        "label": "Chamber of Deputies (Brazil)",
        "kind": "seats",
        "total": 513,
        "aliases": ["camara dos deputados", "chamber of deputies of brazil"],
    },
    "canada_commons": {
        "label": "House of Commons (Canada)",
        "kind": "seats",
        "total": 338,
        "aliases": ["house of commons of canada", "canadian parliament"],
    },
    "italy_chamber": {
        "label": "Chamber of Deputies (Italy)",
        "kind": "seats",
        "total": 400,
        "aliases": ["camera dei deputati", "chamber of deputies of italy"],
    },
    "australia_reps": {
        "label": "House of Representatives (Australia)",
        "kind": "seats",
        "total": 151,
        "aliases": ["australian house of representatives"],
    },
    "unsc": {
        "label": "United Nations Security Council",
        "kind": "members",
        "total": 15,
        "extra": {"permanent": 5},
        "aliases": ["security council", "unsecurity council", "un security council", "unsc"],
    },
    "nato": {
        "label": "NATO",
        "kind": "members",
        "total": 32,
        "aliases": ["nato", "north atlantic treaty organization"],
    },
    "us_supreme_court": {
        "label": "Supreme Court of the United States",
        "kind": "members",
        "total": 9,
        "aliases": ["supreme court of the united states", "us supreme court", "u.s. supreme court"],
    },
}

# Words that signal a "seat / member / vote" quantity around a number.
_COUNT_WORDS = (
    "seat", "seats", "escano", "escanos", "escaño", "escaños", "diputado",
    "diputados", "member", "members", "miembro", "miembros", "vote", "votes",
    "voto", "votos", "delegate", "delegates", "elector", "electors",
)

_NUM_RE = r"(\d{1,3}(?:[.,]\d{3})+|\d+)"


def seats_majority(total: int) -> int:
    """Simple majority for a chamber of ``total`` seats (floor(total/2)+1)."""
    return total // 2 + 1


def _parse_int(raw: str) -> Optional[int]:
    raw = raw.strip().replace(" ", "")
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw):
        return int(re.sub(r"[.,]", "", raw))
    try:
        return int(raw)
    except ValueError:
        return None


def get(key: str) -> Optional[dict]:
    return CONSTRAINTS.get(key)


def find(text: str) -> List[Tuple[str, dict]]:
    """Return ``(key, constraint)`` for every constraint whose alias appears in text.

    Longest alias first so "us senate" beats the generic "the house" style hits.
    """
    low = text.lower().replace("'", "").replace("’", "")
    low = re.sub(r"\s+", " ", low)
    hits: List[Tuple[str, dict]] = []
    for key, c in CONSTRAINTS.items():
        for alias in c["aliases"]:
            if alias in low:
                hits.append((key, c))
                break
    return hits


def validate_seats(value: int, total: int) -> Tuple[bool, str]:
    if value < 0:
        return False, f"{value} is negative"
    if value > total:
        return False, f"{value} exceeds the total of {total}"
    return True, "ok"


def validate_share(value: float) -> Tuple[bool, str]:
    if value < 0 or value > 100:
        return False, f"{value}% is outside 0-100%"
    return True, "ok"


def validate_number(value: float, constraint_key: str, kind: Optional[str] = None) -> Tuple[bool, str]:
    """Validate ``value`` against a constraint's total for the given kind."""
    c = CONSTRAINTS.get(constraint_key)
    if not c:
        return True, "unknown constraint (not checked)"
    kind = kind or c["kind"]
    if kind in ("seats", "members", "electoral_votes"):
        return validate_seats(int(value), int(c["total"]))
    return True, "ok"


def enforce_text(text: str) -> Tuple[str, List[str]]:
    """Clamp impossible seat/member counts and >100% shares inside ``text``.

    Returns the corrected text plus a list of human-readable violations found.
    If no constraint is referenced, the text is returned unchanged.
    """
    if not text:
        return text, []

    matched = find(text)
    violations: List[str] = []

    if matched:
        # Largest total available for this text (conservative cap).
        cap = max(c["total"] for _, c in matched)

        def _clamp_count(m: "re.Match") -> str:
            value = _parse_int(m.group(1))
            if value is not None and value > cap:
                violations.append(f"{value} > cap {cap} ({matched[0][0]})")
                return f"{cap} {m.group(2)}"
            return m.group(0)

        count_pat = re.compile(
            rf"{_NUM_RE}\s*({'|'.join(_COUNT_WORDS)})\b", flags=re.IGNORECASE
        )
        text = count_pat.sub(_clamp_count, text)

    def _clamp_pct(m: "re.Match") -> str:
        value = m.group(1)
        try:
            fval = float(value.replace(",", "."))
        except ValueError:
            return m.group(0)
        if fval > 100:
            violations.append(f"{value}% > 100%")
            return "100%"
        return m.group(0)

    text = re.sub(rf"{_NUM_RE}\s*%", _clamp_pct, text)
    return text, violations


def check_text(text: str) -> List[str]:
    """Return violations only (convenience wrapper around :func:`enforce_text`)."""
    _, violations = enforce_text(text)
    return violations
