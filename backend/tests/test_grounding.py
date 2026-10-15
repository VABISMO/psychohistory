"""Tests for temporal grounding + citation attribution (ROADMAP section E).

Run:  backend/venv/bin/python backend/tests/test_grounding.py
"""
import inspect
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.models import Event, Citation
from psychohistory.services import grounding as G

TODAY = date(2026, 1, 15)


def test_resolve_years():
    refs = G.resolve_relative_dates("A climate crisis in 4 years", today=TODAY)
    assert refs and refs[0]["year"] == 2030
    assert refs[0]["iso"] == "2030"


def test_resolve_spanish_years():
    refs = G.resolve_relative_dates("una crisis en 3 años", today=TODAY)
    assert refs and refs[0]["year"] == 2029


def test_resolve_next_year():
    ref = G.resolve_date("The next election happens next year", today=TODAY)
    assert ref is not None
    assert ref["iso"] == "2027"


def test_resolve_months():
    refs = G.resolve_relative_dates("in 3 months there is a vote", today=TODAY)
    assert refs and refs[0]["iso"] == "2026-04"


def test_resolve_days():
    refs = G.resolve_relative_dates("in 10 days the summit opens", today=TODAY)
    assert refs and refs[0]["iso"] == "2026-01-25"


def test_resolve_none():
    assert G.resolve_date("Nothing temporal here", today=TODAY) is None
    assert G.resolve_relative_dates("", today=TODAY) == []


def test_extract_iso_dates():
    found = G.extract_iso_dates("On 2026-11-29 the snap election is held; budget 2027-03.")
    assert "2026-11-29" in found
    assert "2027-03" in found


def test_normalize_citation():
    c = G.normalize_citation("Title", "http://x.test/a", "snippet here", "NEWS")
    assert isinstance(c, Citation)
    assert c.title == "Title" and c.url.endswith("/a")
    assert G.normalize_citation("", "") is None


def test_dedupe_citations():
    a = G.normalize_citation("A", "http://x.test/1", "s")
    b = G.normalize_citation("A", "http://x.test/1/", "s")  # same URL modulo slash
    c = G.normalize_citation("B", "http://x.test/2", "s")
    out = G.dedupe_citations([a, b, c])
    assert len(out) == 2


def test_keyword_overlap():
    assert G.keyword_overlap("Spain election results", "election in Spain today") >= 2
    assert G.keyword_overlap("cats", "quantum physics") == 0


def test_attribute_citations():
    ev = Event(
        name="Spain snap election",
        description="A snap election is called in Spain over the budget.",
        probability=0.5,
        magnitude=0.5,
        trend="stable",
        year=2026,
    )
    sources = [
        Citation(title="Spain calls snap election", url="http://a", snippet="election Spain budget"),
        Citation(title="Weather in Peru", url="http://b", snippet="rain forecast"),
    ]
    G.attribute_citations([ev], sources, top_k=2, min_score=2)
    assert len(ev.citations) == 1
    assert ev.citations[0].url == "http://a"


def test_attribute_citations_no_sources():
    ev = Event(
        name="x", description="y", probability=0.5, magnitude=0.5, trend="stable", year=2026
    )
    assert G.attribute_citations([ev], []) == [ev]
    assert ev.citations == []


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
