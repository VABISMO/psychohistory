"""Tests for user-provided sources (ROADMAP section E).

Covers the request normalisation in ``main`` and the evidence rendering in the
predictor, so a user's own sources flow into grounding and citations.

Run:  backend/venv/bin/python backend/tests/test_user_sources.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.main import UserSourceInput, _normalize_user_sources
from psychohistory.services import predictor as P


def test_normalize_empty():
    assert _normalize_user_sources([]) == []


def test_normalize_full_record():
    out = _normalize_user_sources(
        [UserSourceInput(title="Census 2026", url="https://x.example/c", text="turnout was 71%")]
    )
    assert len(out) == 1
    rec = out[0]
    assert rec["title"] == "Census 2026"
    assert rec["url"] == "https://x.example/c"
    assert rec["snippet"] == "turnout was 71%"
    assert rec["source"] == "USER"


def test_normalize_title_defaults_to_url_then_text():
    by_url = _normalize_user_sources([UserSourceInput(url="https://x.example/a")])
    assert by_url[0]["title"] == "https://x.example/a"

    by_text = _normalize_user_sources([UserSourceInput(text="a long private note about the region")])
    assert by_text[0]["title"].startswith("a long private note")


def test_normalize_skips_blank_entries():
    out = _normalize_user_sources(
        [
            UserSourceInput(),
            UserSourceInput(title="   ", url="  ", text=""),
            UserSourceInput(title="kept"),
        ]
    )
    assert len(out) == 1
    assert out[0]["title"] == "kept"


def test_normalize_caps_count():
    items = [UserSourceInput(title=f"s{i}") for i in range(30)]
    assert len(_normalize_user_sources(items)) == 20


def test_normalize_truncates_fields():
    out = _normalize_user_sources(
        [UserSourceInput(title="t" * 500, url="u" * 900, text="x" * 900)]
    )
    assert len(out[0]["title"]) == 200
    assert len(out[0]["url"]) == 500
    assert len(out[0]["snippet"]) == 300


def test_user_source_lines_renders_title_and_text():
    lines = P._user_source_lines(
        [{"title": "Reuters", "snippet": "coalition talks stall", "url": "https://r.example"}]
    )
    assert lines == ["Reuters — coalition talks stall"]


def test_user_source_lines_falls_back_to_url():
    lines = P._user_source_lines([{"title": "", "snippet": "", "url": "https://only-url.example"}])
    assert lines == ["https://only-url.example"]


def test_user_source_lines_skips_empty():
    assert P._user_source_lines([{"title": "", "snippet": "", "url": ""}]) == []
    assert P._user_source_lines(None) == []


def test_user_sources_become_citations():
    sources = _normalize_user_sources([UserSourceInput(title="My note", text="private data")])
    cits = P._citations_from(sources)
    assert len(cits) == 1
    assert cits[0].title == "My note"
    assert cits[0].source == "USER"


def test_prediction_result_includes_user_sources():
    import asyncio

    # Force the offline math fallback so no Ollama/network is needed, and confirm
    # the user's sources still land in the persisted result's source list.
    orig = P._get_available_models
    P._get_available_models = lambda: []
    try:
        sources = _normalize_user_sources(
            [UserSourceInput(title="Private poll", text="incumbent leads")]
        )
        result = asyncio.run(
            P.generate_prediction_ollama(
                prompt="Spain snap election 29 November 2026",
                depth=1,
                context=["USER SCENARIO: test"],
                model="llama3",
                constraints=[],
                sources=sources,
                user_sources=sources,
            )
        )
    finally:
        P._get_available_models = orig

    assert result.model_used == "mathematical-fallback"
    assert len(result.sources) == 1
    assert result.sources[0].source == "USER"


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
