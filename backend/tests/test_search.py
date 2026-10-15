"""Tests for retrieval hygiene: dedup, rerank, chunking (ROADMAP section F).

Run:  backend/venv/bin/python backend/tests/test_search.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.services.search import (
    chunk_text,
    dedup_and_rerank,
    _keywords,
    _normalize_key,
    _has_number,
)


def test_chunk_text_short_passthrough():
    assert chunk_text("hello world") == ["hello world"]
    assert chunk_text("   ") == []


def test_chunk_text_splits_long():
    text = ". ".join(f"sentence number {i}" for i in range(40)) + "."
    chunks = chunk_text(text, max_len=80)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)  # sentence-aware, near the limit
    assert "sentence" in chunks[0]


def test_keywords_drop_stopwords_and_short():
    kw = _keywords("The economy of Spain will grow because of the recovery")
    assert "the" not in kw and "will" not in kw
    assert "economy" in kw and "spain" in kw and "recovery" in kw


def test_normalize_key_case_and_punct():
    a = _normalize_key("Spain's Economy: Growth!")
    b = _normalize_key("spains economy growth")
    assert a == b


def test_dedup_removes_near_duplicates():
    items = [
        "Spain economy grows strongly this year",
        "Spain economy grows strongly this year",  # exact dup
        "Portugal exports rise sharply",
    ]
    out = dedup_and_rerank(items, "Spain economy")
    assert len(out) == 2


def test_rerank_orders_by_relevance():
    items = [
        "Completely unrelated story about football",
        "Spain economy and elections dominate the news",
        "Weather forecast for the weekend",
    ]
    out = dedup_and_rerank(items, "Spain economy elections")
    assert "Spain economy" in out[0]


def test_rerank_empty():
    assert dedup_and_rerank([], "x") == []
    assert dedup_and_rerank(["  ", ""], "x") == []


def test_rerank_limit():
    items = [f"topic {i} about economy growth" for i in range(10)]
    out = dedup_and_rerank(items, "economy", limit=3)
    assert len(out) == 3


def test_has_number():
    assert _has_number("wins 176 seats") is True
    assert _has_number("no digits here") is False


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
