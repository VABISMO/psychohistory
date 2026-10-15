"""Tests for JSON-schema constrained decoding (ROADMAP section F).

Run:  backend/venv/bin/python backend/tests/test_schema.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.services import schema as S


def test_build_tree_schema_shape():
    sch = S.build_tree_schema(2, include_actors=True)
    assert sch["type"] == "object"
    assert set(sch["required"]) == {"summary", "events"}
    assert sch["properties"]["events"]["items"]["properties"]["level"]["maximum"] == 2
    assert "actors" in sch["properties"]["events"]["items"]["properties"]
    # minItems == 3 + 9 for depth 2
    assert sch["properties"]["events"]["minItems"] == 12


def test_build_tree_schema_without_actors():
    sch = S.build_tree_schema(1, include_actors=False)
    assert "actors" not in sch["properties"]["events"]["items"]["properties"]
    assert sch["properties"]["events"]["minItems"] == 3


def test_trend_enum():
    sch = S.build_tree_schema(1)
    assert sch["properties"]["events"]["items"]["properties"]["trend"]["enum"] == S.TRENDS


def _valid():
    return {
        "summary": "Something plausible happens.",
        "events": [
            {"level": 1, "name": "A", "description": "a happens", "trend": "rising"},
            {"level": 1, "name": "B", "description": "b happens", "trend": "stable"},
            {"level": 1, "name": "C", "description": "c happens", "trend": "volatile"},
            {"level": 2, "name": "D", "description": "d happens", "trend": "declining"},
            {"level": 2, "name": "E", "description": "e happens", "trend": "rising"},
            {"level": 2, "name": "F", "description": "f happens", "trend": "stable"},
            {"level": 2, "name": "G", "description": "g happens", "trend": "rising"},
            {"level": 2, "name": "H", "description": "h happens", "trend": "stable"},
            {"level": 2, "name": "I", "description": "i happens", "trend": "volatile"},
            {"level": 2, "name": "J", "description": "j happens", "trend": "declining"},
            {"level": 2, "name": "K", "description": "k happens", "trend": "rising"},
            {"level": 2, "name": "L", "description": "l happens", "trend": "stable"},
        ],
    }


def test_validate_accepts_valid():
    assert S.validate_flat_tree(_valid(), 2) == []


def test_validate_catches_problems():
    p = _valid()
    p["summary"] = ""
    assert any("summary" in e for e in S.validate_flat_tree(p, 2))

    p = _valid()
    p["events"] = "nope"
    assert any("not a list" in e for e in S.validate_flat_tree(p, 2))

    p = _valid()
    p["events"][0]["level"] = 9
    assert any("bad level" in e for e in S.validate_flat_tree(p, 2))

    p = _valid()
    p["events"][1]["name"] = ""
    assert any("missing name" in e for e in S.validate_flat_tree(p, 2))

    p = _valid()
    p["events"][2]["trend"] = "sideways"
    assert any("bad trend" in e for e in S.validate_flat_tree(p, 2))

    p = _valid()
    p["events"][3]["actors"] = {"oops": 1}
    assert any("actors is not a list" in e for e in S.validate_flat_tree(p, 2))


def test_validate_non_object_root():
    assert S.validate_flat_tree([1, 2, 3], 2) == ["root is not an object"]


def test_validate_empty_events():
    errs = S.validate_flat_tree({"summary": "x", "events": []}, 2)
    assert any("empty" in e for e in errs)


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
