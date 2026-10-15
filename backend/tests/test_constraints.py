"""Tests for the structured numeric constraint store (ROADMAP section E).

Run:  backend/venv/bin/python backend/tests/test_constraints.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.services import constraints as C


def test_seats_majority():
    assert C.seats_majority(350) == 176
    assert C.seats_majority(435) == 218
    assert C.seats_majority(100) == 51
    assert C.seats_majority(720) == 361


def test_find_matches_alias():
    hits = C.find("The Spanish Congress of Deputies votes today")
    assert any(k == "spain_congress" for k, _ in hits)

    hits = C.find("the US Senate must confirm the nominee")
    assert any(k == "us_senate" for k, _ in hits)

    assert C.find("a completely unrelated sentence") == []


def test_validate_seats():
    ok, _ = C.validate_seats(200, 350)
    assert ok
    bad, msg = C.validate_seats(400, 350)
    assert not bad and "350" in msg


def test_validate_share():
    assert C.validate_share(87.5)[0]
    assert not C.validate_share(120)[0]
    assert not C.validate_share(-1)[0]


def test_validate_number_uses_total():
    assert C.validate_number(300, "spain_congress")[0]
    assert not C.validate_number(400, "spain_congress")[0]


def test_enforce_text_clamps_seats():
    text, violations = C.enforce_text(
        "In the Congress of Deputies a party wins 400 seats in a landslide."
    )
    assert "400 seats" not in text
    assert "350 seats" in text
    assert violations


def test_enforce_text_clamps_percentage():
    text, violations = C.enforce_text("Support reaches an impossible 140% of voters.")
    assert "140%" not in text
    assert "100%" in text
    assert violations


def test_enforce_text_noop_without_constraint():
    original = "A party wins 900 seats somewhere unspecified."
    text, violations = C.enforce_text(original)
    assert text == original
    assert violations == []


def test_check_text_wrapper():
    assert C.check_text("Congress of Deputies with 999 seats") != []
    assert C.check_text("nothing risky here") == []


def test_get_unknown():
    assert C.get("does_not_exist") is None


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
