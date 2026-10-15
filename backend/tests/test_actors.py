"""Verification tests for actors (ROADMAP section B).

Run:  backend/venv/bin/python backend/tests/test_actors.py
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.services import actors as A
from psychohistory.services.predictor import _parse_actors, _apply_actor_influence
from psychohistory.models import Event, PredictionNode


def test_mortality_hazard_increases_with_age():
    assert A.mortality_hazard(30) < A.mortality_hazard(60) < A.mortality_hazard(80)


def test_survival_bounds_and_decay():
    assert A.survival(60, 0.0) == 1.0
    assert 0.0 < A.survival(60, 10) < 1.0
    assert A.survival(60, 10) > A.survival(60, 20)
    # 10 years in the wrong direction (already dead-ish) is larger than current
    assert A.death_probability(80, 10) > A.death_probability(40, 10)


def test_death_probability_range():
    for age in (20, 50, 90):
        for yrs in (1, 5, 20):
            p = A.death_probability(age, yrs)
            assert 0.0 <= p <= 1.0


def test_leader_departure_monotone_in_horizon_and_age():
    p1 = A.leader_departure_probability(50, 3, 1)
    p4 = A.leader_departure_probability(50, 3, 4)
    p8 = A.leader_departure_probability(50, 3, 8)
    assert 0.0 < p1 < p4 < p8 < 1.0
    assert A.leader_departure_probability(75, 3, 4) > A.leader_departure_probability(40, 3, 4)


def test_leader_departure_stress_and_tenure():
    calm = A.leader_departure_probability(55, 2, 4, stress=0.0)
    angry = A.leader_departure_probability(55, 2, 4, stress=1.0)
    assert angry > calm
    newbie = A.leader_departure_probability(55, 0, 4)
    veteran = A.leader_departure_probability(55, 15, 4)
    assert veteran > newbie


def test_leader_departure_zero_horizon():
    assert A.leader_departure_probability(60, 5, 0) == 0.0


def test_expected_remaining_tenure():
    e = A.expected_remaining_tenure(60, 5, horizon=60)
    assert 0.0 < e < 60.0


def test_influence_weight_role():
    assert A.influence_weight(0.8, "leader") > A.influence_weight(0.8, "other")
    assert A.influence_weight(1.0, "head_of_state") == 1.0  # clamped
    assert A.influence_weight(0.0, "leader") == 0.0
    assert A.influence_weight(0.5, "unknown_role") == A.influence_weight(0.5, "other")


def test_shift_probability_direction_and_bounds():
    up = A.shift_probability(0.5, 1.0, direction=1)
    down = A.shift_probability(0.5, 1.0, direction=-1)
    assert down < 0.5 < up
    for w in (0.0, 0.5, 1.0):
        for d in (-1, 1):
            p = A.shift_probability(0.9, w, direction=d)
            assert 0.0 <= p <= 1.0
    # zero weight is identity
    assert abs(A.shift_probability(0.37, 0.0, 1) - 0.37) < 1e-12
    # strength scales the effect
    assert A.shift_probability(0.5, 1.0, 1, strength=2.0) > A.shift_probability(0.5, 1.0, 1, strength=1.0)


def test_majority_threshold_and_winning():
    assert A.majority_threshold(350) == 176
    assert A.majority_threshold(9) == 5
    assert A.is_winning({"a": 100, "b": 76}, 176) is True
    assert A.is_winning({"a": 100, "b": 75}, 176) is False


def test_winning_and_minimal_coalitions():
    seats = {"A": 4, "B": 3, "C": 2}
    thr = 5
    wins = A.winning_coalitions(seats, thr)
    assert len(wins) == 4  # {A,B},{A,C},{B,C},{A,B,C}
    minimal = A.minimal_winning_coalitions(seats, thr)
    assert len(minimal) == 3
    assert ["A", "B", "C"] not in minimal


def test_banzhaf_index():
    # a single dominant party is always pivotal; the small ones never are
    idx = A.banzhaf_index({"A": 5, "B": 2, "C": 2}, 5)
    assert abs(idx["A"] - 1.0) < 1e-9
    assert idx["B"] == 0.0 and idx["C"] == 0.0
    # a symmetric game gives equal power
    idx2 = A.banzhaf_index({"A": 4, "B": 3, "C": 2}, 5)
    assert abs(sum(idx2.values()) - 1.0) < 1e-3
    assert abs(idx2["A"] - idx2["B"]) < 1e-9
    assert A.banzhaf_index({}, 5) == {}


def test_succession_viability():
    with_heir = A.succession_viability(True, 0.8, 2, 0.0)
    no_heir = A.succession_viability(False, 0.0, 2, 0.0)
    assert with_heir > no_heir
    assert A.succession_viability(True, 0.8, 2, 0.9) < with_heir
    for args in [(True, 0.5, 0, 0), (False, 0, 20, 1), (True, 1.0, 0, 0)]:
        assert 0.0 <= A.succession_viability(*args) <= 1.0


def test_summarize_actors():
    acts = [
        {"name": "A", "role": "leader", "influence": 0.8, "age": 81, "tenure_years": 10},
        {"name": "B", "role": "other", "influence": 0.2, "age": 45},
    ]
    s = A.summarize_actors(acts, horizon=4)
    assert s["n_actors"] == 2
    assert 0.0 <= s["total_influence"] <= 1.0
    assert s["most_at_risk"] == "A"
    assert s["highest_departure_prob"] > 0.0


# --- integration with the predictor -----------------------------------------

def _ev(**kw):
    base = dict(name="e", description="d", probability=0.5, magnitude=0.5,
                trend="stable", year=2027)
    base.update(kw)
    return Event(**base)


def test_parse_actors_variants():
    raw = {"actors": [
        {"name": "Macron", "role": "leader", "influence": 0.8, "age": 48, "tenure_years": 7},
        "Someone Else",
        {"nombre": "Sanchez", "rol": "party", "influencia": 0.6},
        {"no_name": True},
    ]}
    out = _parse_actors(raw)
    names = [a.name for a in out]
    assert names == ["Macron", "Someone Else", "Sanchez"]
    assert out[0].age == 48 and out[0].role == "leader"
    assert out[2].role == "party"


def test_parse_actors_caps_at_six_and_empty():
    assert _parse_actors({}) == []
    raw = {"actors": [{"name": f"a{i}"} for i in range(10)]}
    assert len(_parse_actors(raw)) == 6


def test_apply_actor_influence_directions():
    root = PredictionNode(id="r", depth=1, branch_type="same", event=_ev(), stability_index=0.5)
    kids = [
        PredictionNode(id="s", depth=2, parent_id="r", branch_type="same",
                       event=_ev(actors=[{"name": "X", "role": "leader", "influence": 1.0}]), stability_index=0.5),
        PredictionNode(id="mx", depth=2, parent_id="r", branch_type="max",
                       event=_ev(actors=[{"name": "X", "role": "leader", "influence": 1.0}]), stability_index=0.5),
        PredictionNode(id="mn", depth=2, parent_id="r", branch_type="min",
                       event=_ev(actors=[{"name": "X", "role": "leader", "influence": 1.0}]), stability_index=0.5),
    ]
    # construct actors properly (Event.actors expects Actor instances via pydantic)
    from psychohistory.models import Actor
    for k in kids:
        k.event.actors = [Actor(name="X", role="leader", influence=1.0)]
    root.children = kids
    _apply_actor_influence([root])
    assert kids[0].stability_index > 0.5   # same -> up
    assert kids[1].stability_index > 0.5   # max  -> up
    assert kids[2].stability_index < 0.5   # min  -> down
    assert root.stability_index == 0.5     # root has no actors


def test_apply_actor_influence_no_actors_is_identity():
    root = PredictionNode(id="r", depth=1, branch_type="same", event=_ev(), stability_index=0.4)
    child = PredictionNode(id="c", depth=2, parent_id="r", branch_type="min",
                           event=_ev(), stability_index=0.3)
    root.children = [child]
    _apply_actor_influence([root])
    assert child.stability_index == 0.3


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
