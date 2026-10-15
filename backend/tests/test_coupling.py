"""Verification tests for the coupling / cascades module (ROADMAP section C).

Run:  backend/venv/bin/python backend/tests/test_coupling.py
"""
import inspect
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.services import coupling as C


def test_conditional_probabilities_marginal_sum():
    r = C.conditional_probabilities(0.4, 0.6, 0.5)
    # P if parent = 0.4 + 0.5*0.6 = 0.7 ; if not = 0.4 - 0.5*0.4 = 0.2
    assert abs(r["p_if_parent"] - 0.7) < 1e-12
    assert abs(r["p_if_not_parent"] - 0.2) < 1e-12
    # marginal = q*p_if + (1-q)*p_if_not = 0.6*0.7 + 0.4*0.2 = 0.5
    assert abs(r["marginal"] - (0.6 * 0.7 + 0.4 * 0.2)) < 1e-12


def test_marginal_closed_form():
    # marginal == p + c(q - p)
    for p in (0.1, 0.4, 0.9):
        for q in (0.05, 0.5, 0.95):
            for c in (-0.6, 0.0, 0.35, 0.6):
                got = C.conditional_probabilities(p, q, c)["marginal"]
                exp = max(0.0, min(1.0, p + c * (q - p)))
                assert abs(got - exp) < 1e-12, (p, q, c, got, exp)


def test_monotonic_in_parent():
    for c in (0.3, -0.3):
        lo = C.conditional_probabilities(0.5, 0.2, c)["marginal"]
        hi = C.conditional_probabilities(0.5, 0.8, c)["marginal"]
        if c > 0:
            assert hi > 0.5 > lo
        else:
            assert hi < 0.5 < lo


def test_bounds_never_violated():
    for p in (0.0, 0.01, 0.99, 1.0):
        for c in (-1.0, -0.9, 0.9, 1.0):
            r = C.conditional_probabilities(p, 0.5, c)
            for k in r:
                assert -1e-12 <= r[k] <= 1 + 1e-12, (p, c, k, r[k])


def test_zero_coupling_is_independent():
    r = C.conditional_probabilities(0.37, 0.9, 0.0)
    assert abs(r["marginal"] - 0.37) < 1e-12


def test_coupling_coefficient_lookup():
    assert C.coupling_coefficient("same") > 0
    assert C.coupling_coefficient("max") > C.coupling_coefficient("same")
    assert C.coupling_coefficient("min") < 0
    assert C.coupling_coefficient("same", "prevents") < 0
    assert C.coupling_coefficient("same", "causes") > C.coupling_coefficient("same", "enables")


def test_adjust_for_parent_default_branch():
    # min branch (%negative) should drop the child below its standalone value
    # when the parent is likely.
    assert C.adjust_for_parent(0.5, 0.9, "min") < 0.5
    # max branch should raise it when the parent is likely.
    assert C.adjust_for_parent(0.5, 0.9, "max") > 0.5


def test_noisy_or():
    # single cause with full weight == its probability
    assert abs(C.noisy_or([(0.3, 1.0)]) - 0.3) < 1e-12
    # two independent causes: 1 - 0.7*0.8 = 0.44
    assert abs(C.noisy_or([(0.3, 1.0), (0.2, 1.0)]) - 0.44) < 1e-12
    # a zero-weight cause is ignored
    assert abs(C.noisy_or([(0.3, 1.0), (0.9, 0.0)]) - 0.3) < 1e-12
    # empty => no effect
    assert C.noisy_or([]) == 0.0
    # monotone in adding causes
    assert C.noisy_or([(0.3, 1.0), (0.2, 1.0)]) > C.noisy_or([(0.3, 1.0)])


def test_noisy_and():
    assert abs(C.noisy_and([(0.5, 1.0), (0.5, 1.0)]) - 0.25) < 1e-12
    assert C.noisy_and([(0.9, 1.0), (0.9, 1.0)]) < C.noisy_and([(0.9, 1.0)])


def test_sir_reduces_to_expected_when_betas_valid():
    out = C.sir_simulate(beta=0.4, gamma=0.1, steps=2000)
    assert abs(out["R0"] - 4.0) < 1e-9
    assert out["final_infected"] > 0.9  # R0=4 -> almost everyone


def test_sir_below_threshold_no_outbreak():
    out = C.sir_simulate(beta=0.05, gamma=0.2, steps=2000)  # R0 = 0.25
    # with a tiny seed the epidemic should not take off
    assert out["final_infected"] < 0.05


def test_sir_final_size_matches_simulation():
    R0 = 2.5
    z = C.sir_final_size(R0)
    assert abs(z - (1.0 - math.exp(-R0 * z))) < 1e-8
    assert 0.0 < z < 1.0
    assert C.sir_final_size(0.9) == 0.0


def test_check_conservation():
    assert C.check_conservation([100, 150, 100], 350)["ok"] is True
    assert C.check_conservation([100, 150, 100], 349)["ok"] is False


def test_clamp_to_total():
    out = C.clamp_to_total([1, 1, 2], 100)
    assert abs(sum(out) - 100) < 1e-9
    assert abs(out[2] - 50) < 1e-9  # keeps proportions
    # all-zero -> uniform split
    out2 = C.clamp_to_total([0, 0, 0], 90)
    assert all(abs(x - 30) < 1e-9 for x in out2)


def test_propagate_coupling_uses_updated_parent():
    nodes = [
        {"id": "root", "parent_id": None, "branch_type": "same", "p": 0.9},
        {"id": "a", "parent_id": "root", "branch_type": "max", "p": 0.2},
        {"id": "b", "parent_id": "a", "branch_type": "max", "p": 0.2},
    ]
    C.propagate_coupling(nodes)
    # root unchanged. a = 0.2 + 0.6*(0.9-0.2) = 0.62.
    # b is coupled to the *already-updated* a: 0.2 + 0.6*(0.62-0.2) = 0.452.
    # If the pass were not sequential, b would have stayed near 0.2.
    assert nodes[0]["p"] == 0.9
    assert abs(nodes[1]["p"] - 0.62) < 1e-9
    assert abs(nodes[2]["p"] - 0.452) < 1e-9


def test_propagate_no_coupling_is_identity():
    nodes = [
        {"id": "root", "parent_id": None, "branch_type": "same", "p": 0.9},
        {"id": "a", "parent_id": "root", "branch_type": "same", "p": 0.2},
    ]
    C.propagate_coupling(nodes, coupling_by_branch={"same": 0.0})
    assert abs(nodes[1]["p"] - 0.2) < 1e-12


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
