"""Verification tests for the probabilistic-rigour module (ROADMAP section D).

Run:  backend/venv/bin/python backend/tests/test_uncertainty.py
"""
import inspect
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.services import uncertainty as U


def test_beta_cdf_bounds_and_monotone():
    assert U.beta_cdf(0.0, 2, 3) == 0.0
    assert U.beta_cdf(1.0, 2, 3) == 1.0
    assert U.beta_cdf(0.4, 2, 3) < U.beta_cdf(0.6, 2, 3)


def test_beta_cdf_uniform():
    # Beta(1,1) is uniform: CDF(x) == x
    for x in (0.1, 0.3, 0.5, 0.9):
        assert abs(U.beta_cdf(x, 1, 1) - x) < 1e-9


def test_beta_ppf_inverts_cdf():
    for a, b in [(2, 3), (5, 2), (1, 1), (8, 8)]:
        for p in (0.05, 0.25, 0.5, 0.75, 0.95):
            x = U.beta_ppf(p, a, b)
            assert abs(U.beta_cdf(x, a, b) - p) < 1e-6, (a, b, p, x)


def test_beta_mean_var():
    assert abs(U.beta_mean(2, 3) - 0.4) < 1e-12
    # uniform variance is 1/12
    assert abs(U.beta_var(1, 1) - 1.0 / 12.0) < 1e-12


def test_credible_interval_contains_mean():
    lo, hi = U.credible_interval(10, 10, 0.9)
    assert lo < 0.5 < hi
    assert 0.0 <= lo < hi <= 1.0
    # tighter with more evidence
    lo2, hi2 = U.credible_interval(100, 100, 0.9)
    assert (hi2 - lo2) < (hi - lo)


def test_beta_prior_and_update():
    a, b = U.beta_prior_from_base_rate(0.6, strength=10)
    assert abs(U.beta_mean(a, b) - 0.6) < 1e-12
    # posterior after 7/10 moves toward the data
    a2, b2 = U.beta_update(a, b, successes=7, failures=3)
    assert a2 == a + 7 and b2 == b + 3
    # strong prior keeps 9/10 from dominating entirely
    assert U.beta_mean(*U.beta_prior_from_base_rate(0.1, 100)) > 0.09


def test_normal_normal_update():
    mu, sigma = U.normal_normal_update(0.0, 1.0, 1.0, 1.0, n=4)
    assert 0.0 < mu < 1.0
    assert sigma < 1.0  # more data -> more precise
    # with a very tight prior the mean barely moves
    mu2, _ = U.normal_normal_update(0.0, 0.001, 5.0, 1.0, n=1)
    assert abs(mu2) < 0.1


def test_sample_beta_and_summary():
    s = U.sample_beta(2, 5, 4000, seed=1)
    assert len(s) == 4000
    assert all(0.0 <= x <= 1.0 for x in s)
    summ = U.mc_summary(s, level=0.9)
    # empirical mean ~ 2/7, within tolerance
    assert abs(summ["mean"] - 2.0 / 7.0) < 0.03
    assert summ["sd"] > 0.0
    assert summ["p05"] < summ["p50"] < summ["p95"]


def test_sample_beta_is_reproducible():
    a = U.sample_beta(3, 3, 100, seed=42)
    b = U.sample_beta(3, 3, 100, seed=42)
    assert a == b


def test_mc_summary_empty():
    s = U.mc_summary([])
    assert s["n"] == 0 and s["mean"] == 0.0


def test_propagate():
    samples = U.sample_beta(5, 5, 3000, seed=2)
    out = U.propagate(lambda x: x * x, samples)
    assert 0.0 <= out["p05"] <= out["p95"] <= 1.0
    assert out["mean"] > 0.0


def test_percentile():
    xs = [1, 2, 3, 4, 5]
    assert U.percentile(xs, 0.0) == 1
    assert U.percentile(xs, 1.0) == 5
    assert U.percentile(xs, 0.5) == 3


def test_log_odds_pool_sharper_than_linear():
    probs = [0.8, 0.9]
    lo = U.log_odds_pool(probs)
    lin = U.linear_pool(probs)
    assert lo > lin  # log-odds pushes above the arithmetic mean
    assert abs(lin - 0.85) < 1e-12


def test_pools_bounds_and_weights():
    assert U.log_odds_pool([0.5]) == 0.5
    assert U.linear_pool([]) == 0.5
    # weight fully on one member returns that member
    assert abs(U.linear_pool([0.2, 0.9], [0.0, 1.0]) - 0.9) < 1e-12
    for p in ([0.0, 1.0], [0.3, 0.7, 0.9]):
        assert 0.0 <= U.aggregate(p) <= 1.0
        assert 0.0 <= U.aggregate(p, "linear") <= 1.0


def test_ensemble_spread():
    assert U.ensemble_spread([0.5]) == 0.0
    assert U.ensemble_spread([0.2, 0.8]) > U.ensemble_spread([0.45, 0.55])


def test_markov_step_and_n():
    P = [[0.9, 0.1], [0.2, 0.8]]
    d = [1.0, 0.0]
    d1 = U.markov_step(d, P)
    assert abs(sum(d1) - 1.0) < 1e-12
    assert abs(d1[0] - 0.9) < 1e-12
    d5 = U.markov_n(d, P, 5)
    assert abs(sum(d5) - 1.0) < 1e-9


def test_stationary_distribution():
    P = [[0.9, 0.1], [0.2, 0.8]]
    pi = U.stationary_distribution(P)
    # check pi P = pi
    chk = U.markov_step(pi, P)
    assert max(abs(a - b) for a, b in zip(pi, chk)) < 1e-6
    # theoretical stationary for this chain: (2/3, 1/3)
    assert abs(pi[0] - 2.0 / 3.0) < 1e-4


def test_hmm_forward_and_viterbi():
    # two regimes; observations strongly identify the state
    pi = [0.5, 0.5]
    A = [[0.7, 0.3], [0.3, 0.7]]
    B = [[0.9, 0.1], [0.1, 0.9]]
    obs = [0, 0, 0, 1, 1]
    fwd = U.hmm_forward(pi, A, B, obs)
    assert fwd["log_likelihood"] < 0.0  # log-prob is negative
    assert abs(sum(fwd["alpha"]) - 1.0) < 1e-9
    path = U.viterbi(pi, A, B, obs)
    assert path == [0, 0, 0, 1, 1]


def test_sensitivity_ranks_parameters():
    def fn(p):
        return p["a"] * 2.0 + p["b"] * 0.1

    s = U.one_at_a_time(fn, {"a": 1.0, "b": 1.0}, rel=0.1)
    assert set(s) == {"a", "b"}
    # 'a' has 20x the leverage of 'b'
    assert s["a"] > s["b"]


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
