"""Verification tests for the temporal / rare-event mathematics.

Pure standard library, runnable without pytest:

    backend/venv/bin/python backend/tests/test_temporal.py
"""
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.services import temporal as T


def test_survival_properties():
    k, lam = 1.5, 3.0
    assert abs(T.survival(0.0, k, lam) - 1.0) < 1e-12
    assert abs(T.survival(-5.0, k, lam) - 1.0) < 1e-12
    assert T.survival(1000.0, k, lam) < 1e-6
    prev = 1.0
    for t in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        s = T.survival(t, k, lam)
        assert 0.0 <= s <= 1.0
        assert s <= prev + 1e-12
        prev = s


def test_cdf_density_consistency():
    k, lam = 1.3, 2.0
    for t in [0.1, 1.0, 3.0, 8.0]:
        assert abs(T.cdf(t, k, lam) - (1.0 - T.survival(t, k, lam))) < 1e-12
        assert abs(T.density(t, k, lam) - T.hazards(t, k, lam) * T.survival(t, k, lam)) < 1e-12


def test_density_integrates_to_one():
    k, lam = 1.6, 2.5
    n = 200000
    hi = 60.0
    h = hi / n
    total = 0.0
    for i in range(n):
        total += T.density((i + 0.5) * h, k, lam) * h
    assert abs(total - 1.0) < 1e-3, total


def test_cumulative_hazard_equals_minus_log_survival():
    k, lam = 0.8, 4.0
    for t in [0.5, 2.0, 7.0]:
        assert abs(T.cumulative_hazard(t, k, lam) + math.log(T.survival(t, k, lam))) < 1e-12


def test_hazard_shape_monotonicity():
    lam = 2.0
    h1a, h1b = T.hazards(1.0, 1.5, lam), T.hazards(2.0, 1.5, lam)
    assert h1b > h1a, "k > 1 debe tener hazard creciente"
    h0a, h0b = T.hazards(1.0, 0.7, lam), T.hazards(2.0, 0.7, lam)
    assert h0b < h0a, "k < 1 debe tener hazard decreciente"
    assert abs(T.hazards(1.0, 1.0, lam) - T.hazards(5.0, 1.0, lam)) < 1e-12, "k = 1 constante"


def test_hazard_nonnegative():
    for k in [0.5, 1.0, 1.5, 3.0]:
        for t in [0.0, 0.01, 1.0, 5.0]:
            assert T.hazards(t, k, 2.0) >= 0.0


def test_conditional_probability_bounds_and_monotonicity():
    # k > 1 -> overdue: conditional probability increases with age
    k, lam, dt = 1.5, 3.0, 1.0
    p0 = T.conditional_probability(0.0, dt, k, lam)
    p1 = T.conditional_probability(1.0, dt, k, lam)
    p3 = T.conditional_probability(3.0, dt, k, lam)
    assert p0 < p1 < p3
    for p in (p0, p1, p3):
        assert 0.0 <= p <= 1.0
    # k = 1 -> memoryless: constant
    c0 = T.conditional_probability(0.0, dt, 1.0, lam)
    c9 = T.conditional_probability(9.0, dt, 1.0, lam)
    assert abs(c0 - c9) < 1e-12
    # k < 1 -> cooling: decreasing
    d0 = T.conditional_probability(0.0, dt, 0.7, lam)
    d3 = T.conditional_probability(3.0, dt, 0.7, lam)
    assert d3 < d0


def test_conditional_matches_closed_form():
    k, lam, t, dt = 1.4, 2.2, 1.7, 0.6
    expected = 1.0 - math.exp(-(T.cumulative_hazard(t + dt, k, lam) - T.cumulative_hazard(t, k, lam)))
    got = T.conditional_probability(t, dt, k, lam)
    assert abs(got - expected) < 1e-12


def test_overdue_factor():
    k, lam, dt = 1.5, 3.0, 1.0
    assert T.overdue_factor(0.0, dt, k, lam) == 1.0
    assert T.overdue_factor(3.0, dt, k, lam) > 1.0
    assert T.overdue_factor(3.0, dt, 0.7, lam) < 1.0
    assert abs(T.overdue_factor(5.0, dt, 1.0, lam) - 1.0) < 1e-9


def test_scale_for_horizon():
    for p in [0.1, 0.5, 0.9]:
        for k in [0.7, 1.0, 1.5]:
            h = 5.0
            lam = T.scale_for_horizon(p, h, k)
            assert abs(T.cdf(h, k, lam) - p) < 1e-9


def test_onset_statistics():
    k, lam = 1.5, 3.0
    assert T.median_onset(k, lam) > 0.0
    assert T.mean_onset(k, lam) >= T.median_onset(k, lam)  # right-skewed for k>1
    # exponential: mean = lam
    assert abs(T.mean_onset(1.0, 4.0) - 4.0) < 1e-9


def test_weibull_mle_recovers_parameters():
    for k_true, lam_true in [(1.0, 2.0), (1.5, 3.0), (0.8, 1.5)]:
        n = 8000
        sample = []
        for i in range(n):
            u = (i + 0.5) / n
            sample.append(lam_true * (-math.log(1.0 - u)) ** (1.0 / k_true))
        k_hat, lam_hat = T.weibull_mle(sample)
        assert abs(k_hat - k_true) / k_true < 0.05, (k_true, k_hat)
        assert abs(lam_hat - lam_true) / lam_true < 0.05, (lam_true, lam_hat)


def test_weibull_mle_exponential_is_k_one():
    # exponential inter-arrivals -> Weibull shape ~ 1
    lam = 2.5
    n = 8000
    sample = [-lam * math.log(1.0 - (i + 0.5) / n) for i in range(n)]
    k_hat, lam_hat = T.weibull_mle(sample)
    assert abs(k_hat - 1.0) < 0.05, k_hat
    assert abs(lam_hat - lam) / lam < 0.05, lam_hat


def test_weibull_mle_rejects_tiny_sample():
    try:
        T.weibull_mle([1.0])
        assert False, "debía lanzar ValueError"
    except ValueError:
        pass


def test_renewal_function():
    k, lam = 1.2, 2.0
    m = T.renewal_function(40.0, k, lam, steps=600)
    assert m[0] == 0.0
    assert all(m[i] <= m[i + 1] + 1e-9 for i in range(len(m) - 1)), "m(t) no decreciente"
    assert all(v >= 0.0 for v in m)
    # renewal theorem: m(t) ~ t / mean(T) for large t
    mean = T.mean_onset(k, lam)
    approx = 40.0 / mean
    assert abs(m[-1] - approx) / approx < 0.2, (m[-1], approx)
    # exponential case: m(t) = t / lam exactly
    me = T.renewal_function(20.0, 1.0, 2.0, steps=400)
    assert abs(me[-1] - 20.0 / 2.0) < 0.2, me[-1]


def test_hawkes_intensity():
    events = [0.0, 1.0, 2.5]
    mu, alpha, beta = 0.1, 0.5, 1.0
    val = T.hawkes_intensity(3.0, events, mu, alpha, beta)
    assert val > mu, "debe haber excitación sobre el fondo"
    # intensity decays back towards mu far in the future
    far = T.hawkes_intensity(100.0, events, mu, alpha, beta)
    assert abs(far - mu) < 1e-12
    # an event exactly at t is not counted (strictly ti < t)
    assert abs(T.hawkes_intensity(0.0, [0.0], mu, alpha, beta) - mu) < 1e-12
    # additive excitation
    more = T.hawkes_intensity(3.0, events + [2.9], mu, alpha, beta)
    assert more > val


def test_branching_ratio():
    assert abs(T.branching_ratio(0.5, 1.0) - 0.5) < 1e-12
    assert T.branching_ratio(0.5, 1.0) < 1.0


def test_pareto_survival_and_density():
    alpha, y_m = 2.5, 1.0
    assert abs(T.pareto_survival(0.5, alpha, y_m) - 1.0) < 1e-12
    assert abs(T.pareto_survival(y_m, alpha, y_m) - 1.0) < 1e-12
    assert T.pareto_survival(10.0, alpha, y_m) < T.pareto_survival(2.0, alpha, y_m)
    # density integrates to 1
    n = 200000
    lo, hi = y_m, 1e6
    # log-spaced integration for the heavy tail
    total = 0.0
    ratio = (hi / lo) ** (1.0 / n)
    x = lo
    for _ in range(n):
        x2 = x * ratio
        total += 0.5 * (T.pareto_density(x, alpha, y_m) + T.pareto_density(x2, alpha, y_m)) * (x2 - x)
        x = x2
    assert abs(total - 1.0) < 1e-2, total


def test_pareto_mean():
    assert T.pareto_mean(2.0, 1.0) == 2.0
    assert T.pareto_mean(3.0, 2.0) == 3.0
    assert T.pareto_mean(1.0, 1.0) == math.inf
    assert T.pareto_mean(0.5, 1.0) == math.inf


def test_pareto_quantile():
    alpha, y_m = 2.5, 1.0
    q = T.pareto_quantile(0.5, alpha, y_m)
    # P(Y > q) should be ~0.5
    assert abs(T.pareto_survival(q, alpha, y_m) - 0.5) < 1e-9
    assert T.pareto_quantile(0.9, alpha, y_m) > q
    assert abs(T.pareto_quantile(0.0, alpha, y_m) - y_m) < 1e-6


def test_hill_estimator_recovers_alpha():
    alpha_true = 2.0
    y_m = 1.0
    # deterministic inverse-transform sample of Pareto(alpha, 1)
    n = 20000
    sample = [T.pareto_quantile((i + 0.5) / n, alpha_true, y_m) for i in range(n)]
    est = T.hill_estimator(sample, k=int(0.1 * n))
    assert abs(est - alpha_true) < 0.15, est


def test_risk_score():
    assert T.risk_score(0.0, 5.0) == 0.0
    assert T.risk_score(0.5, 2.0, 3.0) == 3.0
    assert T.risk_score(0.5, math.inf, 1.0) == math.inf
    assert T.risk_score(-1.0, 2.0) == 0.0


def test_expected_shortfall():
    alpha = 3.0
    es = T.expected_shortfall(alpha, 1.0, tail_prob=0.05)
    assert math.isfinite(es)
    assert T.expected_shortfall(1.0, 1.0) == math.inf


def test_assess_structure_and_bounds():
    out = T.assess(0.4, "rising", 0.8, horizon=5.0, age=2.0)
    for key in [
        "shape_k", "scale_lambda", "occurrence_prob", "survival_horizon",
        "hazard_now", "median_onset", "mean_onset", "conditional_next",
        "overdue_factor", "tail_alpha", "severity_mean", "tail_es", "risk_score",
    ]:
        assert key in out, key
    assert abs(out["occurrence_prob"] - 0.4) < 1e-4, out["occurrence_prob"]
    assert 0.0 <= out["survival_horizon"] <= 1.0
    assert out["overdue_factor"] > 1.0  # rising -> overdue
    assert out["tail_alpha"] == T.TAIL_ALPHA_BY_TREND["rising"]


def test_assess_stable_is_memoryless():
    out = T.assess(0.3, "stable", 0.5, horizon=4.0, age=3.0, dt=1.0)
    assert out["shape_k"] == 1.0
    assert abs(out["overdue_factor"] - 1.0) < 1e-9


if __name__ == "__main__":
    import inspect

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
