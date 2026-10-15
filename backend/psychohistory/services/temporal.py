r"""Temporal dynamics and rare-event mathematics for the Psychohistory engine.

This module is pure (standard library only) and side-effect free, so it can be
unit-tested deterministically without Ollama, MongoDB or the network.

Scope (ROADMAP section A)
-------------------------
The base predictor is *time-blind*: it produces branch probabilities but has no
notion of *when* something happens, of the fact that the longer an event has not
occurred the more/less likely it becomes, of recurring events (renewal), of
event clustering (self-excitation) or of heavy-tailed impact (black swans).
This module supplies that layer.

1. Onset timing  -> Weibull survival / hazard
------------------------------------------------
Let T >= 0 be the (random) time from "now" until an event occurs. Model T with a
Weibull distribution with shape ``k > 0`` and scale ``lambda > 0``:

    cumulative hazard   H(t)   = (t / lambda) ** k
    survival            S(t)   = exp(-H(t))
    density             f(t)   = h(t) * S(t)
    hazard              h(t)   = (k / lambda) * (t / lambda) ** (k - 1)
    CDF                 F(t)   = 1 - S(t)

Interpretation of the shape ``k``:
    k > 1  increasing hazard  -> "overdue" effect (the longer it hasn't
                                 happened, the more imminent it becomes)
    k = 1  constant hazard    -> memoryless (Poisson / exponential)
    k < 1  decreasing hazard   -> risk decays the longer calm persists

Conditional ("overdue") probability of occurrence in a window (t, t + dt],
given no occurrence by t:

    P(t < T <= t+dt | T > t) = 1 - S(t+dt)/S(t)
                             = 1 - exp(-(H(t+dt) - H(t)))

For k > 1 this is increasing in the age t (holding dt fixed); for k = 1 it is
constant (= 1 - exp(-dt/lambda)); for k < 1 it decreases.

Calibrating the scale from a horizon probability: given a prior occurrence
probability ``p`` within a horizon ``H`` (H = F^{-1} equivalent), choose lambda
so that S(H) = 1 - p, i.e.

    lambda = H / (-ln(1 - p)) ** (1 / k)

so ``p`` in (0, 1) is required. The median and mean onset are

    median = lambda * (ln 2) ** (1/k)
    mean   = lambda * Gamma(1 + 1/k)          (finite for all k > 0)

2. Renewal (recurring events)
------------------------------
For events that repeat (elections, congresses, censuses), the residual time to
the next occurrence is *not* the unconditional S. The renewal function

    m(t) = E[N(t)] = sum_{n>=1} F^{*n}(t)

counts expected renewals by time t. It satisfies the renewal equation

    m(t) = F(t) + \int_0^t m(t - s) f(s) ds

which we solve numerically by Euler discretisation
(``renewal_function``). As t -> infinity, m(t) ~ t / mean(T) (renewal theorem),
which we use as a test. The age-conditioned probability of recurrence in
(age, age + dt] is the hazard-style conditional above.

3. Self-excitation / clustering -> Hawkes process
-------------------------------------------------
Related events cluster (a shock raises the short-term rate of follow-on events):

    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))

with background rate ``mu > 0``, jump ``alpha >= 0`` and decay ``beta > 0``. The
branching ratio ``n = alpha / beta`` must satisfy n < 1 for a stationary process
(each event triggers, on average, n additional events).

4. Heavy tails / black swans -> Pareto severity
-----------------------------------------------
Impact (severity) Y is modelled as Pareto:

    P(Y > y) = (y_m / y) ** alpha,   y >= y_m > 0
    f(y)     = alpha * y_m**alpha * y**(-alpha - 1)
    mean     = alpha * y_m / (alpha - 1)   (finite iff alpha > 1)
    q-th quantile = y_m * (1 - q) ** (-1/alpha)

The tail index ``alpha`` is estimated from data by the Hill estimator on the
upper order statistics (``hill_estimator``). alpha < 2 => infinite variance
("black swan"); alpha = 2 => borderline; alpha > 2 => finite variance.

5. Risk combination
--------------------
Rare-but-huge should be able to dominate. We expose

    risk_score = P(occurs in horizon) * E[severity] * magnitude_scale

and an expected-shortfall style exceedance measure, rather than raw probability.

All mapping constants below (shape per trend, tail index per trend) are explicit,
documented and intended to be *calibrated by backtesting* (see
``backend/backtest``) rather than treated as ground truth.
"""

import math
from typing import Dict, List, Optional, Sequence

_EPS = 1e-9

# --- trend -> hazard shape k -------------------------------------------------
# Documented priors; calibrate via backtesting.
SHAPE_BY_TREND: Dict[str, float] = {
    "rising": 1.5,    # increasing hazard -> overdue effect
    "stable": 1.0,    # memoryless
    "declining": 0.7,  # decreasing hazard -> calm begets calm
    "volatile": 1.2,   # mildly increasing hazard
}

# --- trend -> Pareto tail index (severity) ----------------------------------
TAIL_ALPHA_BY_TREND: Dict[str, float] = {
    "stable": 4.0,     # light tail, finite variance
    "rising": 3.0,
    "declining": 3.5,
    "volatile": 2.0,   # heavy tail, borderline infinite variance
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# 1. Weibull survival / hazard
# ---------------------------------------------------------------------------

def cumulative_hazard(t: float, k: float, lam: float) -> float:
    """H(t) = (t / lambda) ** k, H(t) = 0 for t <= 0."""
    if t <= 0.0:
        return 0.0
    return (t / lam) ** k


def survival(t: float, k: float, lam: float) -> float:
    """S(t) = exp(-H(t)); S(t) = 1 for t <= 0."""
    return math.exp(-cumulative_hazard(t, k, lam))


def hazards(t: float, k: float, lam: float) -> float:
    """Weibull hazard h(t) = (k/lambda) * (t/lambda) ** (k - 1)."""
    if t <= 0.0:
        if k > 1.0:
            return 0.0
        if abs(k - 1.0) < _EPS:
            return 1.0 / lam
        return math.inf
    return (k / lam) * (t / lam) ** (k - 1.0)


def density(t: float, k: float, lam: float) -> float:
    """Weibull density f(t) = h(t) * S(t)."""
    return hazards(t, k, lam) * survival(t, k, lam)


def cdf(t: float, k: float, lam: float) -> float:
    """F(t) = 1 - S(t)."""
    return 1.0 - survival(t, k, lam)


def conditional_probability(t: float, dt: float, k: float, lam: float) -> float:
    """P(t < T <= t + dt | T > t) for the Weibull distribution."""
    if dt <= 0.0:
        return 0.0
    s_t = survival(t, k, lam)
    if s_t <= 0.0:
        return 1.0
    return _clamp(1.0 - survival(t + dt, k, lam) / s_t, 0.0, 1.0)


def overdue_factor(age: float, dt: float, k: float, lam: float) -> float:
    """Ratio of the conditional probability at ``age`` to that at age 0.

    > 1 means the event is "overdue" (hazard increasing), < 1 means it is
    "cooling off", == 1 means memoryless.
    """
    base = conditional_probability(0.0, dt, k, lam)
    if base <= 0.0:
        return 1.0
    return conditional_probability(age, dt, k, lam) / base


def scale_for_horizon(base_prob: float, horizon: float, k: float) -> float:
    """Choose lambda so that P(T <= horizon) == base_prob."""
    p = _clamp(base_prob, _EPS, 1.0 - _EPS)
    h = max(horizon, _EPS)
    return h / ((-math.log(1.0 - p)) ** (1.0 / k))


def weibull_mle(sample: Sequence[float]) -> tuple:
    """Maximum-likelihood fit of Weibull(k, lambda) to positive observations.

    The profile score for k,

        g(k) = (sum x_i^k ln x_i) / (sum x_i^k) - 1/k - mean(ln x_i),

    is strictly increasing, so we solve g(k) = 0 by bisection and then set

        lambda = (mean(x_i^k)) ** (1/k).

    Returns the pair ``(k, lambda)``. Raises on fewer than two positive samples.
    """
    data = [float(x) for x in sample if x > 0.0]
    n = len(data)
    if n < 2:
        raise ValueError("weibull_mle needs at least two positive samples")
    logs = [math.log(x) for x in data]
    mean_log = sum(logs) / n
    m = max(logs)

    def g(k: float) -> float:
        # Stable form: factor out exp(k*m) so exponents are <= 0.
        weights = [math.exp(k * (li - m)) for li in logs]
        s = sum(weights)
        return sum(weights[i] * logs[i] for i in range(n)) / s - 1.0 / k - mean_log

    lo, hi = 1e-4, 1e4
    if g(lo) > 0.0:
        lo = 1e-6
    if g(hi) < 0.0:
        hi = 1e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    k = 0.5 * (lo + hi)
    weights = [math.exp(k * (li - m)) for li in logs]
    lam = math.exp(m + math.log(sum(weights) / n) / k)
    return k, lam


def median_onset(k: float, lam: float) -> float:
    """Median of Weibull(k, lambda)."""
    return lam * (math.log(2.0) ** (1.0 / k))


def mean_onset(k: float, lam: float) -> float:
    """Mean of Weibull(k, lambda); finite for all k > 0."""
    return lam * math.gamma(1.0 + 1.0 / k)


# ---------------------------------------------------------------------------
# 2. Renewal
# ---------------------------------------------------------------------------

def renewal_function(t_max: float, k: float, lam: float, steps: int = 1000) -> List[float]:
    """Expected number of renewals m(t) on [0, t_max].

    Solves the renewal equation m(t) = F(t) + \\int_0^t m(t - s) f(s) ds by
    trapezoidal discretisation on a uniform grid. When f(0) != 0 (the
    exponential case, k == 1) the scheme is implicit in m_i and solved directly.

    The trapezoidal rule is second order (O(h^2)), unlike the naive left/right
    Riemann sum which converges only as O(h) with a constant that grows with t.
    Returns ``steps + 1`` values on a uniform grid.
    """
    if t_max <= 0.0 or steps <= 0:
        return [0.0]
    h = t_max / steps
    f_vals = [density(i * h, k, lam) for i in range(steps + 1)]
    m = [0.0] * (steps + 1)
    denom = 1.0 - 0.5 * h * f_vals[0]
    for i in range(1, steps + 1):
        integral = 0.0
        for j in range(1, i):
            integral += m[i - j] * f_vals[j]
        m[i] = (cdf(i * h, k, lam) + h * integral) / denom
    return m


# ---------------------------------------------------------------------------
# 3. Hawkes (self-exciting) process
# ---------------------------------------------------------------------------

def hawkes_intensity(t: float, event_times: Sequence[float], mu: float,
                     alpha: float, beta: float) -> float:
    """lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))."""
    acc = mu
    for ti in event_times:
        if ti < t:
            acc += alpha * math.exp(-beta * (t - ti))
    return acc


def branching_ratio(alpha: float, beta: float) -> float:
    """n = alpha / beta; the process is stationary only when n < 1."""
    return alpha / beta


# ---------------------------------------------------------------------------
# 4. Heavy tails (Pareto)
# ---------------------------------------------------------------------------

def pareto_survival(y: float, alpha: float, y_m: float = 1.0) -> float:
    """P(Y > y) = (y_m / y) ** alpha for y >= y_m, else 1."""
    if y < y_m:
        return 1.0
    return (y_m / y) ** alpha


def pareto_density(y: float, alpha: float, y_m: float = 1.0) -> float:
    """f(y) = alpha * y_m**alpha * y**(-alpha - 1), y >= y_m."""
    if y < y_m:
        return 0.0
    return alpha * (y_m ** alpha) * (y ** (-alpha - 1.0))


def pareto_mean(alpha: float, y_m: float = 1.0) -> float:
    """Mean of Pareto(alpha, y_m); infinite when alpha <= 1."""
    if alpha <= 1.0:
        return math.inf
    return alpha * y_m / (alpha - 1.0)


def pareto_quantile(q: float, alpha: float, y_m: float = 1.0) -> float:
    """q-th quantile, y_m * (1 - q) ** (-1/alpha), q in [0, 1)."""
    q = _clamp(q, 0.0, 1.0 - _EPS)
    return y_m * (1.0 - q) ** (-1.0 / alpha)


def hill_estimator(sample: Sequence[float], k: Optional[int] = None) -> float:
    """Hill tail-index estimator on the upper order statistics.

    Given a positive sample, use the k largest values ordered ascending
    X_(1) <= ... <= X_(n); the estimator is

        1/alpha_hat = (1/k) * sum_{i=0}^{k-1} ln(X_(n-i) / X_(n-k))

    ``k`` defaults to ~10% of the sample. Requires all values > 0.
    """
    data = sorted(float(x) for x in sample)
    n = len(data)
    if n < 3:
        raise ValueError("hill_estimator needs at least 3 positive samples")
    if data[0] <= 0.0:
        raise ValueError("hill_estimator requires strictly positive data")
    if k is None:
        k = max(2, int(0.1 * n))
    k = int(_clamp(k, 2, n - 1))
    threshold = data[n - k - 1]
    tail = data[n - k:]
    inv_alpha = sum(math.log(x / threshold) for x in tail) / k
    if inv_alpha <= 0.0:
        return math.inf
    return 1.0 / inv_alpha


# ---------------------------------------------------------------------------
# 5. Risk combination
# ---------------------------------------------------------------------------

def risk_score(occurrence_prob: float, severity_mean: float,
               magnitude: float = 1.0) -> float:
    """R = P(occurs) * E[severity] * magnitude_scale (>= 0)."""
    return max(0.0, occurrence_prob) * max(0.0, severity_mean) * max(0.0, magnitude)


def expected_shortfall(alpha: float, y_m: float = 1.0, tail_prob: float = 0.05) -> float:
    """Pareto expected shortfall E[Y | Y > quantile(1 - tail_prob)]."""
    if alpha <= 1.0:
        return math.inf
    q = pareto_quantile(1.0 - tail_prob, alpha, y_m)
    return alpha * q / (alpha - 1.0)


# ---------------------------------------------------------------------------
# High-level assessment
# ---------------------------------------------------------------------------

def assess(base_prob: float, trend: str, magnitude: float, horizon: float,
           age: float = 0.0, dt: Optional[float] = None,
           tail_alpha: Optional[float] = None) -> Dict[str, float]:
    """Combine all temporal ingredients for one event.

    ``base_prob`` is the prior probability of occurrence within ``horizon``;
    ``age`` is time already elapsed since the process last "armed"/occurred;
    ``dt`` is the evaluation window for the conditional/overdue statistics
    (defaults to one year, or the horizon if shorter).
    """
    k = SHAPE_BY_TREND.get(trend, 1.0)
    lam = scale_for_horizon(base_prob, horizon, k)
    alpha = float(tail_alpha) if tail_alpha is not None else TAIL_ALPHA_BY_TREND.get(trend, 3.0)
    if dt is None:
        dt = min(1.0, horizon) if horizon > 0 else 1.0

    sev_mean = pareto_mean(alpha, 1.0)
    occ = cdf(horizon, k, lam)

    return {
        "shape_k": round(k, 4),
        "scale_lambda": round(lam, 6),
        "occurrence_prob": round(occ, 6),
        "survival_horizon": round(survival(horizon, k, lam), 6),
        "hazard_now": round(hazards(max(age, _EPS), k, lam), 6),
        "median_onset": round(median_onset(k, lam), 4),
        "mean_onset": round(mean_onset(k, lam), 4),
        "conditional_next": round(conditional_probability(age, dt, k, lam), 6),
        "overdue_factor": round(overdue_factor(age, dt, k, lam), 4),
        "tail_alpha": round(alpha, 4),
        "severity_mean": round(sev_mean, 4) if math.isfinite(sev_mean) else math.inf,
        "tail_es": round(expected_shortfall(alpha, 1.0), 4) if alpha > 1.0 else math.inf,
        "risk_score": round(risk_score(occ, min(sev_mean, 100.0), 1.0 + magnitude), 4),
    }
