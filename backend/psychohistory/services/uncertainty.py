"""Probabilistic rigour & calibration (ROADMAP section D).

Point probabilities hide their own uncertainty. This module adds the machinery to
express and propagate it, all in pure ``math``/stdlib (no numpy/scipy).

1. Beta distribution and Bayesian updating
------------------------------------------
Treat a binary probability as a Beta prior ``Beta(a, b)``. Observing ``s``
successes and ``f`` failures gives the conjugate posterior ``Beta(a+s, b+f)``.
We provide the density, CDF (regularised incomplete beta via a continued
fraction), and quantile (bisection), plus mean/variance and **credible
intervals**:

    P(L <= theta <= U) = level,   L = ppf((1-level)/2),  U = ppf(1-(1-level)/2)

A prior is built from a base rate ``p`` with a pseudo-sample strength ``n``:
``a = p*n``, ``b = (1-p)*n``.

2. Normal-Normal updating
-------------------------
For a noisy Gaussian observation, precisions add:
``tau' = tau0 + n*tau_obs``, ``mu' = (tau0*mu0 + n*tau_obs*x_bar) / tau'``.

3. Monte Carlo
--------------
``mc_summary`` returns mean, sd and percentiles from a sample; ``sample_beta``
draws from a Beta; ``propagate`` applies a function to each sample and summarises.

4. Ensembles
------------
``log_odds_pool`` averages forecasts in log-odds space (sharp and well behaved),
``linear_pool`` averages probabilities, and ``ensemble_spread`` reports their
disagreement as an uncertainty signal.

5. Markov chains / HMMs
-----------------------
``markov_step`` / ``markov_n`` / ``stationary_distribution`` for regime
transitions; ``hmm_forward`` and ``viterbi`` for hidden-regime filtering.

6. Sensitivity
--------------
``one_at_a_time`` perturbs each parameter and reports which ones move the output
most, so magic constants can be prioritised for justification.
"""
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_EPS = 1e-12
_FPMIN = 1e-300


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, float(p)))
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Beta distribution
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    MAXIT, EPS = 200, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta I_x(a, b) = P(X <= x) for Beta(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def beta_pdf(x: float, a: float, b: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    log_b = (
        (a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x)
        - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    )
    return math.exp(log_b)


def beta_ppf(p: float, a: float, b: float, tol: float = 1e-10,
             iters: int = 200) -> float:
    """Quantile function of Beta(a, b) by bisection on the CDF."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if beta_cdf(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def beta_mean(a: float, b: float) -> float:
    return a / (a + b)


def beta_var(a: float, b: float) -> float:
    s = a + b
    return a * b / (s * s * (s + 1.0))


def credible_interval(a: float, b: float, level: float = 0.9) -> Tuple[float, float]:
    """Equal-tailed credible interval for Beta(a, b)."""
    level = _clamp01(level)
    tail = (1.0 - level) / 2.0
    return beta_ppf(tail, a, b), beta_ppf(1.0 - tail, a, b)


# ---------------------------------------------------------------------------
# Bayesian updating
# ---------------------------------------------------------------------------

def beta_prior_from_base_rate(p: float, strength: float = 4.0) -> Tuple[float, float]:
    """Beta prior with mean ``p`` and pseudo-sample size ``strength``."""
    p = _clamp01(p)
    strength = max(_EPS, float(strength))
    return p * strength, (1.0 - p) * strength


def beta_update(a: float, b: float, successes: float, failures: float) -> Tuple[float, float]:
    """Conjugate Beta-Bernoulli posterior after observing counts."""
    return a + max(0.0, successes), b + max(0.0, failures)


def normal_normal_update(mu0: float, sigma0: float, x_bar: float,
                         sigma_obs: float, n: float = 1.0) -> Tuple[float, float]:
    """Conjugate Gaussian posterior after ``n`` observations with mean ``x_bar``."""
    tau0 = 1.0 / max(_EPS, sigma0 * sigma0)
    tau_obs = 1.0 / max(_EPS, sigma_obs * sigma_obs)
    tau = tau0 + n * tau_obs
    mu = (tau0 * mu0 + n * tau_obs * x_bar) / tau
    return mu, math.sqrt(1.0 / tau)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def sample_beta(a: float, b: float, n: int, seed: Optional[int] = None) -> List[float]:
    """Draw ``n`` samples from Beta(a, b) by two Gammas (Marsaglia-Tsang)."""
    import random

    rng = random.Random(seed)

    def gamma(shape: float) -> float:
        if shape < 1.0:
            u = rng.random()
            return gamma(shape + 1.0) * (u ** (1.0 / shape))
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = rng.gauss(0.0, 1.0)
            v = 1.0 + c * x
            if v <= 0:
                continue
            v = v ** 3
            u = rng.random()
            if u < 1.0 - 0.0331 * x ** 4:
                return d * v
            if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                return d * v

    out = []
    for _ in range(n):
        g1, g2 = gamma(a), gamma(b)
        out.append(g1 / (g1 + g2))
    return out


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of a sample."""
    if not values:
        return 0.0
    xs = sorted(values)
    if q <= 0:
        return xs[0]
    if q >= 1:
        return xs[-1]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def mc_summary(samples: Sequence[float], level: float = 0.9) -> Dict[str, float]:
    n = len(samples)
    if n == 0:
        return {"n": 0, "mean": 0.0, "sd": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    mean = sum(samples) / n
    var = sum((s - mean) ** 2 for s in samples) / max(1, n - 1)
    tail = (1.0 - _clamp01(level)) / 2.0
    return {
        "n": n,
        "mean": mean,
        "sd": math.sqrt(var),
        "p05": percentile(samples, tail),
        "p50": percentile(samples, 0.5),
        "p95": percentile(samples, 1.0 - tail),
    }


def propagate(fn: Callable[[float], float], samples: Sequence[float],
              level: float = 0.9) -> Dict[str, float]:
    """Apply ``fn`` to each Monte-Carlo sample and summarise the outputs."""
    return mc_summary([fn(s) for s in samples], level=level)


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

def log_odds_pool(probs: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """Average forecasts in log-odds space (sharp, respects extremes)."""
    if not probs:
        return 0.5
    ws = list(weights) if weights else [1.0] * len(probs)
    tot = sum(ws) or 1.0
    z = sum(w * _logit(p) for w, p in zip(ws, probs)) / tot
    return _clamp01(_sigmoid(z))


def linear_pool(probs: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """Weighted arithmetic mean of probabilities (conservative)."""
    if not probs:
        return 0.5
    ws = list(weights) if weights else [1.0] * len(probs)
    tot = sum(ws) or 1.0
    return _clamp01(sum(w * p for w, p in zip(ws, probs)) / tot)


def aggregate(probs: Sequence[float], method: str = "log_odds",
              weights: Optional[Sequence[float]] = None) -> float:
    if method == "linear":
        return linear_pool(probs, weights)
    return log_odds_pool(probs, weights)


def ensemble_spread(probs: Sequence[float]) -> float:
    """Sample standard deviation of the ensemble members (disagreement)."""
    n = len(probs)
    if n < 2:
        return 0.0
    m = sum(probs) / n
    return math.sqrt(sum((p - m) ** 2 for p in probs) / (n - 1))


# ---------------------------------------------------------------------------
# Markov chains / HMM
# ---------------------------------------------------------------------------

def markov_step(dist: Sequence[float], P: Sequence[Sequence[float]]) -> List[float]:
    n = len(dist)
    return [sum(dist[i] * P[i][j] for i in range(n)) for j in range(n)]


def markov_n(dist: Sequence[float], P: Sequence[Sequence[float]], n: int) -> List[float]:
    d = list(dist)
    for _ in range(max(0, n)):
        d = markov_step(d, P)
    return d


def stationary_distribution(P: Sequence[Sequence[float]], tol: float = 1e-12,
                            iters: int = 10000) -> List[float]:
    """Stationary distribution by repeated application from the uniform start."""
    n = len(P)
    d = [1.0 / n] * n
    for _ in range(iters):
        nd = markov_step(d, P)
        if max(abs(a - b) for a, b in zip(nd, d)) < tol:
            return nd
        d = nd
    return d


def hmm_forward(pi: Sequence[float], A: Sequence[Sequence[float]],
                B: Sequence[Sequence[float]], obs: Sequence[int]) -> Dict[str, object]:
    """Forward algorithm. ``B[s][o]`` is P(observation o | state s)."""
    n = len(pi)
    if not obs:
        return {"log_likelihood": 0.0, "alpha": [], "scales": []}
    alpha = [pi[s] * B[s][obs[0]] for s in range(n)]
    scales = [sum(alpha) or _EPS]
    alpha = [a / scales[0] for a in alpha]
    for o in obs[1:]:
        new = [0.0] * n
        for j in range(n):
            new[j] = B[j][o] * sum(alpha[i] * A[i][j] for i in range(n))
        s = sum(new) or _EPS
        scales.append(s)
        alpha = [a / s for a in new]
    loglik = sum(math.log(s) for s in scales)
    return {"log_likelihood": loglik, "alpha": alpha, "scales": scales}


def viterbi(pi: Sequence[float], A: Sequence[Sequence[float]],
            B: Sequence[Sequence[float]], obs: Sequence[int]) -> List[int]:
    """Most likely hidden-state path (log-space Viterbi)."""
    n = len(pi)
    if not obs:
        return []
    log = lambda x: math.log(max(x, _EPS))
    delta = [log(pi[s]) + log(B[s][obs[0]]) for s in range(n)]
    back = [[0] * n for _ in obs]
    for t in range(1, len(obs)):
        nd = [0.0] * n
        for j in range(n):
            best, arg = -1e18, 0
            for i in range(n):
                v = delta[i] + log(A[i][j])
                if v > best:
                    best, arg = v, i
            nd[j] = best + log(B[j][obs[t]])
            back[t][j] = arg
        delta = nd
    path = [max(range(n), key=lambda s: delta[s])]
    for t in range(len(obs) - 1, 0, -1):
        path.append(back[t][path[-1]])
    return list(reversed(path))


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

def one_at_a_time(fn: Callable[[Dict[str, float]], float],
                  params: Dict[str, float], rel: float = 0.1) -> Dict[str, float]:
    """Relative sensitivity of ``fn`` to each parameter (central difference).

    Sensitivities are normalised by the baseline output, so they are comparable
    across parameters of different scale. Larger = the output moves more.
    """
    base = fn(dict(params))
    denom_base = abs(base) if abs(base) > _EPS else 1.0
    out: Dict[str, float] = {}
    for k, v in params.items():
        delta = rel * (abs(v) if abs(v) > _EPS else 1.0)
        plus = dict(params)
        minus = dict(params)
        plus[k] = v + delta
        minus[k] = v - delta
        out[k] = round(abs(fn(plus) - fn(minus)) / (2.0 * delta) * (abs(v) if abs(v) > _EPS else 1.0) / denom_base, 6)
    return out
