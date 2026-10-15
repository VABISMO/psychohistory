"""Walk-forward calibration backtest for the temporal (Weibull hazard) model.

Method
------
For each historical recurring-event series with event times t_0 < ... < t_N and
gaps g_i = t_i - t_{i-1}:

  * After observing gaps g_1..g_{i-1}, fit Weibull(k_i, lambda_i) by MLE.
  * Predict the *next* gap distribution and evaluate the probability integral
    transform (PIT) u_i = F(g_i; k_i, lambda_i).
  * Under a correctly specified model the {u_i} are Uniform(0, 1): we score with
    the Kolmogorov-Smirnov statistic and its asymptotic p-value.
  * Also produce binary forecasts P(next gap <= H) at several horizons H and
    score with the Brier score and a reliability table.

A synthetic series drawn from a *known* Weibull is included as a sanity check of
the whole pipeline (the PITs must be uniform).

Run:  backend/venv/bin/python backend/backtest/temporal_backtest.py
"""
import math
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psychohistory.services import temporal as T
from backtest.events_data import SERIES

MIN_FIT = 3  # minimum observed gaps before we start forecasting
HORIZONS = [1.0, 2.0, 3.0, 4.0, 5.0]
RELIABILITY_H = 3.0
REGULAR_K = 20.0  # shape above this => effectively a scheduled/atomic process


def _to_decimal_year(iso: str) -> float:
    y, m, d = (int(p) for p in iso.split("-"))
    start = date(y, 1, 1)
    end = date(y + 1, 1, 1)
    return y + (date(y, m, d) - start).days / (end - start).days


def gaps_years(iso_dates):
    ts = [_to_decimal_year(x) for x in iso_dates]
    return [ts[i] - ts[i - 1] for i in range(1, len(ts))]


def walk_forward(gaps):
    """Return (pits, forecasts) where forecasts = [(k, lam, realized_gap)]."""
    pits, forecasts = [], []
    for i in range(MIN_FIT, len(gaps)):
        k, lam = T.weibull_mle(gaps[:i])
        pits.append(T.cdf(gaps[i], k, lam))
        forecasts.append((k, lam, gaps[i]))
    return pits, forecasts


def ks_uniform(pits):
    """Two-sided Kolmogorov-Smirnov statistic vs Uniform(0,1) and its p-value."""
    n = len(pits)
    if n == 0:
        return 0.0, 1.0
    xs = sorted(pits)
    d = max(max((i + 1) / n - xs[i], xs[i] - i / n) for i in range(n))
    en = math.sqrt(n)
    lam = (en + 0.12 + 0.11 / en) * d
    p = 0.0
    for j in range(1, 101):
        p += ((-1) ** (j - 1)) * math.exp(-2.0 * j * j * lam * lam)
    return d, max(0.0, min(1.0, 2.0 * p))


def brier(pairs):
    """Brier score, climatological baseline and base rate for (p, y) pairs."""
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    n = len(pairs)
    bs = sum((p - y) ** 2 for p, y in pairs) / n
    rate = sum(y for _, y in pairs) / n
    base = sum((rate - y) ** 2 for _, y in pairs) / n
    return bs, base, rate


def reliability(forecasts, horizon, nbins=5):
    """(mean_predicted, observed_freq, count) per probability bin."""
    bins = [[] for _ in range(nbins)]
    for k, lam, g in forecasts:
        p = T.cdf(horizon, k, lam)
        y = 1.0 if g <= horizon else 0.0
        bins[min(nbins - 1, int(p * nbins))].append((p, y))
    out = []
    for b in bins:
        if b:
            mp = sum(p for p, _ in b) / len(b)
            of = sum(y for _, y in b) / len(b)
            out.append((mp, of, len(b)))
    return out


def pit_histogram(pits, nbins=10):
    """Counts of PITs per equal-width bin; flat == calibrated."""
    counts = [0] * nbins
    for u in pits:
        counts[min(nbins - 1, int(u * nbins))] += 1
    return counts


def synthetic_check(n=400, k_true=1.4, lam_true=3.0, seed=12345):
    """PIT uniformity on data generated from a known Weibull (LCG, no numpy)."""
    state = seed
    xs = []
    for _ in range(n):
        state = (1103515245 * state + 12345) % (2 ** 31)
        u = (state + 0.5) / (2 ** 31)
        xs.append(lam_true * (-math.log(1.0 - u)) ** (1.0 / k_true))
    pits, _ = walk_forward(xs)
    d, p = ks_uniform(pits)
    return d, p, len(pits)


def main():
    print("=" * 70)
    print("Temporal model backtest (walk-forward Weibull hazard)")
    print("=" * 70)

    all_pits, all_fc = [], []
    irr_pits, irr_fc = [], []
    for name, dates in SERIES.items():
        gaps = gaps_years(dates)
        pits, fc = walk_forward(gaps)
        k_full, lam_full = T.weibull_mle(gaps)
        scheduled = k_full >= REGULAR_K
        if not scheduled:
            irr_pits += pits
            irr_fc += fc
        all_pits += pits
        all_fc += fc
        d, p = ks_uniform(pits)
        tag = "SCHEDULED (atomic)" if scheduled else "irregular"
        print(f"\n[{name}]  {tag}  events={len(dates)}  n_forecasts={len(pits)}")
        print(f"  full fit : k={k_full:.3f}  lambda={lam_full:.3f}  "
              f"mean={T.mean_onset(k_full, lam_full):.2f}y  median={T.median_onset(k_full, lam_full):.2f}y")
        print(f"  PIT/KS   : D={d:.3f}  p={p:.3f}  (large p => calibrated)")

    print("\n" + "-" * 70)
    print(f"Pooled ALL        n_pit={len(all_pits)}  (mixes irregular + scheduled)")
    d, p = ks_uniform(all_pits)
    print(f"  PIT/KS D={d:.3f}  p={p:.3f}   <- dominated by the scheduled series")
    print(f"Pooled IRREGULAR  n_pit={len(irr_pits)}  (the population the model targets)")
    d, p = ks_uniform(irr_pits)
    print(f"  PIT/KS D={d:.3f}  p={p:.3f}")

    print("\nPIT histogram (irregular, 10 bins; flat => calibrated):")
    hist = pit_histogram(irr_pits)
    print("   " + " ".join(f"{c:>2}" for c in hist))

    print("\nBrier by horizon (irregular):  H  brier  baseline  obs_rate  skill")
    for h in HORIZONS:
        pairs = [(T.cdf(h, k, lam), 1.0 if g <= h else 0.0) for k, lam, g in irr_fc]
        bs, base, rate = brier(pairs)
        skill = 1.0 - bs / base if base > 0 else float("nan")
        print(f"    {h:>3.0f}y  {bs:.4f}  {base:.4f}  {rate:.2f}      {skill:+.3f}")

    print(f"\nReliability @ H={RELIABILITY_H:.0f}y (irregular):")
    print("  bin mean_pred  obs_freq  n")
    for mp, of, cnt in reliability(irr_fc, RELIABILITY_H):
        print(f"      {mp:.2f}      {of:.2f}     {cnt}")

    d, p, n = synthetic_check()
    print("\n" + "-" * 70)
    print(f"SYNTHETIC (known Weibull)  n_pit={n}  PIT/KS D={d:.3f}  p={p:.3f}")
    print("  (large p => pipeline recovers a correctly specified model)")

    print("\n" + "=" * 70)
    print("Findings:")
    print("  - Elections show k>1: the hazard rises with age (the 'overdue' effect).")
    print("  - PIT/KS is the meaningful calibration signal here; short-horizon Brier")
    print("    is degenerate when almost no event falls inside the horizon.")
    print("  - Near-deterministic cycles (US=4y) need a concentrated/atomic model;")
    print("    a plain Weibull is overconfident in the tails of the cycle.")
    print("=" * 70)


if __name__ == "__main__":
    main()
