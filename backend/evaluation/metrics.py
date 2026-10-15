"""Forecast evaluation metrics (ROADMAP section I).

Pure standard library. Every function scores a set of probabilistic forecasts of
binary outcomes: a forecast is a probability ``p in [0, 1]`` and the outcome is
``y in {0, 1}``.

Metrics provided
----------------
Proper scoring rules (lower is better):
    * ``brier_score``        : mean((p - y)^2)
    * ``log_loss``           : -mean(y ln p + (1-y) ln(1-p))  (clipped)
    * ``brier_skill_score``  : 1 - BS / BS_climatology  (>0 beats base rate)

Calibration & sharpness (a forecast is "good" only if both are good):
    * ``reliability_curve``  : (mean forecast, observed frequency, count) per bin
    * ``expected_calibration_error`` (ECE) and ``maximum_calibration_error`` (MCE)
    * ``brier_decomposition``: BS = reliability - resolution + uncertainty
    * ``resolution``         : how much the forecasts separate the outcomes
    * ``sharpness``          : variance of the forecasts (informational content)

Discrimination:
    * ``roc_auc``            : rank-based AUC with tie handling

Timing:
    * ``lead_time_errors``   : mean absolute error and bias of predicted vs
                               actual dates (in years)

All functions validate their inputs and raise ``ValueError`` on empty data.
"""
import math
from typing import List, Sequence, Tuple

_EPS = 1e-15


def _clip(p: float) -> float:
    return min(1.0 - _EPS, max(_EPS, float(p)))


def _validate(ps: Sequence[float], ys: Sequence[int]) -> Tuple[List[float], List[float]]:
    if len(ps) != len(ys):
        raise ValueError("ps and ys must have the same length")
    if len(ps) == 0:
        raise ValueError("empty forecast set")
    ps = [float(p) for p in ps]
    ys = [float(y) for y in ys]
    for p in ps:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"probability out of range: {p}")
    for y in ys:
        if y not in (0.0, 1.0):
            raise ValueError(f"outcome must be 0 or 1: {y}")
    return ps, ys


def base_rate(ys: Sequence[int]) -> float:
    """Climatological frequency of the positive class."""
    if len(ys) == 0:
        raise ValueError("empty outcome set")
    return sum(float(y) for y in ys) / len(ys)


# ---------------------------------------------------------------------------
# Proper scoring rules
# ---------------------------------------------------------------------------

def brier_score(ps: Sequence[float], ys: Sequence[int]) -> float:
    ps, ys = _validate(ps, ys)
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


def brier_skill_score(ps: Sequence[float], ys: Sequence[int]) -> float:
    """1 - BS / BS_ref with the reference being the base-rate forecast."""
    ps, ys = _validate(ps, ys)
    bs = brier_score(ps, ys)
    r = base_rate(ys)
    bs_ref = sum((r - y) ** 2 for y in ys) / len(ys)
    if bs_ref <= 0.0:
        return float("nan")
    return 1.0 - bs / bs_ref


def log_loss(ps: Sequence[float], ys: Sequence[int]) -> float:
    ps, ys = _validate(ps, ys)
    total = 0.0
    for p, y in zip(ps, ys):
        q = _clip(p)
        total += -(y * math.log(q) + (1.0 - y) * math.log(1.0 - q))
    return total / len(ps)


# ---------------------------------------------------------------------------
# Calibration / reliability
# ---------------------------------------------------------------------------

def reliability_curve(ps: Sequence[float], ys: Sequence[int],
                      n_bins: int = 10) -> List[Tuple[float, float, int]]:
    """Return (mean_forecast, observed_frequency, count) for each non-empty bin."""
    ps, ys = _validate(ps, ys)
    bins: List[List[Tuple[float, float]]] = [[] for _ in range(n_bins)]
    for p, y in zip(ps, ys):
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, y))
    out = []
    for b in bins:
        if b:
            mp = sum(p for p, _ in b) / len(b)
            of = sum(y for _, y in b) / len(b)
            out.append((mp, of, len(b)))
    return out


def expected_calibration_error(ps: Sequence[float], ys: Sequence[int],
                               n_bins: int = 10) -> float:
    ps, ys = _validate(ps, ys)
    n = len(ps)
    ece = 0.0
    for mp, of, cnt in reliability_curve(ps, ys, n_bins):
        ece += (cnt / n) * abs(mp - of)
    return ece


def maximum_calibration_error(ps: Sequence[float], ys: Sequence[int],
                              n_bins: int = 10) -> float:
    ps, ys = _validate(ps, ys)
    mce = 0.0
    for mp, of, _ in reliability_curve(ps, ys, n_bins):
        mce = max(mce, abs(mp - of))
    return mce


def resolution(ps: Sequence[float], ys: Sequence[int], n_bins: int = 10) -> float:
    """Resolution term: weighted variance of the observed frequencies around the
    base rate. Higher is better; a constant forecast has resolution 0."""
    ps, ys = _validate(ps, ys)
    n = len(ps)
    r = base_rate(ys)
    res = 0.0
    for _mp, of, cnt in reliability_curve(ps, ys, n_bins):
        res += (cnt / n) * (of - r) ** 2
    return res


def sharpness(ps: Sequence[float], ys: Sequence[int]) -> float:
    """Variance of the forecasts (mean squared deviation from the base rate)."""
    ps, ys = _validate(ps, ys)
    r = base_rate(ys)
    return sum((p - r) ** 2 for p in ps) / len(ps)


def brier_decomposition(ps: Sequence[float], ys: Sequence[int],
                        n_bins: int = 10) -> dict:
    """Murphy decomposition BS = reliability - resolution + uncertainty."""
    ps, ys = _validate(ps, ys)
    n = len(ps)
    r = base_rate(ys)
    reliability = 0.0
    for mp, of, cnt in reliability_curve(ps, ys, n_bins):
        reliability += (cnt / n) * (mp - of) ** 2
    return {
        "reliability": reliability,
        "resolution": resolution(ps, ys, n_bins),
        "uncertainty": r * (1.0 - r),
        "brier": brier_score(ps, ys),
    }


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------

def roc_auc(ps: Sequence[float], ys: Sequence[int]) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties = 0.5)."""
    ps, ys = _validate(ps, ys)
    pos = [p for p, y in zip(ps, ys) if y == 1.0]
    neg = [p for p, y in zip(ps, ys) if y == 0.0]
    if not pos or not neg:
        return float("nan")
    combined = sorted(zip(ps, ys), key=lambda t: t[0])
    ranks = {}
    i = 0
    rank_of = [0.0] * len(combined)
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            rank_of[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, (_, y) in zip(rank_of, combined) if y == 1.0)
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def lead_time_errors(predicted: Sequence[float], actual: Sequence[float]) -> dict:
    """MAE and bias (mean signed error) between predicted and actual dates."""
    if len(predicted) != len(actual) or len(predicted) == 0:
        raise ValueError("predicted and actual must be non-empty and equal length")
    errs = [float(p) - float(a) for p, a in zip(predicted, actual)]
    mae = sum(abs(e) for e in errs) / len(errs)
    bias = sum(errs) / len(errs)
    return {"mae": mae, "bias": bias, "n": len(errs)}


def pit_histogram(pits: Sequence[float], n_bins: int = 10) -> List[int]:
    """Counts per equal-width bin; flat == uniformly calibrated."""
    counts = [0] * n_bins
    for u in pits:
        if not (0.0 <= u <= 1.0):
            raise ValueError(f"PIT out of range: {u}")
        counts[min(n_bins - 1, int(u * n_bins))] += 1
    return counts


def summarize(ps: Sequence[float], ys: Sequence[int], n_bins: int = 10) -> dict:
    """Convenience bundle of the headline metrics."""
    return {
        "n": len(ps),
        "base_rate": round(base_rate(ys), 4),
        "brier": round(brier_score(ps, ys), 4),
        "brier_skill": round(brier_skill_score(ps, ys), 4),
        "log_loss": round(log_loss(ps, ys), 4),
        "ece": round(expected_calibration_error(ps, ys, n_bins), 4),
        "mce": round(maximum_calibration_error(ps, ys, n_bins), 4),
        "resolution": round(resolution(ps, ys, n_bins), 4),
        "sharpness": round(sharpness(ps, ys), 4),
        "auc": round(roc_auc(ps, ys), 4),
    }
