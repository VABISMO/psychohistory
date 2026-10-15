"""Actors as first-class entities (ROADMAP section B).

The base engine treats "who" as decoration inside a sentence. Here actors become
quantities: a leader can die, resign or be removed; a party can be pivotal in a
coalition; an actor's leverage can shift a branch probability.

1. Leader departure (Gompertz-Makeham + political risk)
------------------------------------------------------
Natural mortality uses the Gompertz-Makeham hazard

    mu(age) = a * exp(b * age) + c

(a: Makeham's constant floor, b: Gompertz growth, c: background). To this we add
an exit hazard that grows with tenure (leaders accumulate enemies and fatigue)
and is amplified by political ``stress``:

    h(t) = mu(age + t) + (base_exit + slope * (tenure + t)) * exp(k * stress)

The departure probability over a horizon is ``1 - exp(-integral h dt)``.

2. Influence and probability shifting
-------------------------------------
An actor carries an ``influence`` in [0, 1] and a role weight (a head of state
moves more than a mid-ranking official). Combined weight shifts a probability in
log-odds space:

    logit(p') = logit(p) + direction * strength * weight * LOGIT_SCALE

3. Coalitions (game theory)
---------------------------
Given parties and their seat counts, a coalition is winning when it reaches the
majority threshold. We provide winning coalitions, minimal winning coalitions,
and the **Banzhaf power index** (how often an actor is pivotal).

4. Succession
-------------
An orderly-succession probability from whether an heir exists, their strength,
and how long the incumbent has ruled.

Everything is pure ``math``/stdlib so it is testable without numpy or scipy.
"""
import itertools
import math
from typing import Dict, List, Optional, Sequence

_EPS = 1e-12

# Gompertz-Makeham defaults, chosen so annual mortality is ~3% at 60 and ~18%
# at 80 (roughly developed-world adult mortality).
GM_A = 0.0002
GM_B = 0.085
GM_C = 0.0005

ROLE_WEIGHTS: Dict[str, float] = {
    "head_of_state": 1.10,
    "leader": 1.00,
    "prime_minister": 1.00,
    "general": 0.90,
    "central_banker": 0.80,
    "party": 0.70,
    "ceo": 0.70,
    "union": 0.60,
    "institution": 0.60,
    "other": 0.50,
}

LOGIT_SCALE = 1.2


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, float(p)))
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Mortality / survival
# ---------------------------------------------------------------------------

def mortality_hazard(age: float, a: float = GM_A, b: float = GM_B,
                     c: float = GM_C) -> float:
    """Gompertz-Makeham instantaneous hazard of death at a given age."""
    return a * math.exp(b * max(0.0, age)) + c


def survival(age: float, years: float, a: float = GM_A, b: float = GM_B,
             c: float = GM_C, dt: float = 0.25) -> float:
    """Probability of surviving ``years`` starting from ``age``."""
    if years <= 0:
        return 1.0
    steps = max(1, int(math.ceil(years / dt)))
    h = years / steps
    integral = 0.0
    for i in range(steps):
        integral += mortality_hazard(age + (i + 0.5) * h, a, b, c) * h
    return math.exp(-integral)


def death_probability(age: float, years: float, **kw) -> float:
    return _clamp01(1.0 - survival(age, years, **kw))


def leader_departure_probability(age: float, tenure: float, horizon: float,
                                 stress: float = 0.0, base_exit: float = 0.05,
                                 tenure_slope: float = 0.02,
                                 stress_k: float = 1.5,
                                 dt: float = 0.25) -> float:
    """P(leader leaves office within ``horizon`` years) for any reason."""
    if horizon <= 0:
        return 0.0
    stress = _clamp01(stress)
    steps = max(1, int(math.ceil(horizon / dt)))
    h = horizon / steps
    integral = 0.0
    for i in range(steps):
        t = (i + 0.5) * h
        natural = mortality_hazard(age + t)
        political = (base_exit + tenure_slope * (tenure + t)) * math.exp(stress_k * stress)
        integral += (natural + political) * h
    return _clamp01(1.0 - math.exp(-integral))


def expected_remaining_tenure(age: float, tenure: float, stress: float = 0.0,
                              horizon: float = 40.0, dt: float = 0.5) -> float:
    """Rough E[remaining years in office], integrating the survival curve."""
    steps = max(1, int(math.ceil(horizon / dt)))
    h = horizon / steps
    prev_surv = 1.0
    expected = 0.0
    for i in range(steps):
        t = (i + 1) * h
        p_leave = leader_departure_probability(age, tenure, t, stress)
        surv = 1.0 - p_leave
        expected += 0.5 * (prev_surv + surv) * h
        prev_surv = surv
    return expected


# ---------------------------------------------------------------------------
# Influence and probability shift
# ---------------------------------------------------------------------------

def influence_weight(influence: float, role: str = "other") -> float:
    """Combine a raw influence with a role weight, clamped to [0, 1]."""
    w = ROLE_WEIGHTS.get((role or "other").strip().lower(), ROLE_WEIGHTS["other"])
    return _clamp01(_clamp01(influence) * w)


def shift_probability(p: float, weight: float, direction: float = 1.0,
                      strength: float = 1.0) -> float:
    """Shift a probability in log-odds space by a bounded weight.

    ``direction`` > 0 pushes the probability up (actor enables the outcome),
    < 0 pushes it down. ``strength`` scales the effect.
    """
    d = 1.0 if direction > 0 else -1.0
    delta = d * max(0.0, strength) * _clamp01(weight) * LOGIT_SCALE
    return _clamp01(_sigmoid(_logit(p) + delta))


# ---------------------------------------------------------------------------
# Coalitions
# ---------------------------------------------------------------------------

def majority_threshold(total_seats: int) -> int:
    return total_seats // 2 + 1


def is_winning(seats: Dict[str, float], threshold: float) -> bool:
    return sum(seats.values()) >= threshold


def winning_coalitions(seats: Dict[str, float], threshold: float) -> List[List[str]]:
    """All subsets of parties whose combined seats reach the threshold."""
    names = list(seats)
    out = []
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            if sum(seats[n] for n in combo) >= threshold:
                out.append(list(combo))
    return out


def minimal_winning_coalitions(seats: Dict[str, float], threshold: float) -> List[List[str]]:
    """Winning coalitions where no member can be removed and still win."""
    out = []
    for combo in winning_coalitions(seats, threshold):
        if all(
            sum(seats[m] for m in combo if m != member) < threshold
            for member in combo
        ):
            out.append(combo)
    return out


def banzhaf_index(seats: Dict[str, float], threshold: float) -> Dict[str, float]:
    """Banzhaf power index: share of coalitions where each party is pivotal."""
    names = list(seats)
    n = len(names)
    if n == 0:
        return {}
    pivotal = {name: 0 for name in names}
    for r in range(n + 1):
        for combo in itertools.combinations(names, r):
            total = sum(seats[p] for p in combo)
            for name in names:
                if name in combo:
                    continue
                if total < threshold <= total + seats[name]:
                    pivotal[name] += 1
    denom = sum(pivotal.values())
    if denom <= _EPS:
        return {name: 0.0 for name in names}
    return {name: round(count / denom, 4) for name, count in pivotal.items()}


# ---------------------------------------------------------------------------
# Succession
# ---------------------------------------------------------------------------

def succession_viability(heir_present: bool, heir_strength: float = 0.5,
                         regime_tenure: float = 0.0, instability: float = 0.0) -> float:
    """P(orderly succession) from heir presence/strength, tenure and instability."""
    base = 0.35 + 0.5 * _clamp01(heir_strength) if heir_present else 0.12
    # long personalist tenures make a vacuum more dangerous
    tenure_penalty = min(0.2, 0.01 * max(0.0, regime_tenure))
    return _clamp01(base - tenure_penalty - 0.5 * _clamp01(instability))


def summarize_actors(actors: Sequence[dict], horizon: float = 4.0) -> Dict[str, object]:
    """Aggregate a list of actor dicts into a few headline numbers.

    Each actor: {name, role, influence, age?, tenure_years?, stress?}
    """
    total_influence = 0.0
    highest_departure = 0.0
    most_at_risk: Optional[str] = None
    for a in actors:
        w = influence_weight(a.get("influence", 0.5), a.get("role", "other"))
        total_influence += w
        age = a.get("age")
        if age is not None:
            p = leader_departure_probability(
                float(age), float(a.get("tenure_years", 0.0) or 0.0),
                horizon, stress=float(a.get("stress", 0.0) or 0.0),
            )
            if p > highest_departure:
                highest_departure = p
                most_at_risk = a.get("name")
    return {
        "n_actors": len(actors),
        "total_influence": round(_clamp01(total_influence), 4),
        "highest_departure_prob": round(highest_departure, 4),
        "most_at_risk": most_at_risk,
    }
