"""Event coupling & cascades (ROADMAP section C).

The base tree *looks* causal but its nodes are scored in isolation. This module
makes a child's probability actually depend on its parent, and provides the
building blocks for contagion and for physical constraints across the tree.

1. Conditioning a child on its parent
-------------------------------------
Let ``p`` be the child's standalone probability and ``q`` the parent's
probability. A coupling coefficient ``c in [-1, 1]`` encodes how strongly the
child co-moves with the parent. We set

    P(child | parent occurs)     = p + c (1 - p)
    P(child | parent does not)   = p - c p

Both stay in [0, 1] for c in [0, 1]. Marginalising over an uncertain parent,

    P(child) = q P(child|1) + (1-q) P(child|0) = p + c (q - p)

i.e. the child is dragged towards the parent in proportion to ``c``. ``c > 0``
means "enables / causes" (parent success raises the child), ``c < 0`` means
"prevents" (parent success lowers it), ``c = 0`` recovers the independent model.

2. Multiple causes -> noisy-OR
------------------------------
For a node enabled by several independent potential causes
``(p_i, w_i)``, ``P(effect) = 1 - prod_i (1 - w_i p_i)`` (noisy-OR). With all
``w_i = 1`` this is the standard probabilistic OR; weights damp weak causes.

3. Contagion (SIR-like diffusion)
---------------------------------
Discrete SIR on a well-mixed population of size N:

    dS/dt = -beta S I / N,   dI/dt = beta S I / N - gamma I,   dR/dt = gamma I

integrated with explicit Euler (``sir_simulate``). The basic reproduction number
is ``R0 = beta / gamma``; the final outbreak size solves the implicit equation
``z = 1 - exp(-R0 z)`` (``sir_final_size``, fixed-point iteration). This models
protests, violence, bank runs and adoption cascades.

4. Conservation / constraint checks
-----------------------------------
``check_conservation`` validates that sub-quantities sum to a known total within
a tolerance (seats, votes, budgets), so the tree cannot emit impossible numbers.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

_EPS = 1e-12


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# Coupling strength per branch: continuation co-moves mildly, the optimistic
# branch is strongly enabled by parent success, the pessimistic branch is
# suppressed by it (prevents-style). Calibratable.
COUPLING_BY_BRANCH: Dict[str, float] = {
    "same": 0.35,
    "max": 0.60,
    "min": -0.60,
}


def coupling_coefficient(branch_type: str, relationship: Optional[str] = None) -> float:
    """Coupling coefficient for a branch (or an explicit cause/enable/prevents)."""
    if relationship is not None:
        return {
            "causes": 0.7,
            "enables": 0.5,
            "correlates": 0.3,
            "prevents": -0.6,
            "blocks": -0.8,
        }.get(relationship, 0.0)
    return COUPLING_BY_BRANCH.get(branch_type, 0.0)


def conditional_probabilities(p_child: float, p_parent: float,
                              c: float) -> Dict[str, float]:
    """Return P(child|parent), P(child|not parent) and the marginal P(child)."""
    p = _clamp01(p_child)
    q = _clamp01(p_parent)
    c = max(-1.0, min(1.0, float(c)))
    p_if = _clamp01(p + c * (1.0 - p))
    p_if_not = _clamp01(p - c * p)
    marginal = _clamp01(p + c * (q - p))
    return {"p_if_parent": p_if, "p_if_not_parent": p_if_not, "marginal": marginal}


def adjust_for_parent(p_child: float, p_parent: float, branch_type: str) -> float:
    """Child probability after coupling it to its parent's probability."""
    c = coupling_coefficient(branch_type)
    return conditional_probabilities(p_child, p_parent, c)["marginal"]


def noisy_or(causes: Sequence[Tuple[float, float]]) -> float:
    """P(effect) under a noisy-OR of (probability, weight) pairs."""
    prod = 1.0
    for p, w in causes:
        prod *= (1.0 - _clamp01(p) * max(0.0, min(1.0, w)))
    return _clamp01(1.0 - prod)


def noisy_and(causes: Sequence[Tuple[float, float]]) -> float:
    """P(effect) under a noisy-AND (all causes must fire)."""
    prod = 1.0
    for p, w in causes:
        prod *= (1.0 - max(0.0, min(1.0, w)) * (1.0 - _clamp01(p)))
    return _clamp01(prod)


# ---------------------------------------------------------------------------
# Contagion (SIR-like)
# ---------------------------------------------------------------------------

def sir_simulate(beta: float, gamma: float, n: float = 1.0, i0: float = 1e-3,
                 steps: int = 365, dt: Optional[float] = None) -> Dict[str, object]:
    """Explicit-Euler SIR. Returns final infected fraction and the trajectory."""
    if beta < 0.0 or gamma <= 0.0:
        raise ValueError("beta must be >= 0 and gamma > 0")
    if dt is None:
        dt = 1.0
    s = n - i0
    i = i0
    r = 0.0
    peak = i0
    traj = []
    for _ in range(steps):
        new = beta * s * i / n * dt
        rec = gamma * i * dt
        new = min(new, s)
        s -= new
        i += new - rec
        if i < 0:
            i = 0.0
        r += rec
        peak = max(peak, i)
        traj.append((s / n, i / n, r / n))
    attacked = r / n
    return {
        "R0": beta / gamma,
        "final_infected": attacked,
        "peak_infected": peak,
        "trajectory": traj,
    }


def sir_final_size(R0: float, tol: float = 1e-10, iters: int = 1000) -> float:
    """Solve z = 1 - exp(-R0 z) for the final outbreak fraction.

    R0 <= 1 => 0 (no outbreak); R0 > 1 => a fraction in (0, 1).
    """
    if R0 <= 1.0:
        return 0.0
    z = 0.5
    for _ in range(iters):
        z_next = 1.0 - math.exp(-R0 * z)
        if abs(z_next - z) < tol:
            return z_next
        z = z_next
    return z


# ---------------------------------------------------------------------------
# Conservation / constraints
# ---------------------------------------------------------------------------

def check_conservation(parts: Sequence[float], total: float,
                       tol: float = 1e-6) -> Dict[str, object]:
    """Verify that ``sum(parts) == total`` within tolerance."""
    s = sum(float(x) for x in parts)
    ok = abs(s - total) <= tol * max(1.0, abs(total))
    return {"ok": ok, "sum": s, "total": total, "residual": s - total}


def clamp_to_total(parts: Sequence[float], total: float) -> List[float]:
    """Scale non-negative parts so that they sum to ``total`` (conserving)."""
    xs = [max(0.0, float(x)) for x in parts]
    s = sum(xs)
    if s <= _EPS:
        return [total / len(xs) for x in xs] if xs else []
    return [x * total / s for x in xs]


def propagate_coupling(nodes: List[dict], coupling_by_branch: Optional[Dict[str, float]] = None) -> None:
    """In-place coupling pass over a flat list of {id, parent_id, branch_type, p}.

    Each node's ``p`` is pulled towards its parent's (already-updated) ``p``.
    Parents must appear before children (the predictor's tree is level-ordered).
    """
    lut = coupling_by_branch or COUPLING_BY_BRANCH
    by_id = {n["id"]: n for n in nodes}
    for n in nodes:
        pid = n.get("parent_id")
        if pid and pid in by_id:
            c = lut.get(n.get("branch_type", "same"), 0.0)
            n["p"] = conditional_probabilities(n["p"], by_id[pid]["p"], c)["marginal"]
