"""PEO Level 2–3 closure errors E_r, E_v (PROMPT).

Pipeline for a fixed permutation P (from Level 1 choreography):

  Level 2 — Position closure:
      R* = argmin_{R∈SO(3)} Σ_i || r_i(T) − R r_{P(i)}(0) ||²
      E_r = that minimum

  Level 3 — Velocity closure (same R*):
      E_v = Σ_i || v_i(T) − R* v_{P(i)}(0) ||²

No separate velocity Kabsch. No “score” — only these two errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from fairy_orbit.engine.trajectory import Trajectory


def all_permutations(n: int = 4) -> list[tuple[int, ...]]:
    return list(permutations(range(n)))


def kabsch_rotation(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """R* ∈ SO(3) minimizing Σ || target_i − R source_i ||² (no centroid shift)."""
    A = np.asarray(target, dtype=float).reshape(-1, 3)
    B = np.asarray(source, dtype=float).reshape(-1, 3)
    if A.shape != B.shape:
        raise ValueError("target and source shapes must match")
    H = B.T @ A
    U, _S, Vt = np.linalg.svd(H)
    d = float(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, 1.0 if d >= 0.0 else -1.0])
    return Vt.T @ D @ U.T


def E_r(
    r: np.ndarray,
    r0: np.ndarray,
    R: np.ndarray,
    perm: tuple[int, ...],
) -> float:
    """Σ || r_i − R r0_{P(i)} ||²."""
    r = np.asarray(r, dtype=float).reshape(-1, 3)
    r0 = np.asarray(r0, dtype=float).reshape(-1, 3)
    total = 0.0
    for i, j in enumerate(perm):
        d = r[i] - R @ r0[j]
        total += float(d @ d)
    return total


def E_v(
    v: np.ndarray,
    v0: np.ndarray,
    R: np.ndarray,
    perm: tuple[int, ...],
) -> float:
    """Σ || v_i − R v0_{P(i)} ||²  (R must be the position R*)."""
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    v0 = np.asarray(v0, dtype=float).reshape(-1, 3)
    total = 0.0
    for i, j in enumerate(perm):
        d = v[i] - R @ v0[j]
        total += float(d @ d)
    return total


# Back-compat aliases (deprecated names)
shape_score = E_r
velocity_score = E_v


@dataclass(frozen=True)
class ClosureResult:
    E_r: float
    E_v: float
    R: np.ndarray  # R* from positions only
    perm: tuple[int, ...]


def closure_for_perm(
    r: np.ndarray,
    v: np.ndarray,
    r0: np.ndarray,
    v0: np.ndarray,
    perm: tuple[int, ...],
) -> ClosureResult:
    """
    PROMPT Level 2–3 for fixed P:
      R* ← Kabsch on positions
      E_r ← position residual
      E_v ← velocity residual with the same R*
    """
    src = np.stack([np.asarray(r0, dtype=float).reshape(-1, 3)[j] for j in perm], axis=0)
    R_star = kabsch_rotation(r, src)
    return ClosureResult(
        E_r=E_r(r, r0, R_star, perm),
        E_v=E_v(v, v0, R_star, perm),
        R=R_star,
        perm=tuple(perm),
    )


def best_closure_by_Er(
    r: np.ndarray,
    v: np.ndarray,
    r0: np.ndarray,
    v0: np.ndarray,
    *,
    perms: list[tuple[int, ...]] | None = None,
) -> ClosureResult:
    """Enumerate P∈S₄; pick the one with smallest E_r (diagnostic only)."""
    if perms is None:
        perms = all_permutations(len(np.asarray(r0).reshape(-1, 3)))
    best: ClosureResult | None = None
    for p in perms:
        res = closure_for_perm(r, v, r0, v0, p)
        if best is None or res.E_r < best.E_r:
            best = res
    assert best is not None
    return best


# alias used by older tests
def best_closure(*args, primary: str = "shape", **kwargs) -> ClosureResult:
    return best_closure_by_Er(*args, **kwargs)


def radial_order(
    positions: np.ndarray,
    *,
    central_index: int = 0,
) -> tuple[int, ...]:
    """Fairy-local indices sorted by |r − r_central| ascending."""
    pos = np.asarray(positions, dtype=float)
    central = pos[central_index]
    fairy_idx = [i for i in range(pos.shape[0]) if i != central_index]
    radii = [float(np.linalg.norm(pos[i] - central)) for i in fairy_idx]
    return tuple(int(k) for k in np.argsort(radii))


def radial_permutation(
    order_t: tuple[int, ...],
    order_0: tuple[int, ...],
) -> tuple[int, ...]:
    """
    P from S(T)=P_prompt(S(0)): body now at order_t[k] came from order_0[k],
    so our closure perm satisfies perm[order_t[k]] = order_0[k]
    (r_i(t) ↔ R r0[perm[i]]).
    """
    n = len(order_0)
    perm = [0] * n
    for k in range(n):
        perm[order_t[k]] = order_0[k]
    return tuple(perm)


@dataclass
class ClosureSeries:
    times: np.ndarray
    E_r: np.ndarray
    E_v: np.ndarray
    perm: tuple[int, ...]
    R_final: np.ndarray
    order_0: tuple[int, ...]
    order_final: tuple[int, ...]
    choreography_ok: bool


def fairy_states(
    traj: Trajectory,
    k: int,
    *,
    central_index: int = 0,
    frame: str = "com",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fairy (r, v) at sample k.

    frame:
      - 'com': absolute coordinates in the trajectory frame (assumed inertial
        COM). Translation already removed; SO(3) matching is deferred to R*.
      - 'heliocentric': subtract central body (legacy).
    """
    idx = [i for i in range(traj.n_bodies) if i != central_index]
    r = traj.positions[k, idx].copy()
    v = traj.velocities[k, idx].copy()
    if frame == "heliocentric":
        r0 = traj.positions[k, central_index]
        v0 = traj.velocities[k, central_index]
        r = r - r0
        v = v - v0
    elif frame != "com":
        raise ValueError(f"unknown frame {frame!r}; use com|heliocentric")
    return r, v


def closure_series(
    traj: Trajectory,
    *,
    central_index: int = 0,
    perm: tuple[int, ...] | None = None,
    mode: str = "identity",
    frame: str = "com",
) -> ClosureSeries:
    """
    E_r(t), E_v(t) with P fixed first, then R*(t) from positions only.

    States are in the inertial COM frame by default:
        r'_i(T) ≈ R r'_{P(i)}(0),  v'_i(T) ≈ R v'_{P(i)}(0)

    mode:
      - 'identity': P = id
      - 'fixed_radial': P from S(T) vs S(0) (Level 1)
      - 'fixed': use provided perm
    """
    r_ref, v_ref = fairy_states(traj, 0, central_index=central_index, frame=frame)
    order_0 = radial_order(traj.positions[0], central_index=central_index)
    order_f = radial_order(traj.positions[-1], central_index=central_index)
    radial_perm = radial_permutation(order_f, order_0)

    if mode == "fixed":
        if perm is None:
            raise ValueError("mode='fixed' requires perm")
    elif mode == "identity":
        perm = tuple(range(len(r_ref)))
    elif mode == "fixed_radial":
        perm = radial_perm
    else:
        raise ValueError(f"unknown mode {mode!r}; use identity|fixed_radial|fixed")

    T = len(traj)
    Er = np.empty(T)
    Ev = np.empty(T)
    R_last = np.eye(3)
    for k in range(T):
        r, v = fairy_states(traj, k, central_index=central_index, frame=frame)
        res = closure_for_perm(r, v, r_ref, v_ref, perm)
        Er[k] = res.E_r
        Ev[k] = res.E_v
        R_last = res.R

    return ClosureSeries(
        times=traj.times.copy(),
        E_r=Er,
        E_v=Ev,
        perm=perm,
        R_final=R_last,
        order_0=order_0,
        order_final=order_f,
        choreography_ok=(radial_perm == perm) or mode == "identity",
    )
