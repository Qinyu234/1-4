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


# Role exchange on fairy indices T1..T4 ≡ 0..3 (ABCD).
ROLE_CYCLIC_PERMS: dict[str, tuple[int, ...]] = {
    "id": (0, 1, 2, 3),
    "bcda": (1, 2, 3, 0),  # A→B slot: r_i(T) ≈ R r_{i+1}(0)
    "cdab": (2, 3, 0, 1),
    "dabc": (3, 0, 1, 2),
}


def role_cyclic_perms() -> list[tuple[int, ...]]:
    """Non-identity cyclic role shifts (BCDA, CDAB, DABC)."""
    return [ROLE_CYCLIC_PERMS["bcda"], ROLE_CYCLIC_PERMS["cdab"], ROLE_CYCLIC_PERMS["dabc"]]


def resolve_perm(
    mode: str,
    *,
    n: int,
    radial_perm: tuple[int, ...],
    perm: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Map perm_mode string to a fixed P ∈ S_n."""
    if mode == "fixed":
        if perm is None:
            raise ValueError("mode='fixed' requires perm")
        return tuple(perm)
    if mode == "identity":
        return tuple(range(n))
    if mode == "fixed_radial":
        return radial_perm
    if mode in ROLE_CYCLIC_PERMS:
        return ROLE_CYCLIC_PERMS[mode]
    if mode == "fixed_bcda":
        return ROLE_CYCLIC_PERMS["bcda"]
    if mode == "fixed_dabc":
        return ROLE_CYCLIC_PERMS["dabc"]
    raise ValueError(
        f"unknown perm mode {mode!r}; use identity|fixed_radial|fixed|"
        f"fixed_bcda|fixed_dabc|bcda|dabc|cdab"
    )


def cyclic_shift(seq: tuple[int, ...], k: int) -> tuple[int, ...]:
    n = len(seq)
    k %= n
    if k == 0:
        return seq
    return tuple(seq[k:] + seq[:k])


def radial_choreography_shift(
    order_0: tuple[int, ...],
    order_f: tuple[int, ...],
) -> int | None:
    """
    If S(t) is a cyclic left shift of S(0) on the radial ladder, return k ∈ {0,…,n−1}.
    k=0 means unchanged order. None if not a cyclic shift.
    """
    if len(order_0) != len(order_f):
        return None
    n = len(order_0)
    for k in range(n):
        if order_f == cyclic_shift(order_0, k):
            return k
    return None


def required_choreography_shift(perm_mode: str) -> int | None:
    """
    Target radial cyclic shift k at t=T for perm_mode.

    Returns:
      0..n-1: exact shift required
      -1: any non-zero cyclic shift (ABCD→BCDA/CDAB/DABC)
      None: only require every S(t) be *some* cyclic shift of S(0)
    """
    if perm_mode == "identity":
        return 0
    if perm_mode in ("fixed_bcda", "bcda"):
        return 1
    if perm_mode == "cdab":
        return 2
    if perm_mode in ("fixed_dabc", "dabc"):
        return 3
    # fixed_radial, best_cyclic, default PEO: must exchange radial roles
    return -1


@dataclass(frozen=True)
class ChoreographyGateResult:
    ok: bool
    reason: str
    order_0: tuple[int, ...]
    order_final: tuple[int, ...]
    shift_final: int | None
    shifts: tuple[int | None, ...]


def choreography_gate(
    traj: Trajectory,
    perm_mode: str,
    *,
    central_index: int = 0,
) -> ChoreographyGateResult:
    """
  Level 1 hard gate (like escape): at every output time k, S(k) must be a
  cyclic shift of S(0). At t=T an additional shift requirement from perm_mode
  applies (default: k>0, i.e. not ABCD→ABCD).

  Failure → caller must skip E_r/E_v loss (same as escape).
    """
    order_0 = radial_order(traj.positions[0], central_index=central_index)
    shifts: list[int | None] = []
    for k in range(len(traj)):
        order_k = radial_order(traj.positions[k], central_index=central_index)
        sk = radial_choreography_shift(order_0, order_k)
        shifts.append(sk)
        if sk is None:
            return ChoreographyGateResult(
                ok=False,
                reason=f"non_cyclic_radial_order_at_k={k}",
                order_0=order_0,
                order_final=order_k,
                shift_final=None,
                shifts=tuple(shifts),
            )

    shift_final = shifts[-1]
    req = required_choreography_shift(perm_mode)
    if req == -1:
        if shift_final == 0:
            return ChoreographyGateResult(
                ok=False,
                reason="identity_radial_at_T",
                order_0=order_0,
                order_final=radial_order(traj.positions[-1], central_index=central_index),
                shift_final=shift_final,
                shifts=tuple(shifts),
            )
    elif req is not None and shift_final != req:
        return ChoreographyGateResult(
            ok=False,
            reason=f"radial_shift_mismatch_want_k={req}_got_k={shift_final}",
            order_0=order_0,
            order_final=radial_order(traj.positions[-1], central_index=central_index),
            shift_final=shift_final,
            shifts=tuple(shifts),
        )

    return ChoreographyGateResult(
        ok=True,
        reason="ok",
        order_0=order_0,
        order_final=radial_order(traj.positions[-1], central_index=central_index),
        shift_final=shift_final,
        shifts=tuple(shifts),
    )


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
    choreo_k = radial_choreography_shift(order_0, order_f)

    if mode == "fixed":
        if perm is None:
            raise ValueError("mode='fixed' requires perm")
        perm = tuple(perm)
    elif mode == "best_cyclic":
        # Pick role-cyclic P ∈ {id, BCDA, CDAB, DABC} minimizing E_r at T.
        rT, vT = fairy_states(traj, len(traj) - 1, central_index=central_index, frame=frame)
        best_res = None
        best_p: tuple[int, ...] | None = None
        for p in [ROLE_CYCLIC_PERMS["id"], *role_cyclic_perms()]:
            res = closure_for_perm(rT, vT, r_ref, v_ref, p)
            if best_res is None or res.E_r < best_res.E_r:
                best_res = res
                best_p = p
        assert best_p is not None and best_res is not None
        perm = best_p
    else:
        perm = resolve_perm(mode, n=len(r_ref), radial_perm=radial_perm, perm=perm)

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

    if mode == "identity":
        choreography_ok = choreo_k == 0
    elif mode == "fixed_radial":
        choreography_ok = radial_perm == perm
    elif mode in ("fixed_bcda", "bcda"):
        choreography_ok = choreo_k == 1
    elif mode == "fixed_dabc" or mode == "dabc":
        choreography_ok = choreo_k == 3
    elif mode == "cdab":
        choreography_ok = choreo_k == 2
    elif mode == "best_cyclic":
        choreography_ok = choreo_k is not None and choreo_k > 0
    else:
        choreography_ok = choreo_k is not None and choreo_k > 0

    return ClosureSeries(
        times=traj.times.copy(),
        E_r=Er,
        E_v=Ev,
        perm=perm,
        R_final=R_last,
        order_0=order_0,
        order_final=order_f,
        choreography_ok=choreography_ok,
    )
