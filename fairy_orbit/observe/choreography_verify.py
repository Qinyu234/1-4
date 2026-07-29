"""PROMPT §3.2 algebraic choreography gate.

Integrate to τ = T/n and require (positions and velocities):

    x_i(τ) = R · x_{P(i)}(0)

with P a cyclic role map and R ∈ SO(3) (Kabsch from positions; same R for v).
Binary residual — not a soft score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from fairy_orbit.core.body import System
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate
from fairy_orbit.observe.closure import closure_for_perm


def cyclic_role_perm(n: int, shift: int = 1) -> tuple[int, ...]:
    """P with P(i) = (i + shift) mod n  →  x_i(τ) ≈ R x_{(i+shift)}(0)."""
    if n < 2:
        raise ValueError("n >= 2 required")
    shift %= n
    return tuple((i + shift) % n for i in range(n))


def perm_cycle_label(perm: tuple[int, ...]) -> str:
    """Human-readable cycle type, e.g. '(0 1 2 3)' or 'id'."""
    n = len(perm)
    if perm == tuple(range(n)):
        return "id"
    visited = [False] * n
    cycles: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        c = []
        i = start
        while not visited[i]:
            visited[i] = True
            c.append(i)
            i = perm[i]
        if len(c) > 1 or (len(c) == 1 and perm[c[0]] != c[0]):
            cycles.append(c)
        elif len(c) == 1:
            cycles.append(c)
    # For role map as tuple listing source for each slot, show as that tuple.
    return "(" + " ".join(str(j) for j in perm) + ")"


def rotation_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract unit axis and angle (rad) from R ∈ SO(3)."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_th = float((np.trace(R) - 1.0) * 0.5)
    cos_th = max(-1.0, min(1.0, cos_th))
    angle = math.acos(cos_th)
    if angle < 1e-14:
        return np.array([0.0, 0.0, 1.0]), 0.0
    # Rodriguez: axis from skew-symmetric part
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    )
    nrm = float(np.linalg.norm(axis))
    if nrm < 1e-14:
        # angle ~ π: use diagonal eigenspace
        w, V = np.linalg.eigh(R)
        idx = int(np.argmax(w))
        axis = V[:, idx].real
        nrm = float(np.linalg.norm(axis))
    axis = axis / max(nrm, 1e-300)
    return axis, float(angle)


@dataclass(frozen=True)
class ChoreographyVerifyResult:
    ok: bool
    E_r: float
    E_v: float
    E_r_rel: float
    E_v_rel: float
    perm: tuple[int, ...]
    perm_label: str
    R: np.ndarray
    axis: tuple[float, float, float]
    angle: float
    tau: float
    period: float
    n_bodies: int
    shift: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "E_r": self.E_r,
            "E_v": self.E_v,
            "E_r_rel": self.E_r_rel,
            "E_v_rel": self.E_v_rel,
            "perm": list(self.perm),
            "perm_label": self.perm_label,
            "axis": list(self.axis),
            "angle": self.angle,
            "tau": self.tau,
            "period": self.period,
            "n_bodies": self.n_bodies,
            "shift": self.shift,
        }


def verify_choreography_Tn(
    system: System,
    period: float,
    *,
    shift: int = 1,
    n_bodies: int | None = None,
    atol_rel: float = 1e-6,
    n_outputs: int = 64,
    config: ReboundConfig | None = None,
) -> ChoreographyVerifyResult:
    """
    PROMPT §3.2: integrate to τ=T/n; check x_i(τ)=R x_{P(i)}(0) for r and v.
    """
    n = int(n_bodies if n_bodies is not None else system.n)
    if n != system.n:
        raise ValueError(f"n_bodies={n} != system.n={system.n}")
    period = float(period)
    tau = period / n
    perm = cyclic_role_perm(n, shift=shift)

    r0 = system.positions().copy()
    v0 = system.velocities().copy()
    cfg = config or ReboundConfig(
        stop_on_escape=False,
        stop_on_collision=False,
        epsilon=1e-9,
        min_dt=1e-8,
    )
    traj = integrate(system, t_end=tau, n_outputs=n_outputs, config=cfg)
    r = traj.positions[-1]
    v = traj.velocities[-1]
    cl = closure_for_perm(r, v, r0, v0, perm)

    scale_r = max(float(np.sum(r0 * r0)), 1e-300)
    scale_v = max(float(np.sum(v0 * v0)), 1e-300)
    E_r_rel = float(cl.E_r / scale_r)
    E_v_rel = float(cl.E_v / scale_v)
    axis, angle = rotation_axis_angle(cl.R)
    ok = E_r_rel < atol_rel and E_v_rel < atol_rel

    return ChoreographyVerifyResult(
        ok=ok,
        E_r=float(cl.E_r),
        E_v=float(cl.E_v),
        E_r_rel=E_r_rel,
        E_v_rel=E_v_rel,
        perm=perm,
        perm_label=perm_cycle_label(perm),
        R=np.asarray(cl.R, dtype=float),
        axis=(float(axis[0]), float(axis[1]), float(axis[2])),
        angle=angle,
        tau=tau,
        period=period,
        n_bodies=n,
        shift=int(shift % n),
    )


def is_regular_equal_ngon(
    positions: np.ndarray,
    *,
    rtol: float = 0.05,
) -> bool:
    """
    True if bodies form a regular n-gon about the COM at one instant.

    Uses the best-fit plane (not a fixed xy projection), so tilted squares
    count. Instantaneous shape alone is **not** a reject criterion — use
    ``maintains_regular_equal_ngon`` for the policy gate.
    """
    p = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = p.shape[0]
    if n < 3:
        return False
    com = np.mean(p, axis=0)
    centered = p - com
    # PCA: two largest singular vectors span the plane
    try:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return False
    if float(s[0]) < 1e-14:
        return True
    # thickness / in-plane scale
    if s.shape[0] >= 3 and float(s[2]) / float(s[0]) > rtol:
        return False
    u1, u2 = vt[0], vt[1]
    xy = np.column_stack([centered @ u1, centered @ u2])
    radii = np.linalg.norm(xy, axis=1)
    r_mean = float(np.mean(radii))
    if r_mean < 1e-14:
        return True
    r_cv = float(np.std(radii) / r_mean)
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    order = np.argsort(ang)
    cyc = xy[order]
    edges = np.array(
        [float(np.linalg.norm(cyc[(i + 1) % n] - cyc[i])) for i in range(n)],
        dtype=float,
    )
    e_mean = float(np.mean(edges))
    e_cv = float(np.std(edges) / max(e_mean, 1e-300))
    angs = np.sort(ang)
    dth = np.diff(np.concatenate([angs, angs[:1] + 2 * np.pi]))
    d_mean = float(np.mean(dth))
    d_cv = float(np.std(dth) / max(d_mean, 1e-300))
    return r_cv < rtol and e_cv < rtol and d_cv < rtol


def maintains_regular_equal_ngon(
    system: System,
    period: float,
    *,
    rtol: float = 0.05,
    n_samples: int = 16,
    min_fraction: float = 0.9,
    config: ReboundConfig | None = None,
) -> bool:
    """
    True if the orbit **keeps** a regular equal n-gon over one period.

    Samples positions along [0, T]; if ≥ ``min_fraction`` of samples are
    regular n-gons, the motion is treated as a rigid RE (A=B=C=…) and
    rejected by policy. Momentary polygonal IC that then deforms is OK.
    """
    period = float(period)
    if period <= 0.0 or n_samples < 2:
        return False
    cfg = config or ReboundConfig(
        stop_on_escape=False,
        stop_on_collision=False,
        epsilon=0.0,
        dt=max(period / max(200, 8 * n_samples), 1e-4),
        min_dt=1e-6,
    )
    traj = integrate(system, t_end=period, n_outputs=n_samples, config=cfg)
    hits = 0
    for k in range(traj.positions.shape[0]):
        if is_regular_equal_ngon(traj.positions[k], rtol=rtol):
            hits += 1
    return (hits / float(traj.positions.shape[0])) >= float(min_fraction)


@dataclass(frozen=True)
class OrbitAcceptResult:
    """Full free-orbit acceptance: §3.2 plus no maintained regular n-gon RE."""

    ok: bool
    reason: str
    choreography: ChoreographyVerifyResult
    maintains_regular_ngon: bool

    @property
    def is_regular_ngon(self) -> bool:
        """Alias: maintained regular n-gon RE (not instantaneous shape)."""
        return self.maintains_regular_ngon

    def to_dict(self) -> dict[str, Any]:
        d = self.choreography.to_dict()
        d["ok"] = self.ok
        d["reason"] = self.reason
        d["maintains_regular_ngon"] = self.maintains_regular_ngon
        d["is_regular_ngon"] = self.maintains_regular_ngon  # compat
        d["choreography_ok"] = self.choreography.ok
        return d


def accept_free_choreography(
    system: System,
    period: float,
    *,
    shift: int = 1,
    atol_rel: float = 1e-6,
    ngon_rtol: float = 0.05,
    n_outputs: int = 64,
    ngon_samples: int = 16,
) -> OrbitAcceptResult:
    """
    Accept only if §3.2 holds AND the orbit does not maintain a regular n-gon.

    Rigid rotating square/pentagon relative equilibria (roles A=B=C=…) are
    rejected. An IC that looks polygonal for one snapshot is not rejected
    unless the shape stays regular along the period.
    """
    gate = verify_choreography_Tn(
        system,
        period,
        shift=shift,
        atol_rel=atol_rel,
        n_outputs=n_outputs,
    )
    maintained = maintains_regular_equal_ngon(
        system,
        period,
        rtol=ngon_rtol,
        n_samples=ngon_samples,
    )
    if not gate.ok:
        return OrbitAcceptResult(
            ok=False,
            reason="failed_prompt_3_2",
            choreography=gate,
            maintains_regular_ngon=maintained,
        )
    if maintained:
        return OrbitAcceptResult(
            ok=False,
            reason="rejected_maintained_regular_ngon",
            choreography=gate,
            maintains_regular_ngon=True,
        )
    return OrbitAcceptResult(
        ok=True,
        reason="ok",
        choreography=gate,
        maintains_regular_ngon=False,
    )


def accept_seed_choreography(
    seed,
    *,
    shift: int = 1,
    atol_rel: float = 1e-6,
    ngon_rtol: float = 0.05,
    n_outputs: int = 64,
    ngon_samples: int = 16,
) -> OrbitAcceptResult:
    return accept_free_choreography(
        seed.to_system(),
        float(seed.period),
        shift=shift,
        atol_rel=atol_rel,
        ngon_rtol=ngon_rtol,
        n_outputs=n_outputs,
        ngon_samples=ngon_samples,
    )


def verify_seed_choreography(
    seed,
    *,
    shift: int = 1,
    atol_rel: float = 1e-6,
    n_outputs: int = 64,
) -> ChoreographyVerifyResult:
    """Run §3.2 only (no maintained-ngon policy). Prefer accept_seed_choreography."""
    return verify_choreography_Tn(
        seed.to_system(),
        float(seed.period),
        shift=shift,
        n_bodies=seed.n_bodies,
        atol_rel=atol_rel,
        n_outputs=n_outputs,
    )
