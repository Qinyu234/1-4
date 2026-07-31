"""Linear stability via finite-difference monodromy (relative map over τ)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fairy_orbit.design.seeds import OrbitSeed
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate_endpoint
from fairy_orbit.observe.choreography_verify import cyclic_role_perm
from fairy_orbit.observe.closure import closure_for_perm


def _pack(seed: OrbitSeed) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(seed.positions, dtype=float).ravel(),
            np.asarray(seed.velocities, dtype=float).ravel(),
        ]
    )


def _unpack(y: np.ndarray, template: OrbitSeed) -> OrbitSeed:
    n = template.n_bodies
    y = np.asarray(y, dtype=float).ravel()
    pos = y[: 3 * n].reshape(n, 3)
    vel = y[3 * n : 6 * n].reshape(n, 3)
    return OrbitSeed(
        id=template.id,
        family=template.family,
        n_bodies=n,
        G=template.G,
        masses=template.masses,
        period=template.period,
        positions=pos,
        velocities=vel,
        names=template.names,
        symmetry=template.symmetry,
        source=template.source,
        notes=template.notes,
        central_index=template.central_index,
        verification=template.verification,
    )


def _relative_map(
    seed: OrbitSeed,
    *,
    shift: int,
) -> np.ndarray:
    """G^{-1} Φ_τ(X): integrate τ then map back by Kabsch R and role perm."""
    n = seed.n_bodies
    tau = float(seed.period) / n
    perm = cyclic_role_perm(n, shift=shift)
    r0 = np.asarray(seed.positions, dtype=float)
    v0 = np.asarray(seed.velocities, dtype=float)
    r, v = integrate_endpoint(
        seed.to_system(),
        tau,
        config=ReboundConfig(
            stop_on_escape=False,
            stop_on_collision=False,
            epsilon=0.0,
            dt=max(tau / 200.0, 1e-3),
            min_dt=1e-5,
        ),
    )
    cl = closure_for_perm(r, v, r0, v0, perm)
    R = cl.R
    Rt = R.T
    # Pull Φ_τ state back by R^{-1} and undo role map: compare in seed ordering
    pos = np.zeros_like(r0)
    vel = np.zeros_like(v0)
    for i, j in enumerate(perm):
        # r_i(τ) ≈ R r_j(0)  ⇒  r_j(0)_pred = R^{-1} r_i(τ)
        pos[j] = Rt @ r[i]
        vel[j] = Rt @ v[i]
    return np.concatenate([pos.ravel(), vel.ravel()])


@dataclass(frozen=True)
class FloquetResult:
    multipliers: np.ndarray
    max_abs: float
    stable: bool
    n_unstable: int
    map_residual: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "multipliers": [
                {"re": float(np.real(z)), "im": float(np.imag(z)), "abs": float(abs(z))}
                for z in self.multipliers.tolist()
            ],
            "max_abs": self.max_abs,
            "stable": self.stable,
            "n_unstable": self.n_unstable,
            "map_residual": self.map_residual,
        }


def floquet_multipliers_fd(
    seed: OrbitSeed,
    *,
    shift: int = 1,
    eps: float = 1e-7,
    stable_atol: float = 0.05,
) -> FloquetResult:
    """
    Finite-difference monodromy of the relative map F(X)=G^{-1}Φ_τ(X).

    ``stable`` iff all |λ| ≤ 1 + ``stable_atol`` (phase/trivial modes near 1
    are tolerated inside that band).
    """
    y0 = _pack(seed)
    f0 = _relative_map(seed, shift=shift)
    map_residual = float(np.linalg.norm(f0 - y0))
    dim = y0.size
    J = np.zeros((dim, dim), dtype=float)
    for j in range(dim):
        dy = np.zeros(dim)
        dy[j] = float(eps)
        yp = _unpack(y0 + dy, seed)
        ym = _unpack(y0 - dy, seed)
        fp = _relative_map(yp, shift=shift)
        fm = _relative_map(ym, shift=shift)
        J[:, j] = (fp - fm) / (2.0 * float(eps))
    eigs = np.linalg.eigvals(J)
    abs_e = np.abs(eigs)
    max_abs = float(np.max(abs_e)) if abs_e.size else float("nan")
    n_unstable = int(np.sum(abs_e > 1.0 + float(stable_atol)))
    return FloquetResult(
        multipliers=eigs,
        max_abs=max_abs,
        stable=bool(n_unstable == 0 and np.isfinite(max_abs)),
        n_unstable=n_unstable,
        map_residual=map_residual,
    )
