"""Equal-mass choreography search (PROMPT construct path).

Multi-start polish of free-N IC for §3.2 residual. Orbits that *maintain*
a regular equal n-gon (rigid RE) are rejected; momentary polygonal shape is OK.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares

from fairy_orbit.core.body import Body, System, to_com_inertial_frame
from fairy_orbit.design.seeds import OrbitSeed, save_seed
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate
from fairy_orbit.observe.choreography_verify import (
    accept_free_choreography,
    cyclic_role_perm,
    is_regular_equal_ngon,
)
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
    r = y[: 3 * n].reshape(n, 3)
    v = y[3 * n : 6 * n].reshape(n, 3)
    period = float(template.period)
    sys = System(
        bodies=[
            Body(
                mass=float(template.masses[i]),
                position=r[i].copy(),
                velocity=v[i].copy(),
                name=template.names[i] if i < len(template.names) else f"B{i}",
            )
            for i in range(n)
        ],
        G=float(template.G),
    )
    to_com_inertial_frame(sys)
    r2 = np.stack([b.position for b in sys.bodies])
    v2 = np.stack([b.velocity for b in sys.bodies])
    return OrbitSeed(
        id=template.id,
        family=template.family,
        n_bodies=n,
        G=template.G,
        masses=template.masses,
        period=period,
        positions=r2,
        velocities=v2,
        names=template.names,
        symmetry=template.symmetry,
        source="choreography_search_polish",
        notes="multi-start §3.2 polish",
        central_index=None,
    )


def _collision_penalty(pos: np.ndarray, floor: float = 1e-3) -> float:
    n = pos.shape[0]
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < floor:
                pen += (floor - d) ** 2 * 1e6
    return pen


def _maintained_ngon_soft_penalty(
    r0: np.ndarray,
    r_tau: np.ndarray,
    strength: float = 50.0,
) -> float:
    """Soft push away from rigid RE: regular at t=0 and still regular at T/n."""
    if is_regular_equal_ngon(r0, rtol=0.08) and is_regular_equal_ngon(
        r_tau, rtol=0.08
    ):
        return strength
    return 0.0


def random_asymmetric_seed(
    n: int,
    rng: np.random.Generator,
    *,
    G: float = 1.0,
    mass: float = 1.0,
) -> OrbitSeed:
    """Random planar IC (unequal radii/gaps preferred; polygonal snapshot OK)."""
    gaps = rng.uniform(0.3, 1.7, size=n)
    gaps = gaps / gaps.sum() * 2.0 * math.pi
    angles = np.cumsum(gaps) - gaps[0]
    radii = rng.uniform(0.4, 1.6, size=n)
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    for i in range(n):
        c, s = math.cos(angles[i]), math.sin(angles[i])
        pos[i] = (radii[i] * c, radii[i] * s, 0.0)
        speed = rng.uniform(0.4, 1.4)
        vt = speed * (0.85 + 0.15 * rng.random())
        vr = rng.normal(0.0, 0.15)
        vel[i] = (-vt * s + vr * c, vt * c + vr * s, 0.0)

    period = float(rng.uniform(4.0, 10.0))
    names = tuple(f"B{i+1}" for i in range(n))
    family = f"free_{n}"
    sys = System(
        bodies=[
            Body(mass=mass, position=pos[i], velocity=vel[i], name=names[i])
            for i in range(n)
        ],
        G=G,
    )
    to_com_inertial_frame(sys)
    return OrbitSeed(
        id=f"rand_{n}",
        family=family,
        n_bodies=n,
        G=G,
        masses=tuple(mass for _ in range(n)),
        period=period,
        positions=np.stack([b.position for b in sys.bodies]),
        velocities=np.stack([b.velocity for b in sys.bodies]),
        names=names,
        symmetry="asymmetric_search",
        source="random_asymmetric_ic",
        notes="random planar start",
        central_index=None,
    )


def symmetry_residual_seed(
    seed: OrbitSeed,
    *,
    shift: int = 1,
    n_outputs: int = 12,
) -> np.ndarray:
    n = seed.n_bodies
    tau = float(seed.period) / n
    perm = cyclic_role_perm(n, shift=shift)
    r0 = np.asarray(seed.positions, dtype=float)
    v0 = np.asarray(seed.velocities, dtype=float)
    sys = seed.to_system()
    traj = integrate(
        sys,
        t_end=tau,
        n_outputs=n_outputs,
        config=ReboundConfig(
            stop_on_escape=False,
            stop_on_collision=False,
            epsilon=0.0,
            dt=max(tau / 200.0, 1e-3),
            min_dt=1e-5,
        ),
    )
    r = traj.positions[-1]
    v = traj.velocities[-1]
    cl = closure_for_perm(r, v, r0, v0, perm)
    R = cl.R
    chunks = []
    for i, j in enumerate(perm):
        chunks.append(r[i] - R @ r0[j])
        chunks.append(v[i] - R @ v0[j])
    extras = [
        _collision_penalty(r0),
        _maintained_ngon_soft_penalty(r0, r),
    ]
    return np.concatenate([np.concatenate(chunks).astype(float), np.asarray(extras)])


def polish_seed(
    seed: OrbitSeed,
    *,
    shift: int = 1,
    max_nfev: int = 12,
) -> tuple[OrbitSeed, float]:
    y0 = _pack(seed)

    def fun(y: np.ndarray) -> np.ndarray:
        return symmetry_residual_seed(_unpack(y, seed), shift=shift)

    sol = least_squares(
        fun, y0, method="trf", max_nfev=max_nfev, ftol=1e-10, xtol=1e-10
    )
    polished = _unpack(sol.x, seed)
    return polished, float(np.linalg.norm(sol.fun))


@dataclass
class SearchTrial:
    trial: int
    residual: float
    period: float
    ok_gate: bool
    path: str | None
    reason: str


def run_choreography_search(
    n: int,
    *,
    wall_hours: float | None = None,
    shift: int = 1,
    rng: np.random.Generator | None = None,
    out_dir: Path | None = None,
    max_nfev: int = 14,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Long-running multi-start §3.2 polish for free equal-mass N-body.

    ``wall_hours=None`` runs until interrupted (unlimited).
    Maintained regular n-gon REs never count as passes.
    """
    rng = rng or np.random.default_rng(n * 10007 + 17)
    out_dir = Path(out_dir or f"experiments/output/choreography_search_n{n}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "trials.jsonl"

    t_end = (
        None
        if wall_hours is None or wall_hours <= 0
        else time.time() + float(wall_hours) * 3600.0
    )
    best_res = float("inf")
    best_seed: OrbitSeed | None = None
    trial = 0
    passed = 0
    rejected_maintained = 0

    with log_path.open("a", encoding="utf-8") as logf:
        while t_end is None or time.time() < t_end:
            trial += 1
            start = random_asymmetric_seed(n, rng)
            try:
                polished, res_n = polish_seed(start, shift=shift, max_nfev=max_nfev)
                acc = accept_free_choreography(
                    polished.to_system(),
                    polished.period,
                    shift=shift,
                    atol_rel=1e-5,
                    n_outputs=24,
                    ngon_samples=12,
                )
                ok = bool(acc.ok)
                path = None
                if acc.maintains_regular_ngon:
                    rejected_maintained += 1
                if ok:
                    passed += 1
                    path = str(out_dir / f"pass_{n}_{trial:05d}.json")
                    polished = OrbitSeed(
                        id=f"search_n{n}_{trial:05d}",
                        family=polished.family,
                        n_bodies=polished.n_bodies,
                        G=polished.G,
                        masses=polished.masses,
                        period=polished.period,
                        positions=polished.positions,
                        velocities=polished.velocities,
                        names=polished.names,
                        symmetry="accepted_non_maintained_ngon",
                        source="choreography_search",
                        notes=f"trial={trial} residual={res_n:.3e}",
                        central_index=None,
                        verification=acc.to_dict(),
                    )
                    save_seed(polished, Path(path))
                    if res_n < best_res:
                        best_res = res_n
                        best_seed = polished
                        save_seed(polished, out_dir / "best.json")
                row = {
                    "trial": trial,
                    "residual": res_n,
                    "period": polished.period,
                    "ok_gate": ok,
                    "reason": acc.reason,
                    "maintains_regular_ngon": acc.maintains_regular_ngon,
                    "path": path,
                    "t_left_s": None if t_end is None else max(0.0, t_end - time.time()),
                }
                logf.write(json.dumps(row) + "\n")
                logf.flush()
                if on_progress:
                    on_progress(row)
                # Self-expanding: keep a live summary for unlimited runs
                summary = {
                    "n": n,
                    "trials": trial,
                    "passed_gate": passed,
                    "rejected_maintained_regular_ngon": rejected_maintained,
                    "best_residual": best_res if best_seed is not None else None,
                    "best_path": str(out_dir / "best.json") if best_seed is not None else None,
                    "out_dir": str(out_dir),
                    "wall_hours": wall_hours,
                    "status": "running",
                }
                (out_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
            except Exception as exc:  # pragma: no cover
                row = {
                    "trial": trial,
                    "error": str(exc),
                    "t_left_s": None if t_end is None else max(0.0, t_end - time.time()),
                }
                logf.write(json.dumps(row) + "\n")
                logf.flush()

    summary = {
        "n": n,
        "trials": trial,
        "passed_gate": passed,
        "rejected_maintained_regular_ngon": rejected_maintained,
        "best_residual": best_res if best_seed is not None else None,
        "best_path": str(out_dir / "best.json") if best_seed is not None else None,
        "out_dir": str(out_dir),
        "wall_hours": wall_hours,
        "status": "done",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
