"""Equal-mass choreography search (PROMPT construct path).

Practical loop for long campaigns: multi-start polish of free-N IC so that
§3.2 residual x_i(T/n)=R x_P(i)(0) (r and v) → 0, with a soft collision
penalty. Seeds from regular polygon RE; not Bayes.

True truncated-Fourier + action (Vanderbei-style) can replace the IC vector
later; the campaign API stays the same.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares

from fairy_orbit.core.body import Body, System, to_com_inertial_frame
from fairy_orbit.design.seeds import (
    OrbitSeed,
    build_free_polygon_seed,
    save_seed,
)
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate
from fairy_orbit.observe.choreography_verify import (
    cyclic_role_perm,
    verify_choreography_Tn,
)
from fairy_orbit.observe.closure import closure_for_perm


def _pack(seed: OrbitSeed) -> np.ndarray:
    """State vector: positions + velocities (period fixed on template)."""
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
    pen = _collision_penalty(r0)
    out = np.concatenate(chunks).astype(float)
    if pen > 0:
        out = np.concatenate([out, np.array([pen])])
    return out


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


def run_choreography_search(
    n: int,
    *,
    wall_hours: float = 8.0,
    shift: int = 1,
    seed_scale: float = 0.05,
    rng: np.random.Generator | None = None,
    out_dir: Path | None = None,
    max_nfev: int = 10,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Long-running multi-start §3.2 polish for free equal-mass N-body.

    Returns summary with best residual and paths of gate-passing seeds.
    """
    rng = rng or np.random.default_rng(n * 10007 + 17)
    out_dir = Path(out_dir or f"experiments/output/choreography_search_n{n}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "trials.jsonl"

    family = f"free_{n}"
    base = build_free_polygon_seed(n, seed_id=f"{family}_square_re" if n == 4 else f"{family}_pentagon_re", family=family)
    # fix id naming
    if n == 4:
        base = build_free_polygon_seed(4, seed_id="free_4_square_re", family="free_4")
    else:
        base = build_free_polygon_seed(5, seed_id="free_5_pentagon_re", family="free_5")

    t_end = time.time() + float(wall_hours) * 3600.0
    best_res = float("inf")
    best_seed: OrbitSeed | None = None
    trials: list[SearchTrial] = []
    trial = 0
    passed = 0

    with log_path.open("a", encoding="utf-8") as logf:
        while time.time() < t_end:
            trial += 1
            y = _pack(base)
            noise = rng.normal(0.0, seed_scale, size=y.shape)
            # Mild period jitter via template copy
            period = float(base.period) * float(np.exp(rng.normal(0.0, 0.02)))
            start_tmpl = OrbitSeed(
                id=base.id,
                family=base.family,
                n_bodies=base.n_bodies,
                G=base.G,
                masses=base.masses,
                period=period,
                positions=base.positions,
                velocities=base.velocities,
                names=base.names,
                symmetry=base.symmetry,
                source=base.source,
                notes=base.notes,
                central_index=None,
            )
            start = _unpack(y + noise, start_tmpl)
            try:
                polished, res_n = polish_seed(start, shift=shift, max_nfev=max_nfev)
                gate = verify_choreography_Tn(
                    polished.to_system(),
                    polished.period,
                    shift=shift,
                    atol_rel=1e-5,
                    n_outputs=24,
                )
                ok = bool(gate.ok)
                path = None
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
                        symmetry=polished.symmetry,
                        source="choreography_search",
                        notes=f"trial={trial} residual={res_n:.3e}",
                        central_index=None,
                        verification=gate.to_dict(),
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
                    "path": path,
                    "t_left_s": max(0.0, t_end - time.time()),
                }
                logf.write(json.dumps(row) + "\n")
                logf.flush()
                trials.append(
                    SearchTrial(
                        trial=trial,
                        residual=res_n,
                        period=polished.period,
                        ok_gate=ok,
                        path=path,
                    )
                )
                if on_progress:
                    on_progress(row)
            except Exception as exc:  # pragma: no cover
                row = {
                    "trial": trial,
                    "error": str(exc),
                    "t_left_s": max(0.0, t_end - time.time()),
                }
                logf.write(json.dumps(row) + "\n")
                logf.flush()

    summary = {
        "n": n,
        "trials": trial,
        "passed_gate": passed,
        "best_residual": best_res if best_res < float("inf") else None,
        "best_path": str(out_dir / "best.json") if best_seed is not None else None,
        "out_dir": str(out_dir),
        "wall_hours": wall_hours,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
