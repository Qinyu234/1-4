"""Classify post-continuation (1+N) seeds by gravity ∧ visual-overlap channels.

Free-N choreography archives are out of scope — only orbits after mass
continuation (central present) are classified.

Classes
-------
* ``light_swap`` — ≥1 gravity encounter with ``optical_ok_perp``
* ``gravity_only`` — gravity encounters, none optically overlapping (perp)
* ``angular_distant`` — angular sky overlap with large 3D separation (no light_swap)
* ``quiet`` — no gravity encounter below threshold and no angular overlap
* ``error`` — integrate / evaluate failed
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from fairy_orbit.design.seeds import OrbitSeed
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate
from fairy_orbit.observe.continuation import attach_central_mass
from fairy_orbit.observe.encounters import find_encounters_optical
from fairy_orbit.observe.optical_encounter import (
    DEFAULT_LOG_RHO,
    radii_from_uniform_density,
    scan_visual_overlaps,
)

VisualClass = Literal[
    "light_swap",
    "gravity_only",
    "angular_distant",
    "quiet",
    "error",
]


@dataclass
class VisualSeedReport:
    trial_no: int | None
    residual: float | None
    period: float
    n_bodies: int
    seed_id: str
    M_c: float
    source_path: str
    klass: VisualClass
    n_encounters: int
    n_light_swap: int
    n_enc_angular_agree: int
    n_angular_hits: int
    n_angular_distant: int
    d_min: float
    encounter_threshold: float
    log_rho: float
    observer_ok: bool | None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fairy_mean_sep(seed: OrbitSeed) -> float:
    r = np.asarray(seed.positions, dtype=float)
    n = r.shape[0]
    if n < 2:
        return 1.0
    s = 0.0
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += float(np.linalg.norm(r[i] - r[j]))
            c += 1
    return s / max(c, 1)


def classify_continued_orbit(
    seed: OrbitSeed,
    M_c: float,
    *,
    log_rho: float = DEFAULT_LOG_RHO,
    n_outputs: int = 48,
    encounter_scale: float = 0.35,
    distant_factor: float = 5.0,
    source_path: str = "",
    residual: float | None = None,
) -> VisualSeedReport:
    """
    Integrate one period of 1+N system (central + fairies); classify.

    Observer = central body (index 0 after ``attach_central_mass``).
    """
    masses_fairy = np.asarray(seed.masses, dtype=float)
    mean_sep = _fairy_mean_sep(seed)
    thr = float(encounter_scale) * mean_sep
    Mc = float(M_c)

    if Mc <= 0.0:
        return VisualSeedReport(
            trial_no=None,
            residual=residual,
            period=float(seed.period),
            n_bodies=int(seed.n_bodies),
            seed_id=str(seed.id),
            M_c=Mc,
            source_path=str(source_path),
            klass="quiet",
            n_encounters=0,
            n_light_swap=0,
            n_enc_angular_agree=0,
            n_angular_hits=0,
            n_angular_distant=0,
            d_min=float("nan"),
            encounter_threshold=thr,
            log_rho=float(log_rho),
            observer_ok=None,
            note="M_c<=0: no central optics",
        )

    try:
        sys = attach_central_mass(seed, Mc)
        traj = integrate(
            sys,
            t_end=float(seed.period),
            n_outputs=int(n_outputs),
            config=ReboundConfig(
                stop_on_escape=False,
                stop_on_collision=False,
                epsilon=0.0,
                dt=2e-3,
                min_dt=1e-5,
            ),
        )
    except Exception as exc:  # pragma: no cover
        return VisualSeedReport(
            trial_no=None,
            residual=residual,
            period=float(seed.period),
            n_bodies=int(seed.n_bodies),
            seed_id=str(seed.id),
            M_c=Mc,
            source_path=str(source_path),
            klass="error",
            n_encounters=0,
            n_light_swap=0,
            n_enc_angular_agree=0,
            n_angular_hits=0,
            n_angular_distant=0,
            d_min=float("nan"),
            encounter_threshold=thr,
            log_rho=float(log_rho),
            observer_ok=None,
            note=str(exc),
        )

    # masses on traj: [M_c, *fairy]
    masses = np.concatenate([[Mc], masses_fairy])
    if traj.masses is not None and len(traj.masses) == seed.n_bodies + 1:
        masses = np.asarray(traj.masses, dtype=float)
    R = radii_from_uniform_density(masses, log_rho=log_rho)

    evs = find_encounters_optical(
        traj,
        threshold=thr,
        central_index=0,
        masses=masses,
        log_rho=log_rho,
    )
    n_swap = sum(1 for e in evs if e.light_swap)
    n_agree = sum(1 for e in evs if e.optical_ok_perp and e.optical_ok_angular)
    obs_flags = [e.observer_ok for e in evs if e.observer_ok is not None]
    observer_ok = all(obs_flags) if obs_flags else None

    d_min = min((e.distance for e in evs), default=float("inf"))
    if not evs:
        d_min = float("inf")
        for t in range(len(traj)):
            pos = traj.positions[t]
            for i in range(1, traj.n_bodies):
                for j in range(i + 1, traj.n_bodies):
                    d_min = min(d_min, float(np.linalg.norm(pos[i] - pos[j])))

    ang = scan_visual_overlaps(
        traj, masses, log_rho=log_rho, central_index=0, fairy_only=True
    )
    n_ang = len(ang)
    n_dist = 0
    for h in ang:
        Rs = float(R[h.i]) + float(R[h.j])
        if h.distance_3d > distant_factor * max(Rs, 1e-12):
            n_dist += 1

    if n_swap > 0:
        klass: VisualClass = "light_swap"
    elif len(evs) > 0:
        klass = "gravity_only"
    elif n_dist > 0 or n_ang > 0:
        klass = "angular_distant"
    else:
        klass = "quiet"

    return VisualSeedReport(
        trial_no=None,
        residual=residual,
        period=float(seed.period),
        n_bodies=int(seed.n_bodies),
        seed_id=str(seed.id),
        M_c=Mc,
        source_path=str(source_path),
        klass=klass,
        n_encounters=len(evs),
        n_light_swap=int(n_swap),
        n_enc_angular_agree=int(n_agree),
        n_angular_hits=int(n_ang),
        n_angular_distant=int(n_dist),
        d_min=float(d_min) if np.isfinite(d_min) else float("inf"),
        encounter_threshold=thr,
        log_rho=float(log_rho),
        observer_ok=observer_ok,
    )
