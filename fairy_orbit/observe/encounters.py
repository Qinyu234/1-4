"""Encounter Event index derived from Trajectory (PROMPT §6).

Gravity channel: pairwise fairy–fairy distance local minima.
Optical channel (optional): equal-density perp proxy + angular cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from fairy_orbit.engine.trajectory import Trajectory
from fairy_orbit.observe.optical_encounter import (
    DEFAULT_LOG_RHO,
    delta_r_perp,
    observer_validity,
    optical_overlap_angular,
    optical_overlap_perp,
    radii_from_uniform_density,
)


@dataclass
class EncounterEvent:
    time: float
    i: int
    j: int
    label_i: str
    label_j: str
    distance: float
    position_mid: np.ndarray
    # Optical annotations (None until annotate_encounters / find_encounters_optical)
    delta_r_perp: float | None = None
    radii_sum: float | None = None
    optical_ok_perp: bool | None = None
    optical_ok_angular: bool | None = None
    observer_ok: bool | None = None
    light_swap: bool | None = None  # gravity event ∧ optical_ok_perp


def find_encounters(
    trajectory: Trajectory,
    *,
    threshold: float,
    central_index: int | None = 0,
    min_separation_steps: int = 1,
) -> list[EncounterEvent]:
    """
    Local minima of pairwise fairy–fairy distance below `threshold`.

    ``central_index=None`` treats every body as a fairy (free-N / no star).
    """
    T, N, _ = trajectory.positions.shape
    events: list[EncounterEvent] = []
    if central_index is None:
        fairy_idx = list(range(N))
    else:
        fairy_idx = [k for k in range(N) if k != int(central_index)]

    for a, i in enumerate(fairy_idx):
        for j in fairy_idx[a + 1 :]:
            dists = np.linalg.norm(
                trajectory.positions[:, i, :] - trajectory.positions[:, j, :],
                axis=1,
            )
            for t in range(1, T - 1):
                if dists[t] > threshold:
                    continue
                if dists[t] <= dists[t - 1] and dists[t] <= dists[t + 1]:
                    if events and events[-1].i == i and events[-1].j == j:
                        if t - int(np.searchsorted(trajectory.times, events[-1].time)) < min_separation_steps:
                            # replace if closer
                            if dists[t] < events[-1].distance:
                                events[-1] = EncounterEvent(
                                    time=float(trajectory.times[t]),
                                    i=i,
                                    j=j,
                                    label_i=trajectory.labels[i],
                                    label_j=trajectory.labels[j],
                                    distance=float(dists[t]),
                                    position_mid=0.5
                                    * (
                                        trajectory.positions[t, i]
                                        + trajectory.positions[t, j]
                                    ).copy(),
                                )
                            continue
                    events.append(
                        EncounterEvent(
                            time=float(trajectory.times[t]),
                            i=i,
                            j=j,
                            label_i=trajectory.labels[i],
                            label_j=trajectory.labels[j],
                            distance=float(dists[t]),
                            position_mid=0.5
                            * (
                                trajectory.positions[t, i] + trajectory.positions[t, j]
                            ).copy(),
                        )
                    )
    events.sort(key=lambda e: e.time)
    return events


def annotate_encounters(
    events: list[EncounterEvent],
    trajectory: Trajectory,
    masses: np.ndarray | list[float] | None = None,
    *,
    log_rho: float = DEFAULT_LOG_RHO,
    central_index: int | None = 0,
    observer_max_ratio: float = 0.05,
    observer: np.ndarray | None = None,
) -> list[EncounterEvent]:
    """
    Attach optical channels to gravity encounters.

    * ``optical_ok_perp`` — encounter-conditioned proxy (defines light_swap).
    * ``optical_ok_angular`` — general angular cross-check at the same epoch.

    Observer: explicit ``observer``, else body ``central_index``, else origin
    (free-N COM-frame proxy).
    """
    m = (
        np.asarray(masses, dtype=float)
        if masses is not None
        else (
            np.asarray(trajectory.masses, dtype=float)
            if trajectory.masses is not None
            else None
        )
    )
    if m is None:
        raise ValueError("masses required to annotate optical channel")
    R = radii_from_uniform_density(m, log_rho=log_rho)
    if central_index is not None and 0 <= int(central_index) < len(R):
        R_c = float(R[int(central_index)])
    else:
        R_c = 0.0

    out: list[EncounterEvent] = []
    for ev in events:
        # nearest frame
        t_idx = int(np.argmin(np.abs(trajectory.times - ev.time)))
        pos = trajectory.positions[t_idx]
        if observer is not None:
            obs = np.asarray(observer, dtype=float).ravel()
        elif central_index is not None:
            obs = pos[int(central_index)]
        else:
            obs = None  # origin
        ra, rb = pos[ev.i], pos[ev.j]
        Ri, Rj = float(R[ev.i]), float(R[ev.j])
        dperp = delta_r_perp(ra, rb, observer=obs)
        ok_perp = optical_overlap_perp(ra, rb, Ri, Rj, observer=obs)
        ok_ang = optical_overlap_angular(ra, rb, Ri, Rj, observer=obs)
        if obs is None:
            a_enc = 0.5 * (
                float(np.linalg.norm(ra)) + float(np.linalg.norm(rb))
            )
        else:
            a_enc = 0.5 * (
                float(np.linalg.norm(ra - obs)) + float(np.linalg.norm(rb - obs))
            )
        obs_ok = (
            True
            if R_c <= 0.0
            else observer_validity(R_c, a_enc, max_ratio=observer_max_ratio)
        )
        out.append(
            replace(
                ev,
                delta_r_perp=dperp,
                radii_sum=Ri + Rj,
                optical_ok_perp=ok_perp,
                optical_ok_angular=ok_ang,
                observer_ok=obs_ok,
                light_swap=bool(ok_perp),
            )
        )
    return out


def find_encounters_optical(
    trajectory: Trajectory,
    *,
    threshold: float,
    central_index: int | None = 0,
    min_separation_steps: int = 1,
    masses: np.ndarray | list[float] | None = None,
    log_rho: float = DEFAULT_LOG_RHO,
    observer_max_ratio: float = 0.05,
    observer: np.ndarray | None = None,
) -> list[EncounterEvent]:
    """Gravity encounters + optical annotation (perp + angular cross-check)."""
    raw = find_encounters(
        trajectory,
        threshold=threshold,
        central_index=central_index,
        min_separation_steps=min_separation_steps,
    )
    return annotate_encounters(
        raw,
        trajectory,
        masses,
        log_rho=log_rho,
        central_index=central_index,
        observer_max_ratio=observer_max_ratio,
        observer=observer,
    )
