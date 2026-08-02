"""Brightness-swap optics from the central observer.

Procedure (user-specified)
--------------------------
1. Partition time by which fairy is *farthest* from the observer.
2. Inside each contiguous segment, find the instant of minimum *visual*
   separation between any two fairies; that instant is the brightness-swap
   point for the segment.

Visual distance default: angular separation on the observer's sky
(``fairy_orbit.observe.optical_encounter.angular_separation``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from fairy_orbit.design.seeds import OrbitSeed
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate
from fairy_orbit.engine.trajectory import Trajectory
from fairy_orbit.observe.continuation import attach_central_mass
from fairy_orbit.observe.optical_encounter import angular_separation, delta_r_perp

from .scale import earth_scaled_radii

VisualMetric = Literal["angular", "perp"]


@dataclass(frozen=True)
class FarthestSegment:
    i0: int
    i1: int  # exclusive
    farthest_body: int  # fairy index in traj (1..N if central at 0)
    t0: float
    t1: float


@dataclass(frozen=True)
class BrightnessSwap:
    segment_index: int
    farthest_body: int
    t0: float
    t1: float
    swap_time: float
    swap_frame: int
    body_i: int
    body_j: int
    visual_distance: float
    metric: str
    distance_3d: float
    overlap_angular_sum: float | None


def _observer_and_fairies(
    traj: Trajectory, *, central_index: int = 0
) -> tuple[np.ndarray, list[int]]:
    n = int(traj.n_bodies)
    fairy_idx = [k for k in range(n) if k != int(central_index)]
    if not fairy_idx:
        raise ValueError("need at least one non-central body")
    return np.asarray(traj.positions[:, central_index], dtype=float), fairy_idx


def farthest_body_series(
    traj: Trajectory, *, central_index: int = 0
) -> np.ndarray:
    """Per-frame fairy index (traj index) of the body farthest from central."""
    obs_series, fairy_idx = _observer_and_fairies(traj, central_index=central_index)
    T = len(traj)
    out = np.empty(T, dtype=int)
    for t in range(T):
        obs = obs_series[t]
        best_i = fairy_idx[0]
        best_d = -1.0
        for i in fairy_idx:
            d = float(np.linalg.norm(traj.positions[t, i] - obs))
            if d > best_d:
                best_d = d
                best_i = i
        out[t] = best_i
    return out


def segment_by_farthest(
    traj: Trajectory, *, central_index: int = 0
) -> list[FarthestSegment]:
    series = farthest_body_series(traj, central_index=central_index)
    times = np.asarray(traj.times, dtype=float)
    segs: list[FarthestSegment] = []
    i0 = 0
    for t in range(1, len(series) + 1):
        if t == len(series) or series[t] != series[i0]:
            segs.append(
                FarthestSegment(
                    i0=i0,
                    i1=t,
                    farthest_body=int(series[i0]),
                    t0=float(times[i0]),
                    t1=float(times[t - 1]),
                )
            )
            i0 = t
    return segs


def _visual_distance(
    r_a: np.ndarray,
    r_b: np.ndarray,
    *,
    observer: np.ndarray,
    metric: VisualMetric,
) -> float:
    if metric == "angular":
        return float(angular_separation(r_a, r_b, observer=observer))
    if metric == "perp":
        return float(delta_r_perp(r_a, r_b, observer=observer))
    raise ValueError(f"unknown metric {metric!r}")


def brightness_swap_in_segment(
    traj: Trajectory,
    seg: FarthestSegment,
    *,
    central_index: int = 0,
    metric: VisualMetric = "angular",
    radii: np.ndarray | None = None,
) -> BrightnessSwap | None:
    """Min visual distance over fairy pairs inside ``seg`` → swap point."""
    _, fairy_idx = _observer_and_fairies(traj, central_index=central_index)
    if len(fairy_idx) < 2:
        return None
    best: tuple[float, int, int, int] | None = None  # dist, frame, i, j
    for t in range(seg.i0, seg.i1):
        obs = traj.positions[t, central_index]
        for a, i in enumerate(fairy_idx):
            for j in fairy_idx[a + 1 :]:
                d = _visual_distance(
                    traj.positions[t, i],
                    traj.positions[t, j],
                    observer=obs,
                    metric=metric,
                )
                if not np.isfinite(d):
                    continue
                if best is None or d < best[0]:
                    best = (d, t, i, j)
    if best is None:
        return None
    d, t, i, j = best
    obs = traj.positions[t, central_index]
    d3 = float(np.linalg.norm(traj.positions[t, i] - traj.positions[t, j]))
    alpha_sum = None
    if radii is not None:
        from fairy_orbit.observe.optical_encounter import angular_radius

        aA = angular_radius(traj.positions[t, i], float(radii[i]), observer=obs)
        aB = angular_radius(traj.positions[t, j], float(radii[j]), observer=obs)
        if np.isfinite(aA) and np.isfinite(aB):
            alpha_sum = float(aA + aB)
    return BrightnessSwap(
        segment_index=-1,
        farthest_body=seg.farthest_body,
        t0=seg.t0,
        t1=seg.t1,
        swap_time=float(traj.times[t]),
        swap_frame=int(t),
        body_i=int(i),
        body_j=int(j),
        visual_distance=float(d),
        metric=metric,
        distance_3d=d3,
        overlap_angular_sum=alpha_sum,
    )


def find_brightness_swaps(
    traj: Trajectory,
    *,
    central_index: int = 0,
    metric: VisualMetric = "angular",
    radii: np.ndarray | None = None,
) -> tuple[list[FarthestSegment], list[BrightnessSwap]]:
    segs = segment_by_farthest(traj, central_index=central_index)
    swaps: list[BrightnessSwap] = []
    for k, seg in enumerate(segs):
        sw = brightness_swap_in_segment(
            traj, seg, central_index=central_index, metric=metric, radii=radii
        )
        if sw is None:
            continue
        swaps.append(
            BrightnessSwap(
                segment_index=k,
                farthest_body=sw.farthest_body,
                t0=sw.t0,
                t1=sw.t1,
                swap_time=sw.swap_time,
                swap_frame=sw.swap_frame,
                body_i=sw.body_i,
                body_j=sw.body_j,
                visual_distance=sw.visual_distance,
                metric=sw.metric,
                distance_3d=sw.distance_3d,
                overlap_angular_sum=sw.overlap_angular_sum,
            )
        )
    return segs, swaps


def integrate_central_view(
    seed: OrbitSeed,
    M_c: float,
    *,
    periods: float = 2.0,
    n_outputs: int = 400,
) -> Trajectory:
    sys = attach_central_mass(seed, float(M_c))
    t_end = float(seed.period) * float(periods)
    return integrate(
        sys,
        t_end=t_end,
        n_outputs=max(int(n_outputs), 32),
        config=ReboundConfig(
            stop_on_escape=False,
            stop_on_collision=False,
            epsilon=0.0,
            dt=max(t_end / max(n_outputs * 2, 1), 1e-4),
            min_dt=1e-7,
        ),
    )


def run_brightness_swap_optics(
    seed: OrbitSeed,
    M_c: float,
    *,
    periods: float = 2.0,
    n_outputs: int = 400,
    r_frac: float = 0.02,
    metric: VisualMetric = "angular",
    initial_sun: int = 1,
    L_sun: float = 1.0,
    A: float = 0.25,
    A_earth: float = 0.30,
    phase_power: float = 1.0,  # unused (starry); kept for CLI compat
    role_rotate: bool = True,
    illumination: str = "near",
    occultation: bool = True,
) -> dict[str, Any]:
    from .photometry import (
        PhotoConfig,
        fairy_indices,
        flux_at_frame,
        flux_series_with_roles,
        period_variance_stats,
        rotate_sun_at_swap,
        sun_role_timeline,
    )

    del phase_power
    scale = earth_scaled_radii(seed, M_c, r_frac=r_frac)
    radii = np.asarray(scale["radii"], dtype=float)
    traj = integrate_central_view(
        seed, M_c, periods=periods, n_outputs=n_outputs
    )
    fairies = fairy_indices(traj.n_bodies, central_index=0)
    if initial_sun not in fairies:
        raise ValueError(f"initial_sun={initial_sun} must be a fairy index {fairies}")

    segs, swaps = find_brightness_swaps(
        traj,
        central_index=0,
        metric=metric,
        radii=radii,
    )
    sun_series = sun_role_timeline(
        len(traj), swaps, initial_sun=initial_sun, role_rotate=role_rotate
    )
    cfg = PhotoConfig(
        central_index=0,
        L_sun=float(L_sun),
        A=float(A),
        A_earth=float(A_earth),
        initial_sun=int(initial_sun),
        illumination=str(illumination),
        role_rotate=bool(role_rotate),
        occultation=bool(occultation),
        engine="starry",
    )

    swap_rows = []
    sun = int(initial_sun)
    for s in swaps:
        flux_before = flux_at_frame(
            traj,
            s.swap_frame,
            radii,
            sun_index=sun,
            central_index=0,
            L_sun=cfg.L_sun,
            A=cfg.A,
            illumination=cfg.illumination,  # type: ignore[arg-type]
            occultation=cfg.occultation,
        )
        sun_after = (
            rotate_sun_at_swap(sun, s.body_i, s.body_j) if role_rotate else sun
        )
        row = asdict(s)
        row["sun_before"] = sun
        row["sun_after"] = sun_after
        row["moons_before"] = [k for k in fairies if k != sun]
        row["photometry"] = asdict(flux_before)
        row["photometry"]["F_moon"] = row["photometry"].pop("F_moons")
        swap_rows.append(row)
        sun = sun_after

    series = flux_series_with_roles(
        traj,
        radii,
        sun_series,
        central_index=0,
        L_sun=cfg.L_sun,
        A=cfg.A,
        illumination=cfg.illumination,  # type: ignore[arg-type]
        occultation=cfg.occultation,
    )
    F_tot = np.array([f.F_total for f in series], dtype=float)
    F_sun_a = np.array([f.F_sun for f in series], dtype=float)
    F_moon_a = np.array([f.F_moons for f in series], dtype=float)
    stats_tot = period_variance_stats(F_tot)
    stats_sun = period_variance_stats(F_sun_a)
    stats_moon = period_variance_stats(F_moon_a)

    swap_times = [s.swap_time for s in swaps]
    dts = np.diff(swap_times) if len(swap_times) >= 2 else np.array([])
    role_events = [
        {
            "swap_frame": s.swap_frame,
            "swap_time": s.swap_time,
            "pair": [s.body_i, s.body_j],
            "sun_before": swap_rows[k]["sun_before"],
            "sun_after": swap_rows[k]["sun_after"],
        }
        for k, s in enumerate(swaps)
    ]
    return {
        "seed_id": seed.id,
        "M_c": float(M_c),
        "period": float(seed.period),
        "periods_integrated": float(periods),
        "metric": metric,
        "model": "1_sun_3_moons_equal_albedo",
        "evenness": "uneven" if role_rotate else "uniform",
        "photo": cfg.to_dict(),
        "scale": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in scale.items()
        },
        "n_segments": len(segs),
        "n_swaps": len(swaps),
        "segments": [asdict(s) for s in segs],
        "swaps": swap_rows,
        "role_events": role_events,
        "swap_intervals": dts.tolist(),
        "optical_period_median": float(np.median(dts)) if len(dts) else None,
        "optical_period_mean": float(np.mean(dts)) if len(dts) else None,
        "period_variance": {
            "F_total": stats_tot,
            "F_sun": stats_sun,
            "F_moon": stats_moon,
            "moon_fraction_mean": float(
                np.mean(F_moon_a / np.maximum(F_tot, 1e-300))
            ),
        },
    }
