"""End-to-end diagnosis bundle for a ladder run."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fairy_orbit.core.body import System
from fairy_orbit.core.config import SystemConfig
from fairy_orbit.design.ladder import LadderParams
from fairy_orbit.engine.rebound_engine import ReboundConfig, compute_megno, integrate
from fairy_orbit.engine.trajectory import Trajectory
from fairy_orbit.observe.elements_series import ElementSeries, extract_element_series
from fairy_orbit.observe.amd import extract_amd_series
from fairy_orbit.observe.encounters import EncounterEvent, find_encounters
from fairy_orbit.observe.interest import a_order_changed, interestingness
from fairy_orbit.observe.resonance import ResonanceSeries, resonance_angles


@dataclass
class Diagnosis:
    trajectory: Trajectory
    elements: ElementSeries
    resonance: ResonanceSeries
    encounters: list[EncounterEvent]
    megno: float | None = None
    summary: dict = field(default_factory=dict)


def diagnose(
    system: System,
    config: SystemConfig,
    *,
    t_end: float,
    n_outputs: int = 1000,
    ladder: LadderParams | None = None,
    encounter_threshold: float | None = None,
    run_megno: bool = True,
    rebound_config: ReboundConfig | None = None,
) -> Diagnosis:
    """Integrate and extract ladder diagnostics."""
    traj = integrate(system, t_end=t_end, n_outputs=n_outputs, config=rebound_config)
    elements = extract_element_series(traj, mu=config.mu)

    period_ratios = None
    if ladder is not None:
        period_ratios = list(ladder.period_ratios)
    resonance = resonance_angles(elements, period_ratios=period_ratios)

    if encounter_threshold is None:
        # Overlap scale ~ e * Δa of adjacent rungs
        a = elements.a[0]
        if a.size >= 2:
            encounter_threshold = float(0.5 * np.mean(np.diff(a)) + config.fairy_radius * 10)
        else:
            encounter_threshold = 0.2

    encounters = find_encounters(traj, threshold=float(encounter_threshold))

    megno_val: float | None = None
    if run_megno and traj.status == "success":
        try:
            megno_val = compute_megno(system, t_end=t_end, config=rebound_config)
        except Exception:  # pragma: no cover
            megno_val = None

    a0 = elements.a[0]
    a1 = elements.a[-1]
    a_delta = a1 - a0
    # Resonance "lock" proxy: std of angle after removing linear trend
    res_std: list[float] = []
    for k in range(resonance.angles.shape[1]):
        phi = resonance.angles[:, k]
        if len(phi) < 3:
            res_std.append(float("nan"))
            continue
        t = resonance.times
        coeff = np.polyfit(t, phi, 1)
        resid = phi - np.polyval(coeff, t)
        res_std.append(float(np.std(resid)))

    order_changed = a_order_changed(a0.tolist(), a1.tolist())
    # Peak-to-peak a excursion (secular migration amplitude)
    a_ptp = float(np.mean(np.ptp(elements.a, axis=0)))

    # Fairy masses for AMD (exclude central)
    if traj.masses is not None and traj.masses.shape[0] == traj.positions.shape[1]:
        fairy_masses = np.asarray(traj.masses[1:], dtype=float)
    else:
        fairy_masses = np.full(elements.a.shape[1], config.fairy_mass, dtype=float)
    amd = extract_amd_series(elements, fairy_masses, mu=config.mu)

    summary = {
        "status": traj.status,
        "t_end": float(traj.times[-1]),
        "n_samples": len(traj),
        "n_encounters": len(encounters),
        "megno": megno_val,
        "a_initial": a0.tolist(),
        "a_final": a1.tolist(),
        "a_delta": a_delta.tolist(),
        "a_delta_rms": float(np.sqrt(np.mean(a_delta**2))),
        "a_ptp_mean": a_ptp,
        "a_order_changed": order_changed,
        "e_mean": np.nanmean(elements.e, axis=0).tolist(),
        "e_max": np.nanmax(elements.e, axis=0).tolist(),
        "amd_initial": amd.amd[0].tolist(),
        "amd_final": amd.amd[-1].tolist(),
        "amd_total_initial": float(amd.amd_total[0]),
        "amd_total_final": float(amd.amd_total[-1]),
        "amd_total_ptp": float(np.ptp(amd.amd_total)),
        "amd_body_ptp": np.ptp(amd.amd, axis=0).tolist(),
        "resonance_angle_std": res_std,
        "resonance_std_mean": float(np.nanmean(res_std)) if res_std else float("nan"),
        "energy_drift": float(
            abs(traj.energies[-1] - traj.energies[0]) / max(abs(traj.energies[0]), 1e-30)
        ),
        "resonance_pairs": resonance.pair_labels,
    }
    summary["interest"] = interestingness(summary)
    return Diagnosis(
        trajectory=traj,
        elements=elements,
        resonance=resonance,
        encounters=encounters,
        megno=megno_val,
        summary=summary,
    )
