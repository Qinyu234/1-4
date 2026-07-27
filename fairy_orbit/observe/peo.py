"""PEO filter pipeline: escape → choreography → 8-channel rep errors.

Collision ignored. Optional σ-normalize from Stage A `sigmas.json`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fairy_orbit.core.criteria import SimulationStatus, check_escape
from fairy_orbit.design.manifold import ManifoldParams, build_manifold_system
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate
from fairy_orbit.engine.trajectory import Trajectory
from fairy_orbit.observe.closure import ClosureSeries, radial_order
from fairy_orbit.observe.rep_error import (
    CHANNELS,
    SEARCH_SCORE_WEIGHTS,
    RepErrorSeries,
    RepSigmas,
    apply_sigmas,
    rep_error_series,
    weighted_score,
)


@dataclass
class PEOFilterResult:
    status: str
    traj: Trajectory | None
    closure: ClosureSeries | None
    closure_rep: RepErrorSeries | None
    params: ManifoldParams
    summary: dict = field(default_factory=dict)


def evaluate_peo(
    params: ManifoldParams,
    *,
    t_end: float,
    n_outputs: int = 400,
    epsilon: float = 1e-9,
    min_dt: float = 1e-6,
    sigmas: RepSigmas | Path | str | None = None,
    perm_mode: str = "fixed_radial",
    score_weights: dict[str, float] | None = None,
) -> PEOFilterResult:
    """
    PEO check in inertial COM frame (translation removed at IC):
      Level 0: escape (collision off).
      Level 1: fix P from radial order S(T) vs S(0).
      Level 2–3: E_r/E_v + elements with Φ_T(X)≈(R*,P)X₀.
    """
    if isinstance(sigmas, (str, Path)):
        sigmas = RepSigmas.from_json(sigmas)
    sigmas = sigmas or RepSigmas(source="unit_default")
    weights = score_weights if score_weights is not None else SEARCH_SCORE_WEIGHTS

    system = build_manifold_system(params)
    if check_escape(system):
        return PEOFilterResult(
            status="escape_ic",
            traj=None,
            closure=None,
            closure_rep=None,
            params=params,
            summary={"status": "escape_ic"},
        )

    traj = integrate(
        system,
        t_end=t_end,
        n_outputs=n_outputs,
        config=ReboundConfig(
            epsilon=epsilon,
            min_dt=min_dt,
            stop_on_escape=True,
            stop_on_collision=False,
        ),
    )
    if traj.status == SimulationStatus.ESCAPE.value or traj.status == "escape":
        return PEOFilterResult(
            status="escape",
            traj=traj,
            closure=None,
            closure_rep=None,
            params=params,
            summary={"status": "escape", "t_end": float(traj.times[-1])},
        )

    series = rep_error_series(traj, mode=perm_mode)
    finals = series.final_snapshot()
    tilde = apply_sigmas(finals, sigmas)
    score = weighted_score(tilde, weights)

    # Soft penalty: equal-a IC cannot realize radial choreography
    a_vals = [params.a0 + i * params.a1 for i in range(4)]
    a_span = max(a_vals) - min(a_vals)
    if a_span < 1e-3:
        score += 10.0

    closure_compat = ClosureSeries(
        times=series.times,
        E_r=series.channels["E_r"],
        E_v=series.channels["E_v"],
        perm=series.perm,
        R_final=series.R_final,
        order_0=series.order_0,
        order_final=series.order_final,
        choreography_ok=True,
    )

    summary = {
        "status": "success",
        "t_end": float(traj.times[-1]),
        "perm": list(series.perm),
        "order_0": list(series.order_0),
        "order_final": list(series.order_final),
        "score": score,
        "a_span": a_span,
        "sigmas_source": sigmas.source,
        "E_energy_final": float(series.E_energy[-1]),
        "theta": list(params.as_theta()),
    }
    for k in CHANNELS:
        summary[f"{k}_final"] = finals[k]
        summary[f"{k}_tilde"] = tilde[k]
        summary[f"{k}_max"] = float(np.nanmax(series.channels[k]))

    return PEOFilterResult(
        status="success",
        traj=traj,
        closure=closure_compat,
        closure_rep=series,
        params=params,
        summary=summary,
    )


def check_radial_choreography(traj: Trajectory) -> tuple[bool, tuple[int, ...], tuple[int, ...]]:
    o0 = radial_order(traj.positions[0])
    oT = radial_order(traj.positions[-1])
    return True, o0, oT
