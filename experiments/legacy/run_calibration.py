#!/usr/bin/env python3
"""
PROMPT §2.4.1: Rodrigues same-(a,e) tetrahedron → ε_numerical(N).

At every time the four fairies remain regular-tetrahedron vertices
(theory shape error = 0). Measured drift is pure numerical noise and
calibrates thresholds for real resonant-chain experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.core import SystemConfig
from fairy_orbit.design import LadderParams, build_orbital_ladder, orbital_period
from fairy_orbit.engine import ReboundConfig, integrate
from fairy_orbit.observe import epsilon_at_orbit, measure_calibration
from fairy_orbit.viz.orbits import plot_orbits_xy

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Rodrigues tetrahedron → ε_numerical(N)")
    p.add_argument("--e", type=float, default=0.15)
    p.add_argument("--a", type=float, default=1.0, help="shared semi-major axis")
    p.add_argument("--mass-ratio", type=float, default=1e-4)
    p.add_argument("--t-end", type=float, default=200.0)
    p.add_argument("--n-outputs", type=int, default=800)
    p.add_argument("--epsilon", type=float, default=1e-6, help="IAS15 accuracy")
    p.add_argument("--min-dt", type=float, default=1e-5)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "calibration",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.t_end = 40.0
        args.n_outputs = 120

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    cfg = SystemConfig(mass_ratio=args.mass_ratio)
    params = LadderParams(geometry="calibration", eccentricity=args.e, a_inner=args.a)
    system = build_orbital_ladder(cfg, params)
    traj = integrate(
        system,
        t_end=args.t_end,
        n_outputs=args.n_outputs,
        config=ReboundConfig(epsilon=args.epsilon, min_dt=args.min_dt),
    )
    T_ref = orbital_period(args.a, cfg.mu)
    series = measure_calibration(traj, mu=cfg.mu, period_ref=T_ref)

    n_max = float(series.orbit_index[-1])
    sample_ns = [n for n in (1, 2, 5, 10, 20, 50) if n <= n_max + 1e-9]
    if not sample_ns:
        sample_ns = [float(series.orbit_index[-1])]
    table = [epsilon_at_orbit(series, float(n)) for n in sample_ns]

    plot_orbits_xy(
        traj, out / "xy.png", title="calibration: Rodrigues regular tetrahedron"
    )

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].semilogy(series.orbit_index, np.maximum(series.shape_error, 1e-16), "C0")
    ax[0].set_ylabel("shape error (edge CV)")
    ax[0].set_title("ε_numerical(N) — regular-tetrahedron drift")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[1].semilogy(series.orbit_index, np.maximum(series.amd_drift, 1e-16), "C1")
    ax[1].set_ylabel("|ΔAMD_total|")
    ax[1].set_xlabel("orbit index N = t / T")
    ax[1].set_title("AMD conservation floor")
    ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "epsilon_N.png", dpi=140)
    plt.close(fig)

    shape_error_max = float(np.nanmax(series.shape_error))
    amd_drift_max = float(np.nanmax(series.amd_drift))
    baseline_valid = traj.status == "success" and shape_error_max < 1e-6

    payload = {
        "geometry": "calibration",
        "construction": "rodrigues_vr_vt",
        "e": args.e,
        "a": args.a,
        "mass_ratio": args.mass_ratio,
        "t_end": args.t_end,
        "ias15_epsilon": args.epsilon,
        "min_dt": args.min_dt,
        "status": traj.status,
        "baseline_valid": baseline_valid,
        "period_ref": T_ref,
        "shape_error_max": shape_error_max,
        "amd_drift_max": amd_drift_max,
        "epsilon_table": table,
        "series": {
            "times": series.times.tolist(),
            "orbit_index": series.orbit_index.tolist(),
            "shape_error": series.shape_error.tolist(),
            "amd_drift": series.amd_drift.tolist(),
        },
    }
    (out / "epsilon_numerical.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# ε_numerical(N) calibration (PROMPT §2.4.1)",
        "",
        "Construction: shared Kepler `(a,e,f) ↔ (v_rad, v_tan)` (synonym), "
        "placed on tetrahedron vertices with **Rodrigues** copy of Newton "
        "`(r,v)`. Orbit-element tables are `from_state` of the same 6-DOF — "
        "not a second IC.",
        "",
        f"a={args.a}, e={args.e}, μ={args.mass_ratio:.0e}, "
        f"IAS15 ε={args.epsilon:g}, min_dt={args.min_dt:g}, status={traj.status}.",
        "",
        "| N (orbits) | shape_error | amd_drift |",
        "|------------|-------------|-----------|",
    ]
    for row in table:
        lines.append(
            f"| {row['n_orbit']:.0f} | {row['shape_error']:.3e} | {row['amd_drift']:.3e} |"
        )
    lines += [
        "",
        f"max shape_error = {payload['shape_error_max']:.3e}",
        f"max amd_drift = {payload['amd_drift_max']:.3e}",
        "",
        (
            "**VALID baseline:** use these values as numerical thresholds."
            if baseline_valid
            else "**INVALID baseline:** do not use these values as numerical thresholds; "
            "the trajectory escaped or physically left the regular-tetrahedron manifold."
        ),
        "",
    ]
    (out / "CALIBRATION.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"status={traj.status}")
    print(f"baseline_valid={baseline_valid}")
    print(f"shape_error_max={payload['shape_error_max']:.3e}")
    print(f"amd_drift_max={payload['amd_drift_max']:.3e}")
    print(f"out={out}")
    print(f"report={out / 'CALIBRATION.md'}")


if __name__ == "__main__":
    main()
