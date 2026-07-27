#!/usr/bin/env python3
"""
Rodrigues regular-tetrahedron IC: error growth vs integrator algorithm.

Metrics (vs time / orbit index):
  - relative energy drift |E(t)-E0|/|E0|
  - relative |L| drift
  - fairy position RMS vs high-precision IAS15 reference
  - tetrahedron shape error (mostly physical breakup; shown for context)

Algorithms: IAS15 (ε), WHFast (dt), Leapfrog (dt).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.core import SystemConfig
from fairy_orbit.design import LadderParams, build_orbital_ladder, orbital_period
from fairy_orbit.engine import ReboundConfig, integrate
from fairy_orbit.observe.calibration import measure_calibration

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    integrator: str
    epsilon: float | None = None
    dt: float = 0.01
    is_reference: bool = False


DEFAULT_ALGOS = (
    AlgoSpec("ias15_1e-11 (ref)", "ias15", epsilon=1e-11, is_reference=True),
    AlgoSpec("ias15_1e-9", "ias15", epsilon=1e-9),
    AlgoSpec("ias15_1e-6", "ias15", epsilon=1e-6),
    AlgoSpec("whfast_1e-3", "whfast", dt=1e-3),
    AlgoSpec("whfast_1e-2", "whfast", dt=1e-2),
    AlgoSpec("leapfrog_1e-3", "leapfrog", dt=1e-3),
    AlgoSpec("leapfrog_1e-2", "leapfrog", dt=1e-2),
)


def _rel_energy(energies: np.ndarray) -> np.ndarray:
    e0 = float(energies[0])
    scale = max(abs(e0), 1e-30)
    return np.abs(energies - e0) / scale


def _rel_angmom(L: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(L, axis=1)
    l0 = float(mag[0])
    scale = max(abs(l0), 1e-30)
    return np.abs(mag - l0) / scale


def _pos_rms_vs_ref(pos: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """RMS over central+4 fairies (all coords) at shared sample indices."""
    n = min(len(pos), len(ref))
    diff = pos[:n] - ref[:n]
    return np.sqrt(np.mean(diff.reshape(n, -1) ** 2, axis=1))


def _t_bound(energies: np.ndarray, times: np.ndarray, thresh: float = 1e-6) -> float:
    e0 = float(energies[0])
    scale = max(abs(e0), 1e-30)
    for i, e in enumerate(energies):
        if not np.isfinite(e) or abs(e - e0) / scale > thresh:
            return float(times[max(i - 1, 0)])
    return float(times[-1])


def _run_one(system, spec: AlgoSpec, *, t_end: float, n_out: int, min_dt: float):
    cfg = ReboundConfig(
        integrator=spec.integrator,
        epsilon=spec.epsilon,
        dt=spec.dt,
        min_dt=min_dt,
        stop_on_escape=False,
        stop_on_collision=False,
    )
    return integrate(system, t_end=t_end, n_outputs=n_out, config=cfg)


def main() -> None:
    p = argparse.ArgumentParser(description="Tetrahedron error growth across integrators")
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--e", type=float, default=0.0)
    p.add_argument("--mass-ratio", type=float, default=1e-6)
    p.add_argument("--t-end", type=float, default=10.0)
    p.add_argument("--n-outputs", type=int, default=400)
    p.add_argument("--min-dt", type=float, default=1e-8)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "tetra_error_growth",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.t_end = 3.0
        args.n_outputs = 80

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    sys_cfg = SystemConfig(mass_ratio=args.mass_ratio)
    params = LadderParams(geometry="calibration", eccentricity=args.e, a_inner=args.a)
    system = build_orbital_ladder(sys_cfg, params)
    T_ref = orbital_period(args.a, sys_cfg.mu)

    # Reference first
    ref_spec = next(s for s in DEFAULT_ALGOS if s.is_reference)
    print(f"reference: {ref_spec.name} ...", flush=True)
    ref_traj = _run_one(
        system, ref_spec, t_end=args.t_end, n_out=args.n_outputs, min_dt=args.min_dt
    )
    ref_pos = ref_traj.positions

    series_by_name: dict[str, dict] = {}
    summary_rows = []

    for spec in DEFAULT_ALGOS:
        print(f"run: {spec.name} ...", flush=True)
        traj = (
            ref_traj
            if spec.is_reference
            else _run_one(
                system, spec, t_end=args.t_end, n_out=args.n_outputs, min_dt=args.min_dt
            )
        )
        cal = measure_calibration(traj, mu=sys_cfg.mu, period_ref=T_ref)
        e_rel = _rel_energy(traj.energies)
        l_rel = _rel_angmom(traj.angular_momenta)
        pos_rms = _pos_rms_vs_ref(traj.positions, ref_pos)
        n = len(pos_rms)
        t_bound = _t_bound(traj.energies, traj.times)

        # Growth rates on bound prefix (log10 slope vs orbit index)
        mask = traj.times[:n] <= t_bound + 1e-12
        if int(np.count_nonzero(mask)) < 4:
            mask = np.ones(n, dtype=bool)
        N = cal.orbit_index[:n][mask]
        # Avoid zeros for log fit
        e_fit = np.maximum(e_rel[:n][mask], 1e-20)
        p_fit = np.maximum(pos_rms[mask], 1e-20)
        if len(N) >= 4 and float(N[-1] - N[0]) > 1e-6:
            e_slope = float(np.polyfit(N, np.log10(e_fit), 1)[0])
            p_slope = float(np.polyfit(N, np.log10(p_fit), 1)[0])
        else:
            e_slope = float("nan")
            p_slope = float("nan")

        row = {
            "name": spec.name,
            "integrator": spec.integrator,
            "epsilon": spec.epsilon,
            "dt": spec.dt if spec.integrator != "ias15" else None,
            "status": traj.status,
            "t_last": float(traj.times[-1]),
            "t_bound": t_bound,
            "N_bound": t_bound / T_ref,
            "energy_rel_final_bound": float(e_rel[:n][mask][-1]),
            "energy_rel_max": float(np.nanmax(e_rel)),
            "L_rel_final_bound": float(l_rel[:n][mask][-1]),
            "pos_rms_final_bound": float(pos_rms[mask][-1]),
            "pos_rms_max": float(np.nanmax(pos_rms)),
            "shape_error_at_bound": float(cal.shape_error[:n][mask][-1]),
            "dlog10E_dN": e_slope,
            "dlog10Pos_dN": p_slope,
            "is_reference": spec.is_reference,
        }
        summary_rows.append(row)
        series_by_name[spec.name] = {
            "times": traj.times[:n].tolist(),
            "orbit_index": cal.orbit_index[:n].tolist(),
            "energy_rel": e_rel[:n].tolist(),
            "L_rel": l_rel[:n].tolist(),
            "pos_rms": pos_rms.tolist(),
            "shape_error": cal.shape_error[:n].tolist(),
            "t_bound": t_bound,
        }

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    ax_e, ax_l, ax_p, ax_s = axes.ravel()
    for name, ser in series_by_name.items():
        N = np.asarray(ser["orbit_index"])
        tb = ser["t_bound"] / T_ref
        style = {"lw": 2.0 if "ref" in name else 1.2}
        ax_e.semilogy(N, np.maximum(ser["energy_rel"], 1e-20), label=name, **style)
        ax_l.semilogy(N, np.maximum(ser["L_rel"], 1e-20), label=name, **style)
        ax_p.semilogy(N, np.maximum(ser["pos_rms"], 1e-20), label=name, **style)
        ax_s.semilogy(N, np.maximum(ser["shape_error"], 1e-20), label=name, **style)
        for ax in (ax_e, ax_l, ax_p, ax_s):
            ax.axvline(tb, color="0.5", ls=":", alpha=0.25)

    ax_e.set_ylabel("|ΔE|/|E0|")
    ax_e.set_title("Energy error growth")
    ax_l.set_ylabel("|Δ|L|| / |L0|")
    ax_l.set_title("Angular-momentum error")
    ax_p.set_ylabel("RMS |x − x_ref|")
    ax_p.set_title(f"Position vs {ref_spec.name}")
    ax_s.set_ylabel("shape edge CV")
    ax_s.set_title("Shape error (mostly physical)")
    for ax in (ax_e, ax_l, ax_p, ax_s):
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("orbit index N = t / T")
    ax_e.legend(fontsize=7, loc="best")
    fig.suptitle(
        f"Tetra Rodrigues IC — μ={args.mass_ratio:g}, e={args.e}, a={args.a}, "
        f"t_end={args.t_end}"
    )
    fig.tight_layout()
    fig.savefig(out / "error_growth.png", dpi=140)
    plt.close(fig)

    # Bound-window zoom (common t_bound among IAS15 runs)
    t_zoom = min(r["t_bound"] for r in summary_rows if r["integrator"] == "ias15")
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for name, ser in series_by_name.items():
        N = np.asarray(ser["orbit_index"])
        mask = np.asarray(ser["times"]) <= t_zoom + 1e-12
        axes2[0].semilogy(
            N[mask], np.maximum(np.asarray(ser["energy_rel"])[mask], 1e-20), label=name
        )
        axes2[1].semilogy(
            N[mask], np.maximum(np.asarray(ser["pos_rms"])[mask], 1e-20), label=name
        )
    axes2[0].set_title(f"|ΔE|/|E0| (t ≤ {t_zoom:.2f})")
    axes2[1].set_title(f"pos RMS vs ref (t ≤ {t_zoom:.2f})")
    for ax in axes2:
        ax.set_xlabel("N")
        ax.grid(True, which="both", alpha=0.3)
    axes2[0].legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(out / "error_growth_bound.png", dpi=140)
    plt.close(fig2)

    payload = {
        "params": {
            "a": args.a,
            "e": args.e,
            "mass_ratio": args.mass_ratio,
            "t_end": args.t_end,
            "n_outputs": args.n_outputs,
            "T_ref": T_ref,
            "reference": ref_spec.name,
            "note": (
                "Shape error on Rodrigues same-a tetra is largely physical "
                "(config leaves regular tetra). Use energy / |L| / pos-vs-ref "
                "for integrator comparison. Dotted vertical ≈ energy-spike bound."
            ),
        },
        "summary": summary_rows,
        "algos": [asdict(s) for s in DEFAULT_ALGOS],
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Compact series for replot (omit if huge)
    (out / "series.json").write_text(json.dumps(series_by_name), encoding="utf-8")

    lines = [
        "# Tetrahedron integrator error growth",
        "",
        f"IC: Rodrigues calibration tetra, a={args.a}, e={args.e}, μ={args.mass_ratio:g}, "
        f"t_end={args.t_end}, T≈{T_ref:.4g}.",
        f"Reference: `{ref_spec.name}`.",
        "",
        "Shape error is **mostly physical** (regular tetra not preserved under "
        "central gravity). Integrator ranking uses energy / |L| / position-vs-ref "
        "on the bound prefix (before |ΔE|/|E0| > 1e-6).",
        "",
        "| algo | t_bound | E_rel@bound | |L|_rel@bound | posRMS@bound | "
        "dlog10E/dN | dlog10Pos/dN |",
        "|------|---------|-------------|---------------|--------------|"
        "------------|--------------|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['name']} | {r['t_bound']:.3g} | {r['energy_rel_final_bound']:.3e} | "
            f"{r['L_rel_final_bound']:.3e} | {r['pos_rms_final_bound']:.3e} | "
            f"{r['dlog10E_dN']:.2f} | {r['dlog10Pos_dN']:.2f} |"
        )
    lines += [
        "",
        "Plots: `error_growth.png`, `error_growth_bound.png`.",
        "",
        "## Reading the slopes",
        "",
        "- `dlog10E/dN ≈ 0`: error floor / saturated (good for IAS15 short window).",
        "- positive slope: secular accumulation (typical of fixed-step / symplectic "
        "truncation on this non-hierarchical IC).",
        "- After `t_bound`, energy spikes from physical unbinding — not integrator ranking.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done → {out / 'REPORT.md'}", flush=True)
    print("\n".join(lines[7:7 + len(summary_rows) + 2]))


if __name__ == "__main__":
    main()
