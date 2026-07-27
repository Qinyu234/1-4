#!/usr/bin/env python3
"""
Td error growth under dimensionless IC (m, β, e), ρ0=1, no collision.

Decomposes and tracks over time:
  A  integrator: coarse N-body vs fine N-body   (pos RMS)
  B  model:      group-orbit analytic vs fine    (pos RMS)
  C  breaking:   D_Td(t) on fine N-body
  E  energy:     |E(t)-E0|/|E0| on fine N-body

Also reports rough log-growth rates d log10(err) / dt on early window.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.design.tetra_eff import (
    build_td_system,
    eccentricity_from_beta_alpha,
    energy_from_beta,
    integrate_scale,
    omega_from_vt,
    polar_from_beta_e,
    states_at_rho_theta,
    td_orbit_from_ic,
    vc_scale,
)
from fairy_orbit.engine import ReboundConfig, integrate
from fairy_orbit.observe.calibration import td_breaking

ROOT = Path(__file__).resolve().parents[1]
RHO = 1.0


def _fairy_pos(row: np.ndarray) -> np.ndarray:
    return row[1:5] - row[0:1]


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def _log_slope(t: np.ndarray, y: np.ndarray, *, y_min: float = 1e-16) -> float:
    """Least-squares slope of log10(y) vs t on points with y > y_min."""
    mask = np.isfinite(y) & (y > y_min) & np.isfinite(t)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    x = t[mask]
    z = np.log10(np.maximum(y[mask], y_min))
    # early half of the usable window (capture initial growth)
    mid = x[0] + 0.5 * (x[-1] - x[0])
    early = x <= mid
    if np.count_nonzero(early) < 3:
        early = np.ones_like(x, dtype=bool)
    coef = np.polyfit(x[early], z[early], 1)
    return float(coef[0])


def run_case(
    m: float,
    beta: float,
    e: float,
    *,
    t_end: float,
    n_out: int,
    eps_fine: float,
    eps_coarse: float,
    min_dt: float,
    out: Path,
) -> dict:
    vr, vt, alpha = polar_from_beta_e(m, beta, e, rho=RHO)
    omega = omega_from_vt(RHO, vt)
    orbit = td_orbit_from_ic(m, RHO, vr, omega)
    system, _ = build_td_system(
        m, RHO, vr, omega, central_radius=0.0, fairy_radius=0.0
    )
    vc = vc_scale(m, RHO)
    tag = f"m{m:.0e}_b{beta:.2f}_e{e:.2f}".replace("+", "")
    case_dir = out / tag
    case_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{tag}] α={math.degrees(alpha):.1f}° vr/vc={vr/vc:.3f} vt/vc={vt/vc:.3f} "
        f"E={orbit.E:.4e} J={orbit.J:.4e} t_end={t_end:.3g}",
        flush=True,
    )

    fine = integrate(
        system,
        t_end=t_end,
        n_outputs=n_out,
        config=ReboundConfig(
            epsilon=eps_fine,
            min_dt=min_dt,
            stop_on_escape=False,
            stop_on_collision=False,
        ),
    )
    coarse = integrate(
        system,
        t_end=t_end,
        n_outputs=n_out,
        config=ReboundConfig(
            epsilon=eps_coarse,
            min_dt=min_dt,
            stop_on_escape=False,
            stop_on_collision=False,
        ),
    )

    n = min(len(fine.times), len(coarse.times))
    times = fine.times[:n]
    t_grid, rho_s, rhodot_s, theta_s = integrate_scale(
        orbit, float(times[-1]), n_steps=max(40 * n, 4000)
    )

    err_A = np.empty(n)
    err_B = np.empty(n)
    D_Td = np.empty(n)
    D_Td_ref = np.empty(n)
    rho_nb = np.empty(n)
    e0 = float(fine.energies[0])
    e_rel = np.abs(fine.energies[:n] - e0) / max(abs(e0), 1e-30)

    for k in range(n):
        t = float(times[k])
        pf = _fairy_pos(fine.positions[k])
        pc = _fairy_pos(coarse.positions[k])
        err_A[k] = _rms(pc, pf)
        rho = float(np.interp(t, t_grid, rho_s))
        rhodot = float(np.interp(t, t_grid, rhodot_s))
        theta = float(np.interp(t, t_grid, theta_s))
        ref = states_at_rho_theta(orbit, rho, rhodot, theta)
        pref = np.stack([ref[name][0] for name in ("T1", "T2", "T3", "T4")])
        err_B[k] = _rms(pf, pref)
        D_Td[k] = td_breaking(pf)
        D_Td_ref[k] = td_breaking(pref)
        rho_nb[k] = float(np.mean(np.linalg.norm(pf, axis=1)))

    def _cross(y: np.ndarray, level: float) -> float:
        idx = np.where(y > level)[0]
        return float(times[idx[0]]) if len(idx) else float("nan")

    summary = {
        "tag": tag,
        "m": m,
        "beta": beta,
        "e": e,
        "e_check": eccentricity_from_beta_alpha(beta, alpha),
        "alpha_deg": math.degrees(alpha),
        "vr": vr,
        "vt": vt,
        "vc": vc,
        "omega": omega,
        "J": orbit.J,
        "E": orbit.E,
        "E_formula": energy_from_beta(m, beta, rho=RHO),
        "mu_eff": orbit.mu_eff,
        "t_end": float(times[-1]),
        "D_Td_t0": float(D_Td[0]),
        "D_Td_max": float(np.nanmax(D_Td)),
        "D_Td_ref_max": float(np.nanmax(D_Td_ref)),
        "t_D_1e-3": _cross(D_Td, 1e-3),
        "t_D_0.1": _cross(D_Td, 0.1),
        "A_max": float(np.nanmax(err_A)),
        "B_max": float(np.nanmax(err_B)),
        "A_final": float(err_A[-1]),
        "B_final": float(err_B[-1]),
        "E_rel_max": float(np.nanmax(e_rel)),
        "slope_log10_A": _log_slope(times, err_A),
        "slope_log10_B": _log_slope(times, err_B),
        "slope_log10_D": _log_slope(times, D_Td),
        "slope_log10_E": _log_slope(times, e_rel),
        "fine_status": fine.status,
    }

    np.savez(
        case_dir / "series.npz",
        t=times,
        err_A=err_A,
        err_B=err_B,
        D_Td=D_Td,
        D_Td_ref=D_Td_ref,
        e_rel=e_rel,
        rho_nb=rho_nb,
        t_ref=t_grid,
        rho_ref=rho_s,
        theta_ref=theta_s,
    )
    (case_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axes[0, 0].semilogy(times, np.maximum(err_A, 1e-20), label="A coarse−fine")
    axes[0, 0].semilogy(times, np.maximum(err_B, 1e-20), label="B N-body−analytic")
    axes[0, 0].set_ylabel("pos RMS")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, which="both", alpha=0.3)
    axes[0, 0].set_title("error growth A vs B")

    axes[0, 1].semilogy(times, np.maximum(D_Td, 1e-20), "C3", label="D_Td N-body")
    axes[0, 1].semilogy(
        times, np.maximum(D_Td_ref, 1e-20), "C3", ls="--", label="D_Td analytic"
    )
    axes[0, 1].set_ylabel(r"$D_{Td}$")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, which="both", alpha=0.3)
    axes[0, 1].set_title("C: symmetry breaking")

    axes[1, 0].semilogy(times, np.maximum(e_rel, 1e-20), "C2")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel(r"$|E-E_0|/|E_0|$")
    axes[1, 0].grid(True, which="both", alpha=0.3)
    axes[1, 0].set_title("energy drift")

    axes[1, 1].plot(times, rho_nb, label="ρ N-body")
    axes[1, 1].plot(t_grid, rho_s, ls="--", label="ρ analytic")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel(r"$\rho$")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(rf"Td error growth — $m$={m:g}, $\beta$={beta:g}, $e$={e:g}")
    fig.tight_layout()
    fig.savefig(case_dir / "error_growth.png", dpi=140)
    plt.close(fig)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Td (m,β,e) error growth")
    p.add_argument(
        "--cases",
        default="1e-6,1.0,0.0;1e-3,1.0,0.0;1e-3,1.0,0.4;1e-2,1.0,0.0",
        help="semicolon-separated m,β,e triples",
    )
    p.add_argument("--t-end", type=float, default=10.0)
    p.add_argument("--n-outputs", type=int, default=400)
    p.add_argument("--eps-fine", type=float, default=1e-11)
    p.add_argument("--eps-coarse", type=float, default=1e-6)
    p.add_argument("--min-dt", type=float, default=1e-8)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "td_error_growth",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    cases: list[tuple[float, float, float]] = []
    for chunk in args.cases.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [float(x) for x in chunk.split(",")]
        if len(parts) != 3:
            raise SystemExit(f"bad case {chunk!r}; want m,beta,e")
        cases.append((parts[0], parts[1], parts[2]))

    t_end = args.t_end
    n_out = args.n_outputs
    if args.smoke:
        cases = cases[:2]
        t_end = 3.0
        n_out = 100

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for m, beta, e in cases:
        rows.append(
            run_case(
                m,
                beta,
                e,
                t_end=t_end,
                n_out=n_out,
                eps_fine=args.eps_fine,
                eps_coarse=args.eps_coarse,
                min_dt=args.min_dt,
                out=out,
            )
        )

    # overlay comparison of B and D_Td across cases
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for r in rows:
        data = np.load(out / r["tag"] / "series.npz")
        t = data["t"]
        axes[0].semilogy(t, np.maximum(data["err_A"], 1e-20), label=r["tag"])
        axes[1].semilogy(t, np.maximum(data["err_B"], 1e-20), label=r["tag"])
        axes[2].semilogy(t, np.maximum(data["D_Td"], 1e-20), label=r["tag"])
    axes[0].set_title("A integrator")
    axes[1].set_title("B model")
    axes[2].set_title(r"$D_{Td}$ breaking")
    for ax in axes:
        ax.set_xlabel("t")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Td error growth comparison")
    fig.tight_layout()
    fig.savefig(out / "compare_growth.png", dpi=140)
    plt.close(fig)

    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Td error growth — (m, β, e)",
        "",
        "A = coarse−fine pos RMS · B = N-body−analytic pos RMS · C = D_Td · E = energy drift.",
        "slope ≈ early-window d(log10 err)/dt.",
        "",
        "| tag | D(0) | t(D>0.1) | A_max | B_max | E_rel_max | slopeA | slopeB | slopeD |",
        "|-----|------|----------|-------|-------|-----------|--------|--------|--------|",
    ]
    for r in rows:
        tbrk = r["t_D_0.1"]
        t_s = f"{tbrk:.3g}" if math.isfinite(tbrk) else "—"

        def _s(x: float) -> str:
            return f"{x:.2f}" if math.isfinite(x) else "—"

        lines.append(
            f"| `{r['tag']}` | {r['D_Td_t0']:.2e} | {t_s} | "
            f"{r['A_max']:.2e} | {r['B_max']:.2e} | {r['E_rel_max']:.2e} | "
            f"{_s(r['slope_log10_A'])} | {_s(r['slope_log10_B'])} | {_s(r['slope_log10_D'])} |"
        )
    lines += ["", "Plots: per-case `*/error_growth.png`, overlay `compare_growth.png`."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"done → {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
