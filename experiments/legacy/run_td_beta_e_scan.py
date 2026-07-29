#!/usr/bin/env python3
"""
Scan Td error / symmetry-breaking vs (β, e) at fixed m (ρ=1, no collision).

For each valid (β, e): integrate fine N-body, track D_Td(t) and B=posRMS vs
group-orbit analytic. Report heatmaps of t(D>0.1), D_max, B_max.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.design.tetra_eff import (
    build_td_system,
    e_min_for_beta,
    integrate_scale,
    is_valid_beta_e,
    omega_from_vt,
    polar_from_beta_e,
    states_at_rho_theta,
    td_orbit_from_ic,
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


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def run_one(
    m: float,
    beta: float,
    e: float,
    *,
    t_end: float,
    n_out: int,
    epsilon: float,
    min_dt: float,
) -> dict:
    vr, vt, alpha = polar_from_beta_e(m, beta, e, rho=RHO)
    omega = omega_from_vt(RHO, vt)
    orbit = td_orbit_from_ic(m, RHO, vr, omega)
    system, _ = build_td_system(
        m, RHO, vr, omega, central_radius=0.0, fairy_radius=0.0
    )

    t0 = time.perf_counter()
    traj = integrate(
        system,
        t_end=t_end,
        n_outputs=n_out,
        config=ReboundConfig(
            epsilon=epsilon,
            min_dt=min_dt,
            stop_on_escape=False,
            stop_on_collision=False,
        ),
    )
    elapsed = time.perf_counter() - t0

    times = traj.times
    n = len(times)
    t_grid, rho_s, rhodot_s, theta_s = integrate_scale(
        orbit, float(times[-1]), n_steps=max(20 * n, 2000)
    )

    D = np.empty(n)
    err_B = np.empty(n)
    rho_nb = np.empty(n)
    for k in range(n):
        t = float(times[k])
        pf = _fairy_pos(traj.positions[k])
        D[k] = td_breaking(pf)
        rho = float(np.interp(t, t_grid, rho_s))
        rhodot = float(np.interp(t, t_grid, rhodot_s))
        theta = float(np.interp(t, t_grid, theta_s))
        ref = states_at_rho_theta(orbit, rho, rhodot, theta)
        pref = np.stack([ref[name][0] for name in ("T1", "T2", "T3", "T4")])
        err_B[k] = _rms(pf, pref)
        rho_nb[k] = float(np.mean(np.linalg.norm(pf, axis=1)))

    e0 = float(traj.energies[0])
    e_rel = float(np.nanmax(np.abs(traj.energies - e0) / max(abs(e0), 1e-30)))

    def _cross(level: float) -> float:
        idx = np.where(D > level)[0]
        return float(times[idx[0]]) if len(idx) else float("nan")

    return {
        "m": m,
        "beta": beta,
        "e": e,
        "alpha_deg": math.degrees(alpha),
        "vr": vr,
        "vt": vt,
        "J": orbit.J,
        "E": orbit.E,
        "elapsed_s": elapsed,
        "status": traj.status,
        "D_Td_t0": float(D[0]),
        "D_Td_max": float(np.nanmax(D)),
        "t_D_1e-3": _cross(1e-3),
        "t_D_0.1": _cross(0.1),
        "t_D_1": _cross(1.0),
        "B_max": float(np.nanmax(err_B)),
        "B_at_1": float(err_B[np.argmin(np.abs(times - 1.0))]),
        "E_rel_max": e_rel,
        "rho_ptp": float(np.nanmax(rho_nb) - np.nanmin(rho_nb)),
        "t": times.tolist(),
        "D_Td": D.tolist(),
        "err_B": err_B.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Td (β,e) influence scan")
    p.add_argument("--m", type=float, default=1e-3)
    p.add_argument("--beta", default="0.6,0.75,0.9,1.0,1.1,1.25")
    p.add_argument("--e", default="0.0,0.2,0.4,0.6,0.8")
    p.add_argument("--t-end", type=float, default=8.0)
    p.add_argument("--n-outputs", type=int, default=250)
    p.add_argument("--epsilon", type=float, default=1e-9)
    p.add_argument("--min-dt", type=float, default=1e-5)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "td_beta_e_scan",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    betas = _parse_floats(args.beta)
    es = _parse_floats(args.e)
    if args.smoke:
        betas = [0.9, 1.0, 1.1]
        es = [0.0, 0.4, 0.8]
        args.t_end = 4.0
        args.n_outputs = 120

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "runs.jsonl"
    if jsonl.exists():
        jsonl.unlink()

    rows: list[dict] = []
    skipped = 0
    print(
        f"β×e scan m={args.m:g} β={betas} e={es} t_end={args.t_end}",
        flush=True,
    )
    for beta in betas:
        for e in es:
            if not is_valid_beta_e(beta, e):
                skipped += 1
                print(
                    f"  skip β={beta:.2f} e={e:.2f} "
                    f"(e_min={e_min_for_beta(beta):.3f})",
                    flush=True,
                )
                continue
            row = run_one(
                args.m,
                beta,
                e,
                t_end=args.t_end,
                n_out=args.n_outputs,
                epsilon=args.epsilon,
                min_dt=args.min_dt,
            )
            # keep series out of summary json size; write separately
            series = {
                "t": row.pop("t"),
                "D_Td": row.pop("D_Td"),
                "err_B": row.pop("err_B"),
            }
            tag = f"b{beta:.2f}_e{e:.2f}"
            np.savez(
                out / f"{tag}.npz",
                t=np.array(series["t"]),
                D_Td=np.array(series["D_Td"]),
                err_B=np.array(series["err_B"]),
            )
            rows.append(row)
            with jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            tbrk = row["t_D_0.1"]
            print(
                f"  β={beta:.2f} e={e:.2f} α={row['alpha_deg']:.0f}° "
                f"t(D>0.1)={tbrk if math.isfinite(tbrk) else '—'} "
                f"Dmax={row['D_Td_max']:.3e} Bmax={row['B_max']:.3e}",
                flush=True,
            )

    # --- heatmaps ---
    b_idx = {b: i for i, b in enumerate(betas)}
    e_idx = {e: i for i, e in enumerate(es)}
    shape = (len(betas), len(es))
    t01 = np.full(shape, np.nan)
    dmax = np.full(shape, np.nan)
    bmax = np.full(shape, np.nan)
    for r in rows:
        i, j = b_idx[r["beta"]], e_idx[r["e"]]
        t01[i, j] = r["t_D_0.1"] if math.isfinite(r["t_D_0.1"]) else args.t_end
        dmax[i, j] = r["D_Td_max"]
        bmax[i, j] = r["B_max"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, Z, title, fmt in (
        (axes[0], t01, r"$t(D_{Td}>0.1)$", ".2f"),
        (axes[1], np.log10(np.maximum(dmax, 1e-16)), r"$\log_{10} D_{\max}$", ".1f"),
        (axes[2], np.log10(np.maximum(bmax, 1e-16)), r"$\log_{10} B_{\max}$", ".1f"),
    ):
        im = ax.imshow(
            Z.T,
            origin="lower",
            aspect="auto",
            extent=[
                betas[0] - 0.5 * (betas[1] - betas[0] if len(betas) > 1 else 0.1),
                betas[-1] + 0.5 * (betas[1] - betas[0] if len(betas) > 1 else 0.1),
                es[0] - 0.5 * (es[1] - es[0] if len(es) > 1 else 0.1),
                es[-1] + 0.5 * (es[1] - es[0] if len(es) > 1 else 0.1),
            ],
            cmap="magma",
        )
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$e$")
        ax.set_title(title)
        # mark e_min(β)
        bb = np.linspace(betas[0], betas[-1], 80)
        ax.plot(bb, [e_min_for_beta(b) for b in bb], "c--", lw=1.2, label=r"$e_{\min}$")
        ax.legend(fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(rf"Td $(\beta,e)$ scan — $m$={args.m:g}, $\rho=1$")
    fig.tight_layout()
    fig.savefig(out / "beta_e_heatmaps.png", dpi=140)
    plt.close(fig)

    # --- growth curves colored by β / e ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cmap_b = plt.cm.viridis
    cmap_e = plt.cm.plasma
    b_norm = plt.Normalize(min(betas), max(betas))
    e_norm = plt.Normalize(min(es), max(es))
    for r in rows:
        data = np.load(out / f"b{r['beta']:.2f}_e{r['e']:.2f}.npz")
        axes[0].semilogy(
            data["t"],
            np.maximum(data["D_Td"], 1e-18),
            color=cmap_b(b_norm(r["beta"])),
            alpha=0.85,
            lw=1.2,
        )
        axes[1].semilogy(
            data["t"],
            np.maximum(data["D_Td"], 1e-18),
            color=cmap_e(e_norm(r["e"])),
            alpha=0.85,
            lw=1.2,
        )
    axes[0].set_title(r"$D_{Td}(t)$ colored by $\beta$")
    axes[1].set_title(r"$D_{Td}(t)$ colored by $e$")
    for ax in axes:
        ax.set_xlabel("t")
        ax.set_ylabel(r"$D_{Td}$")
        ax.grid(True, which="both", alpha=0.3)
        ax.axhline(0.1, color="k", ls=":", lw=0.8)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=b_norm, cmap=cmap_b),
        ax=axes[0],
        label=r"$\beta$",
        fraction=0.046,
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=e_norm, cmap=cmap_e),
        ax=axes[1],
        label=r"$e$",
        fraction=0.046,
    )
    fig.suptitle(rf"$D_{{Td}}$ growth — $m$={args.m:g}")
    fig.tight_layout()
    fig.savefig(out / "D_Td_curves.png", dpi=140)
    plt.close(fig)

    # scatter: β,e → t_break
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = [r["beta"] for r in rows]
    ys = [r["e"] for r in rows]
    cs = [
        r["t_D_0.1"] if math.isfinite(r["t_D_0.1"]) else args.t_end for r in rows
    ]
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=80, edgecolors="k", linewidths=0.4)
    bb = np.linspace(min(betas) - 0.05, max(betas) + 0.05, 80)
    ax.plot(bb, [e_min_for_beta(b) for b in bb], "r--", label=r"$e_{\min}(\beta)$")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$e$")
    ax.set_title(r"$t(D_{Td}>0.1)$ on $(\beta,e)$")
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label=r"$t(D>0.1)$")
    fig.tight_layout()
    fig.savefig(out / "tbreak_scatter.png", dpi=140)
    plt.close(fig)

    ranked = sorted(
        rows,
        key=lambda r: r["t_D_0.1"] if math.isfinite(r["t_D_0.1"]) else -1.0,
        reverse=True,
    )
    (out / "summary.json").write_text(
        json.dumps(
            {"m": args.m, "n": len(rows), "skipped": skipped, "rows": rows, "top": ranked[:10]},
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"# Td (β, e) scan — m={args.m:g}",
        "",
        f"n={len(rows)}, skipped={skipped}, t_end={args.t_end}",
        "",
        "| β | e | α° | t(D>1e-3) | t(D>0.1) | Dmax | Bmax | ρ_ptp |",
        "|---|---|----|-----------|----------|------|------|-------|",
    ]
    for r in ranked:
        def _t(x: float) -> str:
            return f"{x:.3g}" if math.isfinite(x) else "—"

        lines.append(
            f"| {r['beta']:.2f} | {r['e']:.2f} | {r['alpha_deg']:.0f} | "
            f"{_t(r['t_D_1e-3'])} | {_t(r['t_D_0.1'])} | "
            f"{r['D_Td_max']:.2e} | {r['B_max']:.2e} | {r['rho_ptp']:.3g} |"
        )
    lines += [
        "",
        "Plots: `beta_e_heatmaps.png`, `D_Td_curves.png`, `tbreak_scatter.png`.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done n={len(rows)} skipped={skipped} → {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
