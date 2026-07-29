#!/usr/bin/env python3
"""
Td (regular tetrahedron) E_r / E_v scan over (m, β, e).

P fixed as identity (equal ρ). At each t:
  R* = argmin_R Σ ||r_i(t) − R r_i(0)||²
  E_r, E_v with the same R* (PROMPT Level 2–3).
Collision off. ρ₀=1.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.design.tetra_eff import (
    e_min_for_beta,
    is_valid_beta_e,
    omega_from_vt,
    polar_from_beta_e,
    build_td_system,
)
from fairy_orbit.engine import ReboundConfig, integrate
from fairy_orbit.observe.closure import closure_series

ROOT = Path(__file__).resolve().parents[1]
RHO = 1.0


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
    series = closure_series(traj, mode="identity")

    def _cross(y: np.ndarray, level: float) -> float:
        idx = np.where(y > level)[0]
        return float(series.times[idx[0]]) if len(idx) else float("nan")

    tag = f"m{m:.0e}_b{beta:.2f}_e{e:.2f}".replace("+", "")
    return {
        "tag": tag,
        "m": m,
        "beta": beta,
        "e": e,
        "alpha_deg": math.degrees(alpha),
        "status": traj.status,
        "elapsed_s": elapsed,
        "t_end": float(series.times[-1]),
        "E_r_0": float(series.E_r[0]),
        "E_v_0": float(series.E_v[0]),
        "E_r_final": float(series.E_r[-1]),
        "E_v_final": float(series.E_v[-1]),
        "E_r_max": float(np.nanmax(series.E_r)),
        "E_v_max": float(np.nanmax(series.E_v)),
        "t_Er_1e-3": _cross(series.E_r, 1e-3),
        "t_Er_0.1": _cross(series.E_r, 0.1),
        "t_Ev_1e-3": _cross(series.E_v, 1e-3),
        "t_Ev_0.1": _cross(series.E_v, 0.1),
        "t": series.times,
        "E_r": series.E_r,
        "E_v": series.E_v,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Td E_r/E_v scan vs (m,β,e)")
    p.add_argument("--m", default="1e-6,1e-4,1e-3,1e-2")
    p.add_argument("--beta", default="0.7,0.9,1.0,1.15")
    p.add_argument("--e", default="0.0,0.3,0.6,0.8")
    p.add_argument("--t-end", type=float, default=8.0)
    p.add_argument("--n-outputs", type=int, default=250)
    p.add_argument("--epsilon", type=float, default=1e-9)
    p.add_argument("--min-dt", type=float, default=1e-5)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "td_closure_scan",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    ms = _parse_floats(args.m)
    betas = _parse_floats(args.beta)
    es = _parse_floats(args.e)
    if args.smoke:
        ms = [1e-3]
        betas = [1.0, 1.15]
        es = [0.0, 0.4]
        args.t_end = 4.0
        args.n_outputs = 100

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    series_dir = out / "series"
    series_dir.mkdir(exist_ok=True)
    jsonl = out / "runs.jsonl"
    if jsonl.exists():
        jsonl.unlink()

    rows: list[dict] = []
    skipped = 0
    print(
        f"Td closure scan m×β×e={ms}×{betas}×{es} t_end={args.t_end}",
        flush=True,
    )
    for m in ms:
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
                    m,
                    beta,
                    e,
                    t_end=args.t_end,
                    n_out=args.n_outputs,
                    epsilon=args.epsilon,
                    min_dt=args.min_dt,
                )
                np.savez(
                    series_dir / f"{row['tag']}.npz",
                    t=row.pop("t"),
                    E_r=row.pop("E_r"),
                    E_v=row.pop("E_v"),
                )
                rows.append(row)
                with jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(
                    f"  m={m:.0e} β={beta:.2f} e={e:.2f} "
                    f"Er_max={row['E_r_max']:.3e} Ev_max={row['E_v_max']:.3e} "
                    f"t(Er>0.1)={row['t_Er_0.1']} "
                    f"t/run={row['elapsed_s']:.2f}s",
                    flush=True,
                )

    # --- curves: E_r(t), E_v(t) colored by m / β / e ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    cmap_m = plt.cm.viridis
    cmap_b = plt.cm.plasma
    cmap_e = plt.cm.cividis
    m_norm = plt.Normalize(math.log10(min(ms)), math.log10(max(ms)))
    b_norm = plt.Normalize(min(betas), max(betas))
    e_norm = plt.Normalize(min(es), max(es))
    for r in rows:
        data = np.load(series_dir / f"{r['tag']}.npz")
        t, Er, Ev = data["t"], data["E_r"], data["E_v"]
        axes[0, 0].semilogy(
            t, np.maximum(Er, 1e-30), color=cmap_m(m_norm(math.log10(r["m"]))), lw=1.0, alpha=0.85
        )
        axes[1, 0].semilogy(
            t, np.maximum(Ev, 1e-30), color=cmap_m(m_norm(math.log10(r["m"]))), lw=1.0, alpha=0.85
        )
        axes[0, 1].semilogy(
            t, np.maximum(Er, 1e-30), color=cmap_b(b_norm(r["beta"])), lw=1.0, alpha=0.85
        )
        axes[1, 1].semilogy(
            t, np.maximum(Ev, 1e-30), color=cmap_b(b_norm(r["beta"])), lw=1.0, alpha=0.85
        )
        axes[0, 2].semilogy(
            t, np.maximum(Er, 1e-30), color=cmap_e(e_norm(r["e"])), lw=1.0, alpha=0.85
        )
        axes[1, 2].semilogy(
            t, np.maximum(Ev, 1e-30), color=cmap_e(e_norm(r["e"])), lw=1.0, alpha=0.85
        )
    axes[0, 0].set_title(r"$E_r(t)$ by $m$")
    axes[0, 1].set_title(r"$E_r(t)$ by $\beta$")
    axes[0, 2].set_title(r"$E_r(t)$ by $e$")
    axes[1, 0].set_title(r"$E_v(t)$ by $m$")
    axes[1, 1].set_title(r"$E_v(t)$ by $\beta$")
    axes[1, 2].set_title(r"$E_v(t)$ by $e$")
    for ax in axes.ravel():
        ax.set_xlabel("t")
        ax.grid(True, which="both", alpha=0.3)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=m_norm, cmap=cmap_m), ax=axes[:, 0], label=r"$\log_{10} m$", fraction=0.05
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=b_norm, cmap=cmap_b), ax=axes[:, 1], label=r"$\beta$", fraction=0.05
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=e_norm, cmap=cmap_e), ax=axes[:, 2], label=r"$e$", fraction=0.05
    )
    fig.suptitle(r"Td $E_r$, $E_v$ (fixed $P=\mathrm{id}$, same $R^*$)")
    fig.tight_layout()
    fig.savefig(out / "Er_Ev_curves.png", dpi=140)
    plt.close(fig)

    # --- scatter / heat proxies on parameter planes ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    beta_a = np.array([r["beta"] for r in rows])
    e_a = np.array([r["e"] for r in rows])
    m_a = np.array([r["m"] for r in rows])
    Er_max = np.array([r["E_r_max"] for r in rows])
    Ev_max = np.array([r["E_v_max"] for r in rows])
    t_br = np.array(
        [r["t_Er_0.1"] if math.isfinite(r["t_Er_0.1"]) else args.t_end for r in rows]
    )

    sc = axes[0, 0].scatter(beta_a, e_a, c=np.log10(np.maximum(Er_max, 1e-30)), cmap="magma", s=40)
    axes[0, 0].set_xlabel(r"$\beta$")
    axes[0, 0].set_ylabel(r"$e$")
    axes[0, 0].set_title(r"$\log_{10} E_r^{\max}$")
    fig.colorbar(sc, ax=axes[0, 0])
    bb = np.linspace(min(betas) - 0.05, max(betas) + 0.05, 80)
    axes[0, 0].plot(bb, [e_min_for_beta(b) for b in bb], "c--", lw=1, label=r"$e_{\min}$")
    axes[0, 0].legend(fontsize=8)

    sc = axes[0, 1].scatter(beta_a, e_a, c=np.log10(np.maximum(Ev_max, 1e-30)), cmap="viridis", s=40)
    axes[0, 1].set_xlabel(r"$\beta$")
    axes[0, 1].set_ylabel(r"$e$")
    axes[0, 1].set_title(r"$\log_{10} E_v^{\max}$")
    fig.colorbar(sc, ax=axes[0, 1])

    sc = axes[1, 0].scatter(
        np.log10(m_a), beta_a, c=np.log10(np.maximum(Er_max, 1e-30)), cmap="magma", s=40
    )
    axes[1, 0].set_xlabel(r"$\log_{10} m$")
    axes[1, 0].set_ylabel(r"$\beta$")
    axes[1, 0].set_title(r"$\log_{10} E_r^{\max}$ vs $(m,\beta)$")
    fig.colorbar(sc, ax=axes[1, 0])

    sc = axes[1, 1].scatter(beta_a, e_a, c=t_br, cmap="coolwarm", s=40)
    axes[1, 1].set_xlabel(r"$\beta$")
    axes[1, 1].set_ylabel(r"$e$")
    axes[1, 1].set_title(r"$t(E_r>0.1)$")
    fig.colorbar(sc, ax=axes[1, 1], label="t")
    fig.suptitle(r"Td parameter maps — $E_r$ / $E_v$")
    fig.tight_layout()
    fig.savefig(out / "param_maps.png", dpi=140)
    plt.close(fig)

    ranked = sorted(rows, key=lambda r: r["E_r_max"])
    slim = [{k: v for k, v in r.items() if k not in ("t", "E_r", "E_v")} for r in rows]
    (out / "summary.json").write_text(
        json.dumps(
            {"n": len(rows), "skipped": skipped, "t_end": args.t_end, "rows": slim, "lowest_Er": ranked[:10]},
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Td closure scan — E_r / E_v vs (m, β, e)",
        "",
        "Regular tetrahedron IC, ρ=1, no collision. "
        "P=id fixed; R* from positions; E_v uses same R*.",
        "",
        f"n={len(rows)}, skipped={skipped}, t_end={args.t_end}",
        "",
        "| m | β | e | Er_max | Ev_max | t(Er>0.1) | t(Ev>0.1) |",
        "|---|---|---|--------|--------|-----------|-----------|",
    ]
    for r in ranked:
        def _t(x):
            return f"{x:.3g}" if math.isfinite(x) else "—"

        lines.append(
            f"| {r['m']:.0e} | {r['beta']:.2f} | {r['e']:.2f} | "
            f"{r['E_r_max']:.3e} | {r['E_v_max']:.3e} | "
            f"{_t(r['t_Er_0.1'])} | {_t(r['t_Ev_0.1'])} |"
        )
    lines += ["", "Plots: `Er_Ev_curves.png`, `param_maps.png`."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done n={len(rows)} → {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
