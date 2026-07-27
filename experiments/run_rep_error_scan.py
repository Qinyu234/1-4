#!/usr/bin/env python3
"""
Representation-error scan: all 8 channels together vs (m, e [, β, ρ]).

Channels: E_r, E_v, E_a, E_e, E_i, E_Omega, E_omega, E_M
(+ E_energy recorded, not in the 8).

Td equal-ρ IC (Stage A). Writes series, REPORT, and sigmas.json.
Whenever scan knobs change, re-run this script.
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
from fairy_orbit.observe.rep_error import (
    CHANNELS,
    compute_sigmas,
    rep_error_series,
)

ROOT = Path(__file__).resolve().parents[1]


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def run_one(
    m: float,
    beta: float,
    e: float,
    *,
    rho: float,
    t_end: float,
    n_out: int,
    epsilon: float,
    min_dt: float,
) -> dict:
    vr, vt, alpha = polar_from_beta_e(m, beta, e, rho=rho)
    omega = omega_from_vt(rho, vt)
    system, _ = build_td_system(
        m, rho, vr, omega, central_radius=0.0, fairy_radius=0.0
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
    series = rep_error_series(traj, mode="identity")
    tag = f"m{m:.0e}_b{beta:.2f}_e{e:.2f}_r{rho:.2f}".replace("+", "")
    finals = series.final_snapshot()
    row = {
        "tag": tag,
        "m": m,
        "beta": beta,
        "e": e,
        "rho": rho,
        "alpha_deg": math.degrees(alpha),
        "status": traj.status,
        "elapsed_s": elapsed,
        "t_end": float(series.times[-1]),
        "E_energy_final": float(series.E_energy[-1]),
        **{f"{k}_final": finals[k] for k in CHANNELS},
        **{f"{k}_max": float(np.nanmax(series.channels[k])) for k in CHANNELS},
        "t": series.times,
        "channels": series.channels,
        "E_energy": series.E_energy,
    }
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="8-channel representation error scan")
    p.add_argument("--m", default="1e-6,1e-4,1e-3,1e-2")
    p.add_argument("--beta", default="0.9,1.0,1.15")
    p.add_argument("--e", default="0.0,0.05,0.3,0.6")
    p.add_argument("--rho", default="1.0")
    p.add_argument("--t-end", type=float, default=8.0)
    p.add_argument("--n-outputs", type=int, default=200)
    p.add_argument("--epsilon", type=float, default=1e-9)
    p.add_argument("--min-dt", type=float, default=1e-5)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "rep_error",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    ms = _parse_floats(args.m)
    betas = _parse_floats(args.beta)
    es = _parse_floats(args.e)
    rhos = _parse_floats(args.rho)
    if args.smoke:
        ms = [1e-3]
        betas = [1.0]
        es = [0.0, 0.05]
        rhos = [1.0]
        args.t_end = 3.0
        args.n_outputs = 80

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
        f"rep-error scan m×β×e×ρ={ms}×{betas}×{es}×{rhos} t_end={args.t_end}",
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
                for rho in rhos:
                    row = run_one(
                        m,
                        beta,
                        e,
                        rho=rho,
                        t_end=args.t_end,
                        n_out=args.n_outputs,
                        epsilon=args.epsilon,
                        min_dt=args.min_dt,
                    )
                    ch = row.pop("channels")
                    t = row.pop("t")
                    Een = row.pop("E_energy")
                    np.savez(
                        series_dir / f"{row['tag']}.npz",
                        t=t,
                        E_energy=Een,
                        **{k: ch[k] for k in CHANNELS},
                    )
                    rows.append(row)
                    with jsonl.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    print(
                        f"  m={m:.0e} β={beta:.2f} e={e:.2f} ρ={rho:.2f} "
                        f"Er={row['E_r_final']:.3e} Ev={row['E_v_final']:.3e} "
                        f"Ea={row['E_a_final']:.3e} "
                        f"t/run={row['elapsed_s']:.2f}s",
                        flush=True,
                    )

    # σ from final snapshots
    finals = [{k: r[f"{k}_final"] for k in CHANNELS} for r in rows]
    sigmas = compute_sigmas(finals, source="rep_error_scan_finals")
    sigmas.to_json(out / "sigmas.json")
    print(f"sigmas → {out / 'sigmas.json'} n={sigmas.n_samples}", flush=True)

    # curves: E_r, E_v, E_a vs t colored by e
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    cmap = plt.cm.cividis
    e_norm = plt.Normalize(min(es) if es else 0.0, max(es) if es else 1.0)
    for r in rows:
        data = np.load(series_dir / f"{r['tag']}.npz")
        t = data["t"]
        c = cmap(e_norm(r["e"]))
        axes[0, 0].semilogy(t, np.maximum(data["E_r"], 1e-30), color=c, lw=1.0, alpha=0.85)
        axes[0, 1].semilogy(t, np.maximum(data["E_v"], 1e-30), color=c, lw=1.0, alpha=0.85)
        axes[1, 0].semilogy(t, np.maximum(data["E_a"], 1e-30), color=c, lw=1.0, alpha=0.85)
        axes[1, 1].semilogy(t, np.maximum(data["E_e"], 1e-30), color=c, lw=1.0, alpha=0.85)
    axes[0, 0].set_title(r"$E_r$")
    axes[0, 1].set_title(r"$E_v$")
    axes[1, 0].set_title(r"$E_a$")
    axes[1, 1].set_title(r"$E_e$")
    for ax in axes.ravel():
        ax.set_xlabel("t")
        ax.grid(True, which="both", alpha=0.3)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=e_norm, cmap=cmap),
        ax=axes,
        label=r"$e$",
        fraction=0.03,
    )
    fig.suptitle("Representation errors (8-channel scan)")
    fig.tight_layout()
    fig.savefig(out / "rep_error_curves.png", dpi=140)
    plt.close(fig)

    slim = list(rows)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "n": len(rows),
                "skipped": skipped,
                "t_end": args.t_end,
                "sigmas": sigmas.as_dict(),
                "rows": slim,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Representation error scan — 8 channels",
        "",
        "Td IC, P=id, shared R*. Channels: "
        + ", ".join(CHANNELS)
        + ". E_energy recorded separately.",
        "",
        f"n={len(rows)}, skipped={skipped}, t_end={args.t_end}",
        "",
        "## σ (from finals)",
        "",
        "```json",
        json.dumps(sigmas.as_dict(), indent=2),
        "```",
        "",
        "| m | β | e | ρ | Er | Ev | Ea | Ee | EM |",
        "|---|---|---|---|----|----|----|----|----|",
    ]
    ranked = sorted(rows, key=lambda r: r["E_r_final"])
    for r in ranked:
        lines.append(
            f"| {r['m']:.0e} | {r['beta']:.2f} | {r['e']:.2f} | {r['rho']:.2f} | "
            f"{r['E_r_final']:.3e} | {r['E_v_final']:.3e} | "
            f"{r['E_a_final']:.3e} | {r['E_e_final']:.3e} | "
            f"{r['E_M_final']:.3e} |"
        )
    lines += ["", "Plots: `rep_error_curves.png`. Sigmas: `sigmas.json`."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done n={len(rows)} → {out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
