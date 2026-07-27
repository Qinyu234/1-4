#!/usr/bin/env python3
"""
PEO smoke: manifold → REBOUND → 8-channel rep errors.
COM-frame IC; free (a1,e1,M1,vx,vy,vz).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.design.manifold import ManifoldParams
from fairy_orbit.observe.peo import evaluate_peo
from fairy_orbit.observe.rep_error import CHANNELS, RepSigmas

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="PEO 8-channel smoke")
    p.add_argument("--t-end", type=float, default=20.0)
    p.add_argument("--n-outputs", type=int, default=120)
    p.add_argument("--a0", type=float, default=1.0)
    p.add_argument("--a1", type=float, default=0.15)
    p.add_argument("--e0", type=float, default=0.05)
    p.add_argument("--e1", type=float, default=0.0)
    p.add_argument("--M0", type=float, default=0.0)
    p.add_argument("--M1", type=float, default=0.4)
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--vz", type=float, default=0.0)
    p.add_argument("--mu", type=float, default=1e-3)
    p.add_argument("--sigmas", type=Path, default=None)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "peo_smoke",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.t_end = 8.0
        args.n_outputs = 80

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    params = ManifoldParams(
        a0=args.a0,
        a1=args.a1,
        e0=args.e0,
        e1=args.e1,
        M0=args.M0,
        M1=args.M1,
        vx=args.vx,
        vy=args.vy,
        vz=args.vz,
        mu_mass=args.mu,
    )
    sigmas = RepSigmas.from_json(args.sigmas) if args.sigmas else None
    result = evaluate_peo(
        params, t_end=args.t_end, n_outputs=args.n_outputs, sigmas=sigmas
    )
    (out / "summary.json").write_text(
        json.dumps(result.summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(result.summary, indent=2))

    if result.closure_rep is not None:
        s = result.closure_rep
        np.savez(out / "rep_error.npz", t=s.times, E_energy=s.E_energy, **s.channels)
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in ("E_r", "E_v", "E_a", "E_e"):
            ax.semilogy(s.times, np.maximum(s.channels[k], 1e-30), label=k)
        ax.set_xlabel("t")
        ax.set_ylabel("rep error")
        ax.set_title(f"PEO 8-channel — perm={s.perm}")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "rep_errors.png", dpi=140)
        plt.close(fig)

    lines = [
        "# PEO smoke — 8-channel representation errors",
        "",
        f"status={result.status}",
        f"θ={list(params.as_theta())}",
        f"channels={list(CHANNELS)}",
        "",
        "```json",
        json.dumps(result.summary, indent=2),
        "```",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done → {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
