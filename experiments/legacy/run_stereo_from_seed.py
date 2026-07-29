#!/usr/bin/env python3
"""
Stereo smoke from Stage-A (m,e) seed.
Anchors a0=1,M0=0,e0=e,μ=m; optional a1,e1,M1,vx,vy,vz.
Prefer run_beam_search.py for grid+beam.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fairy_orbit.design.manifold import from_error_seed
from fairy_orbit.observe.peo import evaluate_peo
from fairy_orbit.observe.rep_error import CHANNELS, RepSigmas

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGMAS = ROOT / "experiments" / "output" / "rep_error" / "sigmas.json"


def _parse_pairs(s: str) -> list[tuple[float, float]]:
    pairs = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [float(x) for x in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(f"expected m,e got {chunk!r}")
        pairs.append((parts[0], parts[1]))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description="Stereo from error-scan seeds")
    p.add_argument("--seeds", default="1e-3,0.05;1e-4,0.05")
    p.add_argument("--a1", type=float, default=0.15)
    p.add_argument("--e1", type=float, default=0.0)
    p.add_argument("--M1", type=float, default=0.5)
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--vz", type=float, default=0.0)
    p.add_argument("--t-end", type=float, default=6.0)
    p.add_argument("--n-outputs", type=int, default=200)
    p.add_argument("--sigmas", type=Path, default=DEFAULT_SIGMAS)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "stereo_from_seed",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.seeds = "1e-3,0.05"
        args.t_end = 2.0
        args.n_outputs = 60

    seeds = _parse_pairs(args.seeds)
    sigmas = (
        RepSigmas.from_json(args.sigmas) if args.sigmas.exists() else RepSigmas(source="unit_fallback")
    )
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for m, e in seeds:
        try:
            params = from_error_seed(
                m, e, a1=args.a1, e1=args.e1, M1=args.M1, vx=args.vx, vy=args.vy, vz=args.vz
            )
        except ValueError as exc:
            print(f"  skip seed m={m} e={e}: {exc}", flush=True)
            results.append({"m": m, "e": e, "status": "bad_seed", "error": str(exc)})
            continue
        print(f"stereo seed m={m:.0e} e={e:.3f} θ={params.as_theta()}", flush=True)
        result = evaluate_peo(
            params, t_end=args.t_end, n_outputs=args.n_outputs, sigmas=sigmas
        )
        summary = dict(result.summary)
        summary["m_seed"] = m
        summary["e_seed"] = e
        results.append(summary)
        if result.closure_rep is not None:
            np.savez(
                out / f"m{m:.0e}_e{e:.2f}.npz".replace("+", ""),
                t=result.closure_rep.times,
                **result.closure_rep.channels,
                E_energy=result.closure_rep.E_energy,
            )

    (out / "summary.json").write_text(
        json.dumps({"sigmas": sigmas.as_dict(), "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"done → {out}", flush=True)


if __name__ == "__main__":
    main()
