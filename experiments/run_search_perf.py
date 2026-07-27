#!/usr/bin/env python3
"""Performance probe for PEO eval + beam-search stages."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fairy_orbit.design.manifold import from_error_seed
from fairy_orbit.observe.peo import evaluate_peo
from fairy_orbit.observe.rep_error import RepSigmas
from fairy_orbit.observe.search import BeamConfig, SearchBounds, grid_beam_search

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Search / eval performance")
    p.add_argument("--m", type=float, default=1e-3)
    p.add_argument("--e", type=float, default=0.05)
    p.add_argument("--t-end", type=float, default=3.0)
    p.add_argument("--n-outputs", type=int, default=60)
    p.add_argument("--n-eval", type=int, default=5)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "search_perf",
    )
    args = p.parse_args()
    if args.smoke:
        args.n_eval = 2
        args.t_end = 1.5
        args.n_outputs = 30

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sigmas = RepSigmas(source="unit_perf")

    # single eval timing
    params = from_error_seed(args.m, args.e, a1=0.15, e1=0.0, M1=0.5, vx=0.0, vy=0.0, vz=0.0)
    times = []
    for _ in range(args.n_eval):
        t0 = time.perf_counter()
        evaluate_peo(params, t_end=args.t_end, n_outputs=args.n_outputs, sigmas=sigmas)
        times.append(time.perf_counter() - t0)
    eval_mean = sum(times) / len(times)

    # tiny beam
    cfg = BeamConfig(
        beam_width=2,
        coarse_points=2,
        refine_points=(3,),
        t_end=args.t_end,
        n_outputs=args.n_outputs,
        max_evals=80 if args.smoke else 200,
        bisect_iters=2,
        grad_steps=2,
    )
    t0 = time.perf_counter()
    res = grid_beam_search(args.m, args.e, bounds=SearchBounds(), config=cfg, sigmas=sigmas)
    beam_wall = time.perf_counter() - t0

    report = {
        "eval_s_mean": eval_mean,
        "eval_s_all": times,
        "evals_per_hour": 3600.0 / eval_mean if eval_mean > 0 else None,
        "beam_n_evals": res.n_evals,
        "beam_wall_s": beam_wall,
        "beam_s_per_eval": beam_wall / max(res.n_evals, 1),
        "best_loss": None if res.best is None else res.best.loss,
        "t_end": args.t_end,
        "n_outputs": args.n_outputs,
    }
    (out / "perf.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Search performance",
        "",
        f"- single PEO eval mean: **{eval_mean:.3f} s** ({args.n_eval} trials)",
        f"- ≈ {report['evals_per_hour']:.1f} evals/hour",
        f"- beam: n_evals={res.n_evals}, wall={beam_wall:.2f}s, "
        f"{report['beam_s_per_eval']:.3f} s/eval",
        f"- best_loss={report['best_loss']}",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"done → {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
