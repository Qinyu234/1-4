#!/usr/bin/env python3
"""
Beam search campaign: for each (m,e) seed, grid+beam over
(a1,e1,M1,vx,vy,vz) with a0=1, M0=0, e0=e, μ=m; inertial COM IC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairy_orbit.observe.rep_error import RepSigmas
from fairy_orbit.observe.search import (
    BeamConfig,
    SearchBounds,
    grid_beam_search,
    result_to_dict,
)

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
    p = argparse.ArgumentParser(description="PEO grid+beam search campaign")
    p.add_argument("--seeds", default="1e-3,0.05;1e-4,0.05")
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--coarse", type=int, default=2)
    p.add_argument("--t-end", type=float, default=None, help="override; default n_periods*2π")
    p.add_argument("--n-periods", type=float, default=2.0)
    p.add_argument("--n-outputs", type=int, default=160)
    p.add_argument("--max-evals", type=int, default=3000)
    p.add_argument("--sigmas", type=Path, default=DEFAULT_SIGMAS)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "beam_search",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--plot", action="store_true", help="plot orbit figures for bests")
    args = p.parse_args()

    if args.smoke:
        args.seeds = "1e-3,0.05"
        args.coarse = 2
        args.beam = 2
        args.n_periods = 1.0
        args.n_outputs = 60
        args.max_evals = 200

    seeds = _parse_pairs(args.seeds)
    sigmas = RepSigmas.from_json(args.sigmas) if args.sigmas.exists() else RepSigmas(source="unit")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    cfg = BeamConfig(
        beam_width=args.beam,
        coarse_points=args.coarse,
        refine_points=(5,) if args.smoke else (5, 7),
        n_periods=args.n_periods,
        t_end=args.t_end,
        n_outputs=args.n_outputs,
        max_evals=args.max_evals,
        bisect_iters=4 if args.smoke else 6,
        grad_steps=4 if args.smoke else 8,
    )
    bounds = SearchBounds()
    all_results = []
    for m, e in seeds:
        print(f"=== seed m={m:.0e} e={e:.4f} free={list(cfg.__dataclass_fields__)} ===", flush=True)
        print(f"search free=(a1,e1,M1,vx,vy,vz) anchors a0=1 e0={e} M0=0 m={m}", flush=True)
        res = grid_beam_search(m, e, bounds=bounds, config=cfg, sigmas=sigmas)
        payload = result_to_dict(res)
        all_results.append(payload)
        tag = f"m{m:.0e}_e{e:.2f}".replace("+", "")
        (out / f"{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        best = payload.get("best")
        print(
            f"  n_evals={res.n_evals} wall={res.wall_s:.1f}s "
            f"best_loss={None if best is None else best['loss']}",
            flush=True,
        )

    (out / "summary.json").write_text(
        json.dumps({"sigmas_source": sigmas.source, "results": all_results}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Beam search campaign",
        "",
        "Anchors: a0=1, M0=0, e0=e, μ=m. Free: a1,e1,M1,vx,vy,vz.",
        "Poly: q_i = q0 + i q1. Td kick δv_i=R_i·(vx,vy,vz). Requires a1>0.",
        f"t_end = {cfg.resolve_t_end():.4f} ({cfg.n_periods} periods).",
        "",
        "| m | e | n_evals | wall_s | best_loss | best free |",
        "|---|---|---------|--------|-----------|-----------|",
    ]
    for r in all_results:
        b = r.get("best")
        free = "" if b is None else json.dumps(b["params"])
        loss = "" if b is None else f"{b['loss']:.6g}"
        lines.append(
            f"| {r['seed']['m']:.0e} | {r['seed']['e']:.3f} | {r['n_evals']} | "
            f"{r['wall_s']:.1f} | {loss} | `{free}` |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done → {out / 'REPORT.md'}", flush=True)

    if args.plot and all_results:
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "plot_campaign_orbits.py"),
                "--summary",
                str(out / "summary.json"),
                "--out",
                str(out / "orbits"),
                "--n-periods",
                str(cfg.n_periods),
                "--sigmas",
                str(args.sigmas),
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
