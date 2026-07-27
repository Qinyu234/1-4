#!/usr/bin/env python3
"""
Long campaign: Stage-A rep-error σ refresh, then beam search until wall budget.

Example (1h30m):
  python experiments/run_long_campaign.py --wall-min 90
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from fairy_orbit.observe.rep_error import RepSigmas
from fairy_orbit.observe.search import (
    BeamConfig,
    SearchBounds,
    grid_beam_search,
    result_to_dict,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "output" / "long_campaign"
SIGMAS = ROOT / "experiments" / "output" / "rep_error" / "sigmas.json"
PY = sys.executable

# Seeds: e small enough that e0 + 3 e1 stays < 1 with default bounds
DEFAULT_SEEDS = [
    (1e-6, 0.0),
    (1e-6, 0.03),
    (1e-6, 0.05),
    (1e-4, 0.0),
    (1e-4, 0.03),
    (1e-4, 0.05),
    (1e-3, 0.0),
    (1e-3, 0.03),
    (1e-3, 0.05),
    (1e-2, 0.0),
    (1e-2, 0.03),
    (1e-2, 0.05),
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_rep_error(wall_left: float) -> None:
    """Refresh σ; skip if almost no time."""
    if wall_left < 120.0:
        _log(f"skip rep_error (only {wall_left:.0f}s left)")
        return
    cmd = [
        PY,
        str(ROOT / "experiments" / "run_rep_error_scan.py"),
        "--m",
        "1e-6,1e-4,1e-3,1e-2",
        "--beta",
        "0.9,1.0,1.15",
        "--e",
        "0.0,0.05,0.3,0.6",
        "--rho",
        "1.0",
        "--t-end",
        "6.0",
        "--n-outputs",
        "160",
        "--out",
        str(ROOT / "experiments" / "output" / "rep_error"),
    ]
    _log("=== Stage A: rep_error_scan ===")
    _log(" ".join(cmd))
    subprocess.run(cmd, check=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Long PEO campaign with wall budget")
    p.add_argument("--wall-min", type=float, default=90.0)
    p.add_argument("--t-end", type=float, default=6.0)
    p.add_argument("--n-outputs", type=int, default=100)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--coarse", type=int, default=2)
    p.add_argument("--max-evals-per-seed", type=int, default=4000)
    p.add_argument("--skip-rep-error", action="store_true")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    deadline = time.perf_counter() + args.wall_min * 60.0
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "seeds").mkdir(exist_ok=True)
    status_path = out / "STATUS.md"

    def write_status(line: str) -> None:
        elapsed = args.wall_min * 60.0 - max(0.0, deadline - time.perf_counter())
        left = max(0.0, deadline - time.perf_counter())
        text = (
            f"# Long campaign status\n\n"
            f"- wall budget: {args.wall_min:.1f} min\n"
            f"- elapsed ≈ {elapsed/60:.1f} min, left ≈ {left/60:.1f} min\n"
            f"- last: {line}\n"
        )
        status_path.write_text(text, encoding="utf-8")
        _log(line)

    write_status("started")

    if not args.skip_rep_error:
        run_rep_error(deadline - time.perf_counter())
        write_status("rep_error_scan finished")

    sigmas = (
        RepSigmas.from_json(SIGMAS)
        if SIGMAS.exists()
        else RepSigmas(source="unit_fallback")
    )
    _log(f"sigmas source={sigmas.source} n={sigmas.n_samples}")

    cfg = BeamConfig(
        beam_width=args.beam,
        coarse_points=args.coarse,
        refine_points=(5, 7, 9),
        n_periods=2.0,
        t_end=None,
        n_outputs=args.n_outputs,
        max_evals=args.max_evals_per_seed,
        bisect_iters=8,
        grad_steps=10,
    )
    bounds = SearchBounds()
    results = []
    seed_i = 0
    rounds = 0

    while time.perf_counter() < deadline:
        m, e = DEFAULT_SEEDS[seed_i % len(DEFAULT_SEEDS)]
        seed_i += 1
        left = deadline - time.perf_counter()
        if left < 30.0:
            write_status("stopping (<30s left)")
            break
        # shrink max_evals if little time left (~0.1 s/eval)
        est_cap = max(64, int(left / 0.12))
        local_cfg = BeamConfig(
            beam_width=cfg.beam_width,
            coarse_points=cfg.coarse_points,
            refine_points=cfg.refine_points if left > 600 else (5,),
            n_periods=cfg.n_periods,
            t_end=None,
            n_outputs=cfg.n_outputs,
            max_evals=min(cfg.max_evals, est_cap),
            bisect_iters=cfg.bisect_iters if left > 300 else 4,
            grad_steps=cfg.grad_steps if left > 300 else 4,
            epsilon=cfg.epsilon,
            min_dt=cfg.min_dt,
        )
        write_status(
            f"beam seed m={m:.0e} e={e:.3f} max_evals={local_cfg.max_evals} left={left/60:.1f}min"
        )
        res = grid_beam_search(m, e, bounds=bounds, config=local_cfg, sigmas=sigmas)
        payload = result_to_dict(res)
        results.append(payload)
        tag = f"r{rounds:03d}_m{m:.0e}_e{e:.2f}".replace("+", "")
        (out / "seeds" / f"{tag}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        best = payload.get("best")
        _log(
            f"  done n_evals={res.n_evals} wall={res.wall_s:.1f}s "
            f"best={None if best is None else best['loss']}"
        )
        rounds += 1
        # persist rolling summary
        ranked = sorted(
            [r for r in results if r.get("best")],
            key=lambda r: r["best"]["loss"],
        )
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "sigmas_source": sigmas.source,
                    "n_rounds": rounds,
                    "results": results,
                    "top": ranked[:10],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    ranked = sorted(
        [r for r in results if r.get("best")],
        key=lambda r: r["best"]["loss"],
    )
    lines = [
        "# Long campaign REPORT",
        "",
        f"wall_min={args.wall_min}, rounds={rounds}, sigmas={sigmas.source}",
        "",
        "| rank | m | e | loss | n_evals | free |",
        "|------|---|---|------|---------|------|",
    ]
    for i, r in enumerate(ranked[:20], 1):
        b = r["best"]
        lines.append(
            f"| {i} | {r['seed']['m']:.0e} | {r['seed']['e']:.3f} | "
            f"{b['loss']:.6g} | {r['n_evals']} | `{json.dumps(b['params'])}` |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_status(f"finished rounds={rounds} → REPORT.md")
    _log(f"done → {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
