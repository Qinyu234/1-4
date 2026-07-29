#!/usr/bin/env python3
"""
PEO campaign orchestrator (heatmap → beam → plots).

Delegates to:
  experiments/run_me_heatmap.py   — progressive (m,e) heatmap
  experiments/run_long_campaign.py — beam until solve / plateau / wall
  experiments/plot_campaign_orbits.py

Example:
  python experiments/run_10h_campaign.py --wall-hours 10 --skip-rep-error
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT_DEFAULT = ROOT / "experiments" / "output" / "campaign_10h"
SIGMAS = ROOT / "experiments" / "output" / "rep_error" / "sigmas.json"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _run(cmd: list[str]) -> int:
    _log("+ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description="PEO orchestrator: heatmap + beam + plots")
    p.add_argument("--wall-hours", type=float, default=10.0)
    p.add_argument("--heatmap-frac", type=float, default=0.35)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--nm0", type=int, default=6)
    p.add_argument("--ne0", type=int, default=5)
    p.add_argument("--cell-evals", type=int, default=100)
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--coarse", type=int, default=3)
    p.add_argument("--max-evals-per-seed", type=int, default=5000)
    p.add_argument("--n-outputs", type=int, default=100)
    p.add_argument("--n-periods", type=float, default=2.0)
    p.add_argument("--plateau-rounds", type=int, default=12)
    p.add_argument("--target-loss", type=float, default=1.0)
    p.add_argument("--skip-rep-error", action="store_true")
    p.add_argument("--skip-heatmaps", action="store_true")
    p.add_argument("--skip-beam", action="store_true")
    p.add_argument("--skip-plot", action="store_true")
    p.add_argument("--resume-heatmap", action="store_true")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()

    t0 = time.perf_counter()
    deadline = t0 + args.wall_hours * 3600.0
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    heat_dir = out / "heatmaps"
    status = out / "STATUS.md"

    def write_status(line: str) -> None:
        left = max(0.0, deadline - time.perf_counter()) / 3600.0
        elapsed = (time.perf_counter() - t0) / 3600.0
        status.write_text(
            f"# PEO campaign status\n\n"
            f"- wall {args.wall_hours:.1f} h, elapsed ≈ {elapsed:.2f} h, left ≈ {left:.2f} h\n"
            f"- last: {line}\n",
            encoding="utf-8",
        )
        _log(line)

    write_status("started")

    if not args.skip_rep_error and (deadline - time.perf_counter()) > 180:
        write_status("rep_error_scan")
        _run(
            [
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
        )

    if not args.skip_heatmaps:
        heat_hours = max(0.05, args.heatmap_frac * args.wall_hours)
        write_status(f"heatmap progressive ({heat_hours:.2f} h)")
        cmd = [
            PY,
            str(ROOT / "experiments" / "run_me_heatmap.py"),
            "--levels",
            str(args.levels),
            "--nm0",
            str(args.nm0),
            "--ne0",
            str(args.ne0),
            "--cell-evals",
            str(args.cell_evals),
            "--n-outputs",
            str(args.n_outputs),
            "--n-periods",
            str(args.n_periods),
            "--wall-hours",
            str(heat_hours),
            "--checkpoint-every",
            "5",
            "--out",
            str(heat_dir),
        ]
        if args.resume_heatmap:
            cmd.append("--resume")
        _run(cmd)

    if not args.skip_beam:
        left_min = max(1.0, (deadline - time.perf_counter()) / 60.0)
        write_status(f"beam campaign ({left_min:.1f} min left)")
        _run(
            [
                PY,
                str(ROOT / "experiments" / "run_long_campaign.py"),
                "--wall-min",
                str(left_min),
                "--skip-rep-error",
                "--wide-bounds",
                "--beam",
                str(args.beam),
                "--coarse",
                str(args.coarse),
                "--max-evals-per-seed",
                str(args.max_evals_per_seed),
                "--n-outputs",
                str(args.n_outputs),
                "--plateau-rounds",
                str(args.plateau_rounds),
                "--target-loss",
                str(args.target_loss),
                "--out",
                str(out / "beam"),
            ]
        )

    if not args.skip_plot:
        summary = out / "beam" / "summary.json"
        if not summary.exists():
            summary = out / "beam" / "beam_summary.json"
        # long_campaign writes summary.json
        if summary.exists() or (out / "beam" / "summary.json").exists():
            write_status("plot orbits")
            sum_path = out / "beam" / "summary.json"
            _run(
                [
                    PY,
                    str(ROOT / "experiments" / "plot_campaign_orbits.py"),
                    "--summary",
                    str(sum_path),
                    "--out",
                    str(out / "orbits"),
                    "--n-periods",
                    str(args.n_periods),
                    "--sigmas",
                    str(SIGMAS),
                    "--top",
                    "6",
                ]
            )

    write_status("finished")
    _log(f"done → {out}")


if __name__ == "__main__":
    main()
