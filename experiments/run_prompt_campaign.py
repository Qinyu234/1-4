#!/usr/bin/env python3
"""Launch PROMPT mainline jobs (self-expanding; optional wall clock limit).

Default: unlimited wall (``--wall-hours 0``). Pass a positive ``--wall-hours``
to stop after that many hours. Outputs append under ``experiments/output/``
(trials.jsonl / steps.jsonl grow; summary.json updates live).

Jobs:
  1. choreography search N=4
  2. choreography search N=5
  3. mass continuation N=4 (only if ``--seed-n4`` given)
  4. mass continuation N=5 (only if ``--seed-n5`` given)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(
        description="PROMPT self-expanding campaign launcher (unlimited by default)"
    )
    p.add_argument(
        "--wall-hours",
        type=float,
        default=0.0,
        help="optional wall clock limit in hours; <=0 means unlimited",
    )
    p.add_argument(
        "--seed-n4",
        type=Path,
        default=None,
        help="accepted free-4 choreography JSON for Path A continuation",
    )
    p.add_argument(
        "--seed-n5",
        type=Path,
        default=None,
        help="accepted free-5 choreography JSON for Path B continuation",
    )
    p.add_argument("--fresh", action="store_true", help="wipe prior search/continuation dirs first")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--wait", action="store_true", help="block until all jobs finish")
    args = p.parse_args()

    py = sys.executable
    wall = str(args.wall_hours)
    out_root = ROOT / "experiments" / "output"
    log_dir = out_root / "prompt_campaign_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        for name in (
            "choreography_search_n4",
            "choreography_search_n5",
            "continuation_n4",
            "continuation_n5",
        ):
            d = out_root / name
            if d.is_dir():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()

    jobs: list[tuple[str, list[str]]] = [
        (
            "choreo_n4",
            [
                py,
                str(ROOT / "experiments" / "run_choreography_search.py"),
                "--n",
                "4",
                "--wall-hours",
                wall,
            ],
        ),
        (
            "choreo_n5",
            [
                py,
                str(ROOT / "experiments" / "run_choreography_search.py"),
                "--n",
                "5",
                "--wall-hours",
                wall,
            ],
        ),
    ]
    if args.seed_n4 is not None:
        jobs.append(
            (
                "cont_n4",
                [
                    py,
                    str(ROOT / "experiments" / "run_mass_continuation_campaign.py"),
                    "--n",
                    "4",
                    "--seed",
                    str(args.seed_n4),
                    "--wall-hours",
                    wall,
                ],
            )
        )
    if args.seed_n5 is not None:
        jobs.append(
            (
                "cont_n5",
                [
                    py,
                    str(ROOT / "experiments" / "run_mass_continuation_campaign.py"),
                    "--n",
                    "5",
                    "--seed",
                    str(args.seed_n5),
                    "--wall-hours",
                    wall,
                ],
            )
        )

    procs: list[tuple[str, subprocess.Popen]] = []
    for name, cmd in jobs:
        log = log_dir / f"{name}.log"
        print(f"START {name}: {' '.join(cmd)} → {log}", flush=True)
        if args.dry_run:
            continue
        f = open(log, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((name, proc))
        print(f"  pid={proc.pid} {name}", flush=True)

    if args.dry_run:
        return

    wall_label = "unlimited" if args.wall_hours <= 0 else f"{args.wall_hours}h"
    print(f"launched {len(procs)} jobs; wall={wall_label}; logs={log_dir}", flush=True)
    if args.wait:
        for name, proc in procs:
            code = proc.wait()
            print(f"DONE {name} exit={code}", flush=True)
    else:
        print(
            "Parent exiting; children keep running. "
            "Monitor trials.jsonl / summary.json under experiments/output/.",
            flush=True,
        )


if __name__ == "__main__":
    main()
