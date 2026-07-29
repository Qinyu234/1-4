#!/usr/bin/env python3
"""Launch four PROMPT campaigns in parallel (default wall 8h).

1. choreography search N=4
2. choreography search N=5
3. mass continuation N=4 (Mc↑)
4. mass continuation N=5 (μ↓)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Launch 4 parallel PROMPT campaigns")
    p.add_argument("--wall-hours", type=float, default=8.0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    py = sys.executable
    jobs = [
        (
            "choreo_n4",
            [
                py,
                str(ROOT / "experiments" / "run_choreography_search.py"),
                "--n",
                "4",
                "--wall-hours",
                str(args.wall_hours),
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
                str(args.wall_hours),
            ],
        ),
        (
            "cont_n4",
            [
                py,
                str(ROOT / "experiments" / "run_mass_continuation_campaign.py"),
                "--n",
                "4",
                "--wall-hours",
                str(args.wall_hours),
            ],
        ),
        (
            "cont_n5",
            [
                py,
                str(ROOT / "experiments" / "run_mass_continuation_campaign.py"),
                "--n",
                "5",
                "--wall-hours",
                str(args.wall_hours),
            ],
        ),
    ]
    log_dir = ROOT / "experiments" / "output" / "prompt_8h_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for name, cmd in jobs:
        log = log_dir / f"{name}.log"
        print(f"START {name}: {' '.join(cmd)} → {log}", flush=True)
        if args.dry_run:
            continue
        f = open(log, "w", encoding="utf-8")
        procs.append(
            (
                name,
                subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                ),
                f,
            )
        )
    if args.dry_run:
        return
    print(f"launched {len(procs)} jobs; wall={args.wall_hours}h", flush=True)
    print(f"logs: {log_dir}", flush=True)
    # Detach: do not wait — parent exits; children keep running if started with CREATE_NEW_PROCESS_GROUP on Windows
    # Actually if parent exits, children may keep running on Windows when not waiting.
    for name, proc, f in procs:
        print(f"  pid={proc.pid} {name}", flush=True)
        f.flush()
    # Keep handles open by waiting would block 8h — instead close after start and let OS keep children
    # On Windows, child survives if we don't kill. Close log files from parent carefully:
    # leave processes running; parent exits 0 immediately for agent background launch.
    print(
        "Parent exiting; campaigns continue. Monitor trials.jsonl / steps.jsonl under experiments/output/.",
        flush=True,
    )


if __name__ == "__main__":
    main()
