#!/usr/bin/env python3
"""Entry points: ladder observe, dynamics scan, query store, perf."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _py() -> str:
    return sys.executable


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fairy Orbit experiment launcher")
    parser.add_argument(
        "mode",
        choices=["smoke", "ladder", "dynamics", "query", "classes", "perf", "report", "refine"],
        help="smoke|ladder|dynamics|query|classes|perf|report|refine",
    )
    parser.add_argument("--t-end", type=float, default=800.0)
    parser.add_argument("--megno", action="store_true")
    parser.add_argument("--e-min", type=float, default=None)
    parser.add_argument("--e-max", type=float, default=None)
    parser.add_argument("--mu-min", type=float, default=None)
    parser.add_argument("--mu-max", type=float, default=None)
    parser.add_argument("--min-interest", type=float, default=None)
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--ids", type=int, nargs="*", help="explicit run ids for refine")
    parser.add_argument("--epsilon", type=float, default=None, help="IAS15 accuracy for refine")
    args = parser.parse_args()

    if args.mode == "smoke":
        code = run([_py(), str(ROOT / "experiments" / "run_orbital_ladder.py"), "--smoke"])
        if code == 0:
            code = run(
                [_py(), str(ROOT / "experiments" / "run_dynamics_scan.py"), "--smoke"]
            )
        if code == 0:
            code = run(
                [_py(), str(ROOT / "experiments" / "query_orbits.py"), "classes"]
            )
        raise SystemExit(code)

    if args.mode == "ladder":
        raise SystemExit(
            run(
                [
                    _py(),
                    str(ROOT / "experiments" / "run_orbital_ladder.py"),
                    "--t-end",
                    str(args.t_end),
                    "--out",
                    str(ROOT / "experiments" / "output" / "ladder"),
                ]
            )
        )

    if args.mode == "dynamics":
        cmd = [
            _py(),
            str(ROOT / "experiments" / "run_dynamics_scan.py"),
            "--t-end",
            str(args.t_end),
        ]
        if args.megno:
            cmd.append("--megno")
        raise SystemExit(run(cmd))

    if args.mode == "classes":
        raise SystemExit(
            run([_py(), str(ROOT / "experiments" / "query_orbits.py"), "classes"])
        )

    if args.mode == "query":
        cmd = [_py(), str(ROOT / "experiments" / "query_orbits.py"), "query", "--limit", str(args.limit)]
        if args.e_min is not None:
            cmd += ["--e-min", str(args.e_min)]
        if args.e_max is not None:
            cmd += ["--e-max", str(args.e_max)]
        if args.mu_min is not None:
            cmd += ["--mu-min", str(args.mu_min)]
        if args.mu_max is not None:
            cmd += ["--mu-max", str(args.mu_max)]
        if args.min_interest is not None:
            cmd += ["--min-interest", str(args.min_interest)]
        if args.swap:
            cmd.append("--swap")
        raise SystemExit(run(cmd))

    if args.mode == "perf":
        raise SystemExit(
            run(
                [
                    _py(),
                    str(ROOT / "experiments" / "analyze_perf.py"),
                    "--campaign",
                    str(ROOT / "experiments" / "output" / "dynamics"),
                    "--out",
                    str(ROOT / "experiments" / "output" / "perf"),
                ]
            )
        )

    if args.mode == "report":
        raise SystemExit(
            run([_py(), str(ROOT / "experiments" / "report_scan.py")])
        )

    if args.mode == "refine":
        cmd = [_py(), str(ROOT / "experiments" / "refine_candidates.py")]
        if args.ids:
            cmd += ["--ids", *[str(i) for i in args.ids]]
        if args.min_interest is not None:
            cmd += ["--min-interest", str(args.min_interest)]
        if args.epsilon is not None:
            cmd += ["--epsilon", str(args.epsilon)]
        cmd += ["--t-end", str(args.t_end)]
        raise SystemExit(run(cmd))


if __name__ == "__main__":
    main()
