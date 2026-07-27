#!/usr/bin/env python3
"""Entry points for error / calibration experiments only."""

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
    parser = argparse.ArgumentParser(description="Fairy Orbit error-experiment launcher")
    parser.add_argument(
        "mode",
        choices=[
            "smoke",
            "calib",
            "td_group",
            "td_error",
            "td_beta_e",
            "td_growth",
            "td_dense",
            "peo",
        ],
        help="smoke|calib|td_group|td_error|td_beta_e|td_growth|td_dense|peo",
    )
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.mode == "smoke":
        code = run([_py(), str(ROOT / "experiments" / "run_calibration.py"), "--smoke"])
        if code == 0:
            code = run(
                [
                    _py(),
                    str(ROOT / "experiments" / "run_peo_smoke.py"),
                    "--smoke",
                ]
            )
        raise SystemExit(code)

    if args.mode == "calib":
        cmd = [_py(), str(ROOT / "experiments" / "run_calibration.py")]
        if args.t_end is not None:
            cmd += ["--t-end", str(args.t_end)]
        if args.epsilon is not None:
            cmd += ["--epsilon", str(args.epsilon)]
        if args.smoke:
            cmd.append("--smoke")
        raise SystemExit(run(cmd))

    scripts = {
        "td_group": "run_td_group_orbit.py",
        "td_error": "run_tetra_error_growth.py",
        "td_beta_e": "run_td_beta_e_scan.py",
        "td_growth": "fit_td_growth_law.py",
        "td_dense": "run_td_dense_growth.py",
        "peo": "run_peo_smoke.py",
    }
    cmd = [_py(), str(ROOT / "experiments" / scripts[args.mode])]
    if args.smoke and args.mode in {"td_group", "td_error", "td_beta_e", "peo"}:
        cmd.append("--smoke")
    if args.t_end is not None and args.mode in {"td_group", "td_error", "td_beta_e", "peo"}:
        cmd += ["--t-end", str(args.t_end)]
    raise SystemExit(run(cmd))


if __name__ == "__main__":
    main()
