#!/usr/bin/env python3
"""Thin launcher: PROMPT mainline + legacy Td modes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "experiments" / "legacy"
ACTIVE = ROOT / "experiments"


def _py() -> str:
    return sys.executable


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fairy Orbit experiment launcher")
    parser.add_argument(
        "mode",
        choices=[
            "prompt",
            "choreo4",
            "choreo5",
            # legacy Td
            "calib",
            "td_group",
            "td_error",
            "td_beta_e",
            "td_growth",
            "td_dense",
        ],
        help="prompt|choreo4|choreo5 | legacy: calib|td_*",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--wall-hours",
        type=float,
        default=0.0,
        help="optional wall clock hours; <=0 unlimited (default)",
    )
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    if args.mode == "prompt":
        cmd = [_py(), str(ACTIVE / "run_prompt_campaign.py"), "--wall-hours", str(args.wall_hours)]
        if args.fresh:
            cmd.append("--fresh")
        if args.wait or args.smoke:
            cmd.append("--wait")
        if args.smoke and args.wall_hours <= 0:
            # smoke: short wall so it can finish under --wait
            cmd = [
                _py(),
                str(ACTIVE / "run_prompt_campaign.py"),
                "--wall-hours",
                "0.01",
                "--wait",
            ]
            if args.fresh:
                cmd.append("--fresh")
        raise SystemExit(run(cmd))

    if args.mode == "choreo4":
        cmd = [
            _py(),
            str(ACTIVE / "run_choreography_search.py"),
            "--n",
            "4",
            "--wall-hours",
            str(args.wall_hours),
        ]
        raise SystemExit(run(cmd))

    if args.mode == "choreo5":
        cmd = [
            _py(),
            str(ACTIVE / "run_choreography_search.py"),
            "--n",
            "5",
            "--wall-hours",
            str(args.wall_hours),
        ]
        raise SystemExit(run(cmd))

    legacy_map = {
        "calib": "run_calibration.py",
        "td_group": "run_td_group_orbit.py",
        "td_error": "run_tetra_error_growth.py",
        "td_beta_e": "run_td_beta_e_scan.py",
        "td_growth": "fit_td_growth_law.py",
        "td_dense": "run_td_dense_growth.py",
    }
    cmd = [_py(), str(LEGACY / legacy_map[args.mode])]
    if args.smoke and args.mode in {"calib", "td_group", "td_error", "td_beta_e"}:
        cmd.append("--smoke")
    raise SystemExit(run(cmd))


if __name__ == "__main__":
    main()
