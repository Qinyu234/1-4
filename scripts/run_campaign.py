#!/usr/bin/env python3
"""Thin launcher: PEO mainline + legacy Td modes."""

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
            "staged",
            "heatmap",
            "beam",
            "campaign",
            "peo_smoke",
            "rep_error",
            # legacy Td
            "calib",
            "td_group",
            "td_error",
            "td_beta_e",
            "td_growth",
            "td_dense",
        ],
        help="staged|heatmap|beam|campaign|peo_smoke|rep_error | legacy: calib|td_*",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wall-hours", type=float, default=None)
    args = parser.parse_args()

    if args.mode == "staged":
        cmd = [_py(), str(ACTIVE / "run_staged_peo.py")]
        if args.smoke:
            cmd.append("--smoke")
        raise SystemExit(run(cmd))

    if args.mode == "heatmap":
        cmd = [_py(), str(ACTIVE / "run_me_heatmap.py"), "--levels", "3", "--nm0", "6", "--ne0", "5"]
        if args.resume:
            cmd.append("--resume")
        raise SystemExit(run(cmd))

    if args.mode == "beam":
        cmd = [_py(), str(ACTIVE / "run_long_campaign.py"), "--skip-rep-error", "--wide-bounds"]
        if args.smoke:
            cmd += ["--wall-min", "5", "--plateau-rounds", "3"]
        raise SystemExit(run(cmd))

    if args.mode == "campaign":
        cmd = [_py(), str(ACTIVE / "run_10h_campaign.py"), "--skip-rep-error"]
        if args.wall_hours is not None:
            cmd += ["--wall-hours", str(args.wall_hours)]
        if args.resume:
            cmd.append("--resume-heatmap")
        if args.smoke:
            cmd += ["--wall-hours", "0.05", "--levels", "1", "--nm0", "3", "--ne0", "3", "--cell-evals", "40"]
        raise SystemExit(run(cmd))

    if args.mode == "peo_smoke":
        cmd = [_py(), str(ACTIVE / "run_peo_smoke.py")]
        if args.smoke:
            cmd.append("--smoke")
        raise SystemExit(run(cmd))

    if args.mode == "rep_error":
        raise SystemExit(run([_py(), str(ACTIVE / "run_rep_error_scan.py")]))

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
