#!/usr/bin/env python3
"""Long-running mass continuation (PROMPT Path A / B-style)."""

from __future__ import annotations

import argparse
from pathlib import Path

from fairy_orbit.design.seeds import SEEDS_DIR, load_seed
from fairy_orbit.observe.continuation import run_path_a_continuation, run_path_b_mass_scan

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Mass continuation campaign")
    p.add_argument("--n", type=int, required=True, choices=[4, 5])
    p.add_argument("--wall-hours", type=float, default=8.0)
    p.add_argument("--shift", type=int, default=1)
    p.add_argument("--max-nfev", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.n == 4:
        seed = load_seed(SEEDS_DIR / "free_4_square_re.json")
        out = args.out or (ROOT / "experiments" / "output" / "continuation_n4")
        print(f"Path A Mc↑ n=4 wall={args.wall_hours}h → {out}", flush=True)
        summary = run_path_a_continuation(
            seed,
            wall_hours=args.wall_hours,
            shift=args.shift,
            max_nfev=args.max_nfev,
            out_dir=out,
        )
    else:
        seed = load_seed(SEEDS_DIR / "free_5_pentagon_re.json")
        out = args.out or (ROOT / "experiments" / "output" / "continuation_n5")
        print(f"Path B μ↓ n=5 wall={args.wall_hours}h → {out}", flush=True)
        summary = run_path_b_mass_scan(
            seed,
            wall_hours=args.wall_hours,
            shift=args.shift,
            max_nfev=args.max_nfev,
            out_dir=out,
        )
    print("DONE", summary, flush=True)


if __name__ == "__main__":
    main()
