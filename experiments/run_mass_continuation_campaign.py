#!/usr/bin/env python3
"""Long-running mass continuation (PROMPT Path A / B-style).

Requires an *accepted* choreography seed (not a maintained regular-ngon RE).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fairy_orbit.design.seeds import load_seed
from fairy_orbit.observe.choreography_verify import accept_seed_choreography
from fairy_orbit.observe.continuation import run_path_a_continuation, run_path_b_mass_scan

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Mass continuation campaign")
    p.add_argument("--n", type=int, required=True, choices=[4, 5])
    p.add_argument(
        "--seed",
        type=Path,
        required=True,
        help="accepted free choreography JSON (not regular n-gon)",
    )
    p.add_argument(
        "--wall-hours",
        type=float,
        default=0.0,
        help="<=0 means unlimited",
    )
    p.add_argument("--shift", type=int, default=1)
    p.add_argument("--max-nfev", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    seed = load_seed(args.seed)
    acc = accept_seed_choreography(seed, shift=args.shift, atol_rel=1e-5)
    if not acc.ok:
        print(f"seed rejected: {acc.reason}", flush=True)
        raise SystemExit(2)

    wall = None if args.wall_hours <= 0 else args.wall_hours

    if args.n == 4:
        out = args.out or (ROOT / "experiments" / "output" / "continuation_n4")
        print(
            f"Path A Mc↑ n=4 wall={'unlimited' if wall is None else f'{wall}h'} "
            f"seed={args.seed} → {out}",
            flush=True,
        )
        summary = run_path_a_continuation(
            seed,
            wall_hours=wall,
            shift=args.shift,
            max_nfev=args.max_nfev,
            out_dir=out,
        )
    else:
        out = args.out or (ROOT / "experiments" / "output" / "continuation_n5")
        print(
            f"Path B μ↓ n=5 wall={'unlimited' if wall is None else f'{wall}h'} "
            f"seed={args.seed} → {out}",
            flush=True,
        )
        summary = run_path_b_mass_scan(
            seed,
            wall_hours=wall,
            shift=args.shift,
            max_nfev=args.max_nfev,
            out_dir=out,
        )
    print("DONE", summary, flush=True)


if __name__ == "__main__":
    main()
