#!/usr/bin/env python3
"""Long-running free-N choreography search (PROMPT construct; SQLite resume)."""

from __future__ import annotations

import argparse
from pathlib import Path

from fairy_orbit.observe.choreography_search import run_choreography_search

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Choreography multi-start search (SQL-backed)")
    p.add_argument("--n", type=int, required=True, choices=[4, 5])
    p.add_argument(
        "--wall-hours",
        type=float,
        default=0.0,
        help="wall clock hours; <=0 means unlimited",
    )
    p.add_argument("--shift", type=int, default=1)
    p.add_argument("--max-nfev", type=int, default=14)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite path (default: <out>/search.sqlite)",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="clear SQLite trials for this N before starting",
    )
    args = p.parse_args()
    out = args.out or (ROOT / "experiments" / "output" / f"choreography_search_n{args.n}")
    wall = None if args.wall_hours <= 0 else args.wall_hours
    print(
        f"choreography search n={args.n} wall={'unlimited' if wall is None else f'{wall}h'} "
        f"fresh={args.fresh} → {out}",
        flush=True,
    )

    def prog(row: dict) -> None:
        if row.get("trial", 0) % 5 == 0 or row.get("ok_gate") or row.get(
            "maintains_regular_ngon"
        ):
            print(row, flush=True)

    summary = run_choreography_search(
        args.n,
        wall_hours=wall,
        shift=args.shift,
        max_nfev=args.max_nfev,
        out_dir=out,
        db_path=args.db,
        fresh=args.fresh,
        on_progress=prog,
    )
    print("DONE", summary, flush=True)


if __name__ == "__main__":
    main()
