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
    p.add_argument(
        "--atol-rel",
        type=float,
        default=1e-8,
        help="§3.2 relative residual gate (default 1e-8)",
    )
    p.add_argument(
        "--max-residual",
        type=float,
        default=1e-6,
        help="max polish residual to count as pass (default 1e-6)",
    )
    p.add_argument(
        "--write-pass-json",
        action="store_true",
        help="also write pass_*.json (default: SQLite + best.json only)",
    )
    p.add_argument(
        "--import-json",
        action="store_true",
        help="force scan pass_*.json into SQLite even if DB already has rows",
    )
    p.add_argument(
        "--keep-pass-json",
        action="store_true",
        help="do not move existing pass_*.json into pass_json_archive/",
    )
    args = p.parse_args()
    out = args.out or (ROOT / "experiments" / "output" / f"choreography_search_n{args.n}")
    wall = None if args.wall_hours <= 0 else args.wall_hours
    print(
        f"choreography search n={args.n} wall={'unlimited' if wall is None else f'{wall}h'} "
        f"fresh={args.fresh} atol_rel={args.atol_rel:g} max_residual={args.max_residual:g} → {out}",
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
        atol_rel=args.atol_rel,
        max_residual=args.max_residual,
        write_pass_json=args.write_pass_json,
        import_json=args.import_json,
        archive_json=not args.keep_pass_json,
        on_progress=prog,
    )
    print("DONE", summary, flush=True)


if __name__ == "__main__":
    main()
