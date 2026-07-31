#!/usr/bin/env python3
"""Long-running free-N choreography search (PROMPT construct; SQLite resume)."""

from __future__ import annotations

import argparse
from pathlib import Path

from fairy_orbit.observe.campaign_prefs import (
    CERTIFY_ATOL_REL,
    CERTIFY_MAX_RESIDUAL,
    FLOQUET_GATE_DEFAULT,
    FLOQUET_STABLE_ATOL,
    SCOUT_ATOL_REL,
    SCOUT_MAX_RESIDUAL,
    campaign_priority_blurb,
)
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
        default=CERTIFY_ATOL_REL,
        help=f"tight certify §3.2 relative residual (default {CERTIFY_ATOL_REL:g})",
    )
    p.add_argument(
        "--max-residual",
        type=float,
        default=CERTIFY_MAX_RESIDUAL,
        help=f"tight certify max polish residual (default {CERTIFY_MAX_RESIDUAL:g})",
    )
    p.add_argument(
        "--scout-atol-rel",
        type=float,
        default=SCOUT_ATOL_REL,
        help=f"loose scout §3.2 relative residual (default {SCOUT_ATOL_REL:g})",
    )
    p.add_argument(
        "--scout-max-residual",
        type=float,
        default=SCOUT_MAX_RESIDUAL,
        help=f"loose scout max polish residual (default {SCOUT_MAX_RESIDUAL:g})",
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
    p.add_argument(
        "--away-family-frac",
        type=float,
        default=None,
        help="fixed away-start probability; default=None uses residual annealing",
    )
    p.add_argument(
        "--away-min-sep",
        type=float,
        default=0.12,
        help="min shape distance to nearest accepted family for away starts",
    )
    p.add_argument(
        "--away-tries",
        type=int,
        default=24,
        help="rejection samples per away start before best-effort fallback",
    )
    p.add_argument(
        "--anneal-window",
        type=int,
        default=40,
        help="rolling window for family-hit-rate annealing",
    )
    p.add_argument(
        "--anneal-warmup",
        type=int,
        default=24,
        help="scout accepts before family-hit-rate raises away-prob",
    )
    p.add_argument(
        "--no-floquet-gate",
        action="store_true",
        help="skip Floquet linear-stability certify gate (not recommended)",
    )
    p.add_argument(
        "--floquet-stable-atol",
        type=float,
        default=FLOQUET_STABLE_ATOL,
        help=f"allow |λ| <= 1+atol for Floquet gate (default {FLOQUET_STABLE_ATOL})",
    )
    args = p.parse_args()
    out = args.out or (ROOT / "experiments" / "output" / f"choreography_search_n{args.n}")
    wall = None if args.wall_hours <= 0 else args.wall_hours
    away_desc = (
        f"fixed:{args.away_family_frac:g}"
        if args.away_family_frac is not None
        else "family_hit"
    )
    use_floquet = FLOQUET_GATE_DEFAULT and not args.no_floquet_gate
    print(campaign_priority_blurb(), flush=True)
    print(
        f"choreography search n={args.n} wall={'unlimited' if wall is None else f'{wall}h'} "
        f"fresh={args.fresh} scout={args.scout_atol_rel:g}/{args.scout_max_residual:g} "
        f"certify={args.atol_rel:g}/{args.max_residual:g} floquet={use_floquet} "
        f"away={away_desc} → {out}",
        flush=True,
    )

    def prog(row: dict) -> None:
        if row.get("trial", 0) % 5 == 0 or row.get("ok_gate") or row.get(
            "maintains_regular_ngon"
        ) or row.get("scout_ok"):
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
        scout_atol_rel=args.scout_atol_rel,
        scout_max_residual=args.scout_max_residual,
        write_pass_json=args.write_pass_json,
        import_json=args.import_json,
        archive_json=not args.keep_pass_json,
        away_family_frac=args.away_family_frac,
        away_min_sep=args.away_min_sep,
        away_tries=args.away_tries,
        anneal_window=args.anneal_window,
        anneal_warmup=args.anneal_warmup,
        require_floquet_stable=use_floquet,
        floquet_stable_atol=args.floquet_stable_atol,
        on_progress=prog,
    )
    print("DONE", summary, flush=True)


if __name__ == "__main__":
    main()
