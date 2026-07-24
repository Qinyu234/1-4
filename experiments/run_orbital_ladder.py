#!/usr/bin/env python3
"""PROMPT §5: construct an orbital ladder and observe emergent dynamics (no optimization)."""

from __future__ import annotations

import argparse
from pathlib import Path

from fairy_orbit.core import SystemConfig
from fairy_orbit.design import LadderParams, build_orbital_ladder
from fairy_orbit.observe import diagnose
from fairy_orbit.store import DEFAULT_DB, OrbitStore
from fairy_orbit.viz import save_ladder_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run orbital-ladder REBOUND verification")
    parser.add_argument("--e", type=float, default=0.15, help="shared eccentricity")
    parser.add_argument("--a-inner", type=float, default=1.0, help="innermost semi-major axis")
    parser.add_argument("--mass-ratio", type=float, default=1e-4)
    parser.add_argument("--t-end", type=float, default=200.0, help="integration time")
    parser.add_argument("--n-outputs", type=int, default=800)
    parser.add_argument("--no-megno", action="store_true")
    parser.add_argument("--no-tetra", action="store_true", help="coplanar legacy IC")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/output/ladder"),
        help="output directory",
    )
    parser.add_argument("--db", type=Path, default=ROOT / DEFAULT_DB)
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="short run for CI")
    args = parser.parse_args()

    if args.smoke:
        args.t_end = 20.0
        args.n_outputs = 100
        args.no_megno = True

    config = SystemConfig(G=1.0, central_mass=1.0, mass_ratio=args.mass_ratio)
    params = LadderParams(
        eccentricity=args.e,
        a_inner=args.a_inner,
        tetrahedral=not args.no_tetra,
    )
    system = build_orbital_ladder(config, params)

    diagnosis = diagnose(
        system,
        config,
        t_end=args.t_end,
        n_outputs=args.n_outputs,
        ladder=params,
        run_megno=not args.no_megno,
    )
    paths = save_ladder_report(diagnosis, args.out)

    run_id = None
    if not args.no_db:
        with OrbitStore(args.db) as store:
            run_id = store.save(
                config=config,
                params=params,
                diagnosis=diagnosis,
                source="ladder",
                t_end=args.t_end,
                n_outputs=args.n_outputs,
                store_trajectory=True,
            )

    print(f"status={diagnosis.summary['status']}")
    print(f"encounters={diagnosis.summary['n_encounters']}")
    print(f"megno={diagnosis.summary['megno']}")
    print(f"interest={diagnosis.summary.get('interest')}")
    print(f"energy_drift={diagnosis.summary['energy_drift']:.3e}")
    if run_id is not None:
        print(f"db_run_id={run_id} db={args.db}")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
