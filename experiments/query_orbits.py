#!/usr/bin/env python3
"""Query / list orbit results stored in SQLite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairy_orbit.store import DEFAULT_DB, OrbitStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Fairy Orbit SQL store")
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / DEFAULT_DB,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("classes", help="list param_class aggregates")
    p_list.add_argument("--limit", type=int, default=40)

    p_q = sub.add_parser("query", help="filter runs by initial params / metrics")
    p_q.add_argument("--class", dest="param_class", default=None)
    p_q.add_argument("--e-min", type=float, default=None)
    p_q.add_argument("--e-max", type=float, default=None)
    p_q.add_argument("--mu-min", type=float, default=None)
    p_q.add_argument("--mu-max", type=float, default=None)
    p_q.add_argument("--tet", type=int, choices=[0, 1], default=None)
    p_q.add_argument("--min-interest", type=float, default=None)
    p_q.add_argument("--swap", action="store_true", help="require a-order swap")
    p_q.add_argument("--order", default="interest", choices=["interest", "a_delta_rms", "id"])
    p_q.add_argument("--limit", type=int, default=20)
    p_q.add_argument("--json", action="store_true")

    p_get = sub.add_parser("get", help="fetch one run (+ optional traj shape)")
    p_get.add_argument("run_id", type=int)
    p_get.add_argument("--traj", action="store_true")

    args = parser.parse_args()
    with OrbitStore(args.db) as store:
        if args.cmd == "classes":
            rows = store.list_classes()[: args.limit]
            print(f"db={args.db} n_runs={store.count()} n_classes={len(rows)}")
            for r in rows:
                print(
                    f"{r['param_class']}\tn={r['n']}\tmaxI={r['max_interest']}\t"
                    f"meanA={r['mean_a_rms']}\tswap={r['n_swap']}"
                )
            return

        if args.cmd == "query":
            recs = store.query(
                param_class=args.param_class,
                e_min=args.e_min,
                e_max=args.e_max,
                mu_min=args.mu_min,
                mu_max=args.mu_max,
                tetrahedral=None if args.tet is None else bool(args.tet),
                min_interest=args.min_interest,
                a_order_changed=True if args.swap else None,
                order_by=args.order,
                limit=args.limit,
            )
            if args.json:
                print(
                    json.dumps(
                        [
                            {
                                "id": r.id,
                                "param_class": r.param_class,
                                "e": r.eccentricity,
                                "mu": r.mass_ratio,
                                "interest": r.interest,
                                "a_delta_rms": r.a_delta_rms,
                                "swap": r.a_order_changed,
                                "status": r.status,
                                "traj_path": r.traj_path,
                            }
                            for r in recs
                        ],
                        indent=2,
                    )
                )
                return
            print(f"matches={len(recs)}")
            for r in recs:
                print(
                    f"id={r.id}\t{r.param_class}\tI={r.interest}\t"
                    f"a_rms={r.a_delta_rms}\tswap={r.a_order_changed}\t"
                    f"enc={r.n_encounters}"
                )
            return

        if args.cmd == "get":
            rec = store.get(args.run_id)
            if rec is None:
                raise SystemExit(f"run {args.run_id} not found")
            payload = {
                "id": rec.id,
                "param_class": rec.param_class,
                "eccentricity": rec.eccentricity,
                "mass_ratio": rec.mass_ratio,
                "period_ratios": rec.period_ratios,
                "tetrahedral": rec.tetrahedral,
                "summary": rec.summary,
                "traj_path": rec.traj_path,
            }
            if args.traj:
                traj = store.load_trajectory(args.run_id)
                payload["trajectory_shape"] = None if traj is None else list(traj.positions.shape)
            print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
