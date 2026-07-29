#!/usr/bin/env python3
"""Query choreography search SQLite store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairy_orbit.store.search_db import ChoreographySearchStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Query choreography search.sqlite")
    p.add_argument("--n", type=int, required=True, choices=[4, 5])
    p.add_argument(
        "--db",
        type=Path,
        default=None,
        help="default: experiments/output/choreography_search_n{N}/search.sqlite",
    )
    p.add_argument("--passes", action="store_true", help="list accepted passes")
    args = p.parse_args()
    db = args.db or (
        ROOT / "experiments" / "output" / f"choreography_search_n{args.n}" / "search.sqlite"
    )
    if not db.exists():
        print(f"missing db: {db}")
        raise SystemExit(1)
    with ChoreographySearchStore(db) as store:
        print(json.dumps(store.summary_dict(args.n), indent=2))
        if args.passes:
            for rec in store.list_passes(args.n):
                print(
                    {
                        "trial_no": rec.trial_no,
                        "residual": rec.residual,
                        "period": rec.period,
                        "reason": rec.reason,
                        "result_fp": rec.result_fp,
                    }
                )


if __name__ == "__main__":
    main()
