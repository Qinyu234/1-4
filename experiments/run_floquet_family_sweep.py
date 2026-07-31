#!/usr/bin/env python3
"""Floquet sweep over shape-diverse accepted N=4 seeds (archive proxy).

True M_c continuation history is not in experiments/output yet; until Path A/B
steps exist, this sweeps across certified equal-mass families and records
max|λ| vs period / residual / family key — useful for spotting stable islands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairy_orbit.observe.campaign_prefs import FLOQUET_STABLE_ATOL
from fairy_orbit.observe.family_class import family_classification_key
from fairy_orbit.observe.shape_families import select_diverse_families
from fairy_orbit.observe.stability import floquet_multipliers_fd
from fairy_orbit.store.search_db import DEFAULT_SEARCH_DB_NAME, ChoreographySearchStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Floquet sweep over diverse N=4 passes")
    p.add_argument("--n-families", type=int, default=6)
    p.add_argument("--stable-atol", type=float, default=FLOQUET_STABLE_ATOL)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments/output/profile/floquet_family_sweep_n4.json",
    )
    args = p.parse_args()

    db = ROOT / "experiments/output/choreography_search_n4" / DEFAULT_SEARCH_DB_NAME
    rows = []
    with ChoreographySearchStore(db) as store:
        picks = select_diverse_families(
            store, 4, n_families=args.n_families, max_residual=1e-6
        )
        for pck in picks:
            ver = pck.seed.verification or {}
            key = ver.get("family_key") or family_classification_key(
                pck.seed,
                perm_label=str(ver.get("perm_label") or "?"),
                action=ver.get("action_proxy"),
            )
            print(
                f"floquet fam{pck.family_id} trial={pck.record.trial_no} "
                f"period={pck.seed.period:.4f} …",
                flush=True,
            )
            fl = floquet_multipliers_fd(
                pck.seed, shift=1, stable_atol=args.stable_atol
            )
            abs_sorted = sorted(
                (m["abs"] for m in fl.to_dict()["multipliers"]), reverse=True
            )
            rows.append(
                {
                    "family_id": pck.family_id,
                    "trial_no": pck.record.trial_no,
                    "residual": pck.residual,
                    "period": float(pck.seed.period),
                    "family_key": key,
                    "min_dist_to_prev": pck.min_dist_to_prev,
                    "map_residual": fl.map_residual,
                    "max_abs": fl.max_abs,
                    "stable": fl.stable,
                    "n_unstable": fl.n_unstable,
                    "top_abs": abs_sorted[:6],
                    "n_near_unit_0.02": sum(
                        1 for a in abs_sorted if abs(a - 1.0) < 0.02
                    ),
                }
            )
            print(
                f"  stable={fl.stable} max_abs={fl.max_abs:.4f} "
                f"n_unstable={fl.n_unstable}",
                flush=True,
            )

    payload = {
        "note": (
            "No M_c continuation steps.jsonl found; this is a cross-family "
            "equal-mass archive sweep. Re-run against continuation checkpoints "
            "when Path A/B history exists."
        ),
        "stable_atol": args.stable_atol,
        "n": len(rows),
        "n_stable": sum(1 for r in rows if r["stable"]),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out} stable={payload['n_stable']}/{payload['n']}", flush=True)


if __name__ == "__main__":
    main()
