#!/usr/bin/env python3
"""Cheap PROMPT §2.4 Branch-2 existence probe (fix ABCD, sample tracer E).

Lowest campaign priority (RESPONSE §6). Default is multi-family ``--diverse``
so a bleak result is not mistaken for a global Branch-2 kill.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairy_orbit.design.seeds import OrbitSeed, build_free_polygon_seed, load_seed
from fairy_orbit.observe.branch2_probe import branch2_existence_probe
from fairy_orbit.observe.campaign_prefs import (
    BRANCH2_PROBE_DEFAULT_DIVERSE,
    BRANCH2_PROBE_DEFAULT_SAMPLES,
    campaign_priority_blurb,
)
from fairy_orbit.observe.family_class import family_classification_key
from fairy_orbit.observe.shape_families import select_diverse_families
from fairy_orbit.store.search_db import DEFAULT_SEARCH_DB_NAME, ChoreographySearchStore

ROOT = Path(__file__).resolve().parents[1]


def _load_diverse_seeds(n_families: int) -> list[tuple[str, OrbitSeed]]:
    db = ROOT / "experiments/output/choreography_search_n4" / DEFAULT_SEARCH_DB_NAME
    out: list[tuple[str, OrbitSeed]] = []
    with ChoreographySearchStore(db) as store:
        picks = select_diverse_families(
            store, 4, n_families=n_families, max_residual=1e-6
        )
        for p in picks:
            ver = p.seed.verification or {}
            key = ver.get("family_key") or family_classification_key(
                p.seed,
                perm_label=str(ver.get("perm_label") or "?"),
                action=ver.get("action_proxy"),
            )
            label = f"fam{p.family_id}_trial{p.record.trial_no}_{key.replace('|', '_')}"
            out.append((label, p.seed))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Branch-2 cheap existence probe (lowest priority; multi-family default)"
    )
    p.add_argument(
        "--seed",
        type=Path,
        default=None,
        help="single ABCD seed JSON (disables default multi-family probe)",
    )
    p.add_argument(
        "--diverse",
        type=int,
        default=None,
        help=(
            f"probe this many shape-diverse N=4 families "
            f"(default {BRANCH2_PROBE_DEFAULT_DIVERSE} unless --seed)"
        ),
    )
    p.add_argument(
        "--samples",
        type=int,
        default=BRANCH2_PROBE_DEFAULT_SAMPLES,
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments/output/profile/branch2_probes",
    )
    args = p.parse_args()
    print(campaign_priority_blurb(), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, OrbitSeed]] = []
    if args.seed is not None:
        jobs = [(args.seed.stem, load_seed(args.seed))]
    else:
        n_div = (
            BRANCH2_PROBE_DEFAULT_DIVERSE if args.diverse is None else int(args.diverse)
        )
        if n_div <= 0:
            path = ROOT / "experiments/output/choreography_search_n4/best.json"
            if path.exists():
                jobs = [(path.stem, load_seed(path))]
            else:
                jobs = [
                    (
                        "poly4_fallback",
                        build_free_polygon_seed(4, seed_id="poly4", family="free_4"),
                    )
                ]
        else:
            jobs = _load_diverse_seeds(n_div)
            if not jobs:
                raise SystemExit("no diverse families found in SQLite")

    summary = []
    for label, seed4 in jobs:
        print(f"=== probe {label} period={seed4.period:.4f} ===", flush=True)
        result = branch2_existence_probe(seed4, n_samples=args.samples, seed=0)
        out = args.out_dir / f"branch2_probe_{label}.json"
        payload = result.to_dict()
        payload["seed_label"] = label
        payload["seed_period"] = float(seed4.period)
        payload["seed_id"] = seed4.id
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"hopeful={result.hopeful} best={result.best_residual:.4f} "
            f"median={result.median_residual:.4f} → {out}",
            flush=True,
        )
        summary.append(
            {
                "label": label,
                "hopeful": result.hopeful,
                "best_residual": result.best_residual,
                "median_residual": result.median_residual,
                "frac_below_0.35": result.frac_below.get("0.35"),
                "out": str(out),
            }
        )

    summary_path = args.out_dir / "branch2_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    n_hope = sum(1 for s in summary if s["hopeful"])
    print(
        f"SUMMARY hopeful={n_hope}/{len(summary)} → {summary_path}",
        flush=True,
    )
    print(
        "NOTE: bleak on listed families = those ABCD (R,τ) look unfriendly; "
        "Branch-2 as a mathematical object is only deprioritized, not ruled out.",
        flush=True,
    )


if __name__ == "__main__":
    main()
