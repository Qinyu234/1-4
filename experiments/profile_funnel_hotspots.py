#!/usr/bin/env python3
"""Timed breakdown of current scout→certify→Floquet funnel (N trials)."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from fairy_orbit.observe.campaign_prefs import (
    CERTIFY_ATOL_REL,
    CERTIFY_MAX_RESIDUAL,
    FLOQUET_STABLE_ATOL,
    SCOUT_ATOL_REL,
    SCOUT_MAX_RESIDUAL,
)
from fairy_orbit.observe.choreography_search import (
    polish_seed,
    random_asymmetric_seed,
    scout_then_certify,
)
from fairy_orbit.observe.stability import floquet_multipliers_fd
from fairy_orbit.store.search_db import trial_rng

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Profile scout/certify/Floquet funnel")
    p.add_argument("--n", type=int, default=4, choices=[4, 5])
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--max-nfev", type=int, default=14)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments/output/profile/funnel_hotspots.json",
    )
    args = p.parse_args()

    buckets: dict[str, float] = defaultdict(float)
    n_scout = n_cert = n_floq_ok = 0
    t0 = time.perf_counter()
    for trial in range(1, args.trials + 1):
        rng = trial_rng(args.n, trial + 900000)
        t = time.perf_counter()
        start = random_asymmetric_seed(args.n, rng)
        buckets["random_ic"] += time.perf_counter() - t

        t = time.perf_counter()
        polished, res_n = polish_seed(start, shift=1, max_nfev=args.max_nfev)
        buckets["polish"] += time.perf_counter() - t

        t = time.perf_counter()
        funnel = scout_then_certify(
            polished,
            residual=res_n,
            shift=1,
            scout_atol_rel=SCOUT_ATOL_REL,
            scout_max_residual=SCOUT_MAX_RESIDUAL,
            certify_atol_rel=CERTIFY_ATOL_REL,
            certify_max_residual=CERTIFY_MAX_RESIDUAL,
            require_floquet_stable=True,
            floquet_stable_atol=FLOQUET_STABLE_ATOL,
        )
        buckets["scout_then_certify"] += time.perf_counter() - t
        if funnel.get("scout_ok"):
            n_scout += 1
        if funnel.get("certified"):
            n_cert += 1
        fl = funnel.get("floquet") or {}
        if fl.get("stable"):
            n_floq_ok += 1

        if trial % 4 == 0 or trial == 1:
            print(
                f"trial={trial}/{args.trials} polish+funnel "
                f"res={res_n:.3e} scout={funnel.get('scout_ok')} "
                f"cert={funnel.get('certified')} reason={funnel.get('reason')}",
                flush=True,
            )

    # optional: cost of Floquet alone on last polished if any
    t = time.perf_counter()
    try:
        floquet_multipliers_fd(polished, shift=1, stable_atol=FLOQUET_STABLE_ATOL)
        buckets["floquet_once_ref"] = time.perf_counter() - t
    except Exception as exc:  # pragma: no cover
        buckets["floquet_once_ref"] = -1.0
        floquet_err = str(exc)
    else:
        floquet_err = None

    total = time.perf_counter() - t0
    accounted = buckets["random_ic"] + buckets["polish"] + buckets["scout_then_certify"]
    report = {
        "n": args.n,
        "trials": args.trials,
        "wall_s": total,
        "s_per_trial": total / max(args.trials, 1),
        "n_scout_ok": n_scout,
        "n_certified": n_cert,
        "n_floquet_stable": n_floq_ok,
        "seconds": dict(buckets),
        "fraction": {
            k: buckets[k] / accounted if accounted else 0.0
            for k in ("random_ic", "polish", "scout_then_certify")
        },
        "floquet_once_ref_error": floquet_err,
        "notes": (
            "scout_then_certify includes verify + ngon + Floquet on survivors; "
            "polish dominates when most trials never scout-pass."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
