#!/usr/bin/env python3
"""Timed hotspot breakdown for one choreography-search trial loop (no py-spy needed)."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from fairy_orbit.observe.choreography_search import (
    polish_seed,
    random_asymmetric_seed,
)
from fairy_orbit.observe.choreography_verify import (
    accept_free_choreography,
    maintains_regular_equal_ngon,
    verify_choreography_Tn,
)
from fairy_orbit.store.search_db import seed_fingerprint, trial_rng

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Time search hotspots for N trials")
    p.add_argument("--n", type=int, default=4, choices=[4, 5])
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--max-nfev", type=int, default=14)
    p.add_argument("--atol-rel", type=float, default=1e-8)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "profile" / "hotspots.json",
    )
    args = p.parse_args()

    buckets: dict[str, float] = defaultdict(float)
    n_ok = 0
    t0 = time.perf_counter()
    for trial in range(1, args.trials + 1):
        rng = trial_rng(args.n, trial)
        t = time.perf_counter()
        start = random_asymmetric_seed(args.n, rng)
        buckets["random_ic"] += time.perf_counter() - t

        t = time.perf_counter()
        _ = seed_fingerprint(start)
        buckets["fingerprint"] += time.perf_counter() - t

        t = time.perf_counter()
        polished, res_n = polish_seed(start, shift=1, max_nfev=args.max_nfev)
        buckets["polish_least_squares"] += time.perf_counter() - t

        t = time.perf_counter()
        gate = verify_choreography_Tn(
            polished.to_system(),
            polished.period,
            shift=1,
            atol_rel=args.atol_rel,
            n_outputs=24,
        )
        buckets["verify_Tn_integrate"] += time.perf_counter() - t

        t = time.perf_counter()
        maintained = maintains_regular_equal_ngon(
            polished.to_system(),
            polished.period,
            n_samples=12,
        )
        buckets["maintains_ngon_integrate"] += time.perf_counter() - t

        t = time.perf_counter()
        acc = accept_free_choreography(
            polished.to_system(),
            polished.period,
            shift=1,
            atol_rel=args.atol_rel,
            n_outputs=24,
            ngon_samples=12,
        )
        buckets["accept_full_again"] += time.perf_counter() - t

        if acc.ok and res_n <= 1e-6:
            n_ok += 1
        if trial % 5 == 0 or trial == 1:
            elapsed = time.perf_counter() - t0
            print(
                f"trial={trial}/{args.trials} elapsed={elapsed:.1f}s "
                f"res={res_n:.3e} gate={gate.ok} maint={maintained}",
                flush=True,
            )

    total = time.perf_counter() - t0
    # accept_full_again double-counts verify+maintains; report exclusive-ish
    exclusive = {
        "random_ic": buckets["random_ic"],
        "fingerprint": buckets["fingerprint"],
        "polish_least_squares": buckets["polish_least_squares"],
        "verify_Tn_integrate": buckets["verify_Tn_integrate"],
        "maintains_ngon_integrate": buckets["maintains_ngon_integrate"],
        # subtract overlapping accept if we want — keep as reference only
        "accept_full_again_overlap": buckets["accept_full_again"],
    }
    accounted = (
        exclusive["random_ic"]
        + exclusive["fingerprint"]
        + exclusive["polish_least_squares"]
        + exclusive["verify_Tn_integrate"]
        + exclusive["maintains_ngon_integrate"]
    )
    report = {
        "n": args.n,
        "trials": args.trials,
        "wall_s": total,
        "s_per_trial": total / args.trials,
        "passed_strict": n_ok,
        "seconds": exclusive,
        "fraction_of_accounted": {
            k: (v / accounted if accounted else 0.0) for k, v in exclusive.items() if k != "accept_full_again_overlap"
        },
        "notes": (
            "polish_least_squares includes many short REBOUND integrates inside LM residuals; "
            "verify_Tn and maintains_ngon are additional full-period/T/n integrates outside polish."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
