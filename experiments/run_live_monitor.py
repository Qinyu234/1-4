#!/usr/bin/env python3
"""Live wall-clock monitor for perpetual N=4 search + Path A (Cursor terminal).

Every ``--interval-s`` seconds appends one JSON line with:
* choreography_search_n4/summary.json fields
* trials/hour since previous sample
* Path A summary (M_c_final, steps) if present
* optional short funnel timing sample (``--funnel-trials`` > 0)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _funnel_sample(n: int, trials: int, max_nfev: int) -> dict:
    from collections import defaultdict

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
    from fairy_orbit.store.search_db import trial_rng

    buckets: dict[str, float] = defaultdict(float)
    t0 = time.perf_counter()
    for trial in range(1, trials + 1):
        rng = trial_rng(n, trial + int(time.time()) % 100000)
        t = time.perf_counter()
        start = random_asymmetric_seed(n, rng)
        buckets["random_ic"] += time.perf_counter() - t
        t = time.perf_counter()
        polished, res_n = polish_seed(start, shift=1, max_nfev=max_nfev)
        buckets["polish"] += time.perf_counter() - t
        t = time.perf_counter()
        scout_then_certify(
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
    wall = time.perf_counter() - t0
    return {
        "trials": trials,
        "wall_s": wall,
        "s_per_trial": wall / max(trials, 1),
        "seconds": dict(buckets),
        "polish_frac": buckets["polish"] / max(wall, 1e-12),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Perpetual search + Path A live monitor")
    p.add_argument("--interval-s", type=float, default=600.0)
    p.add_argument("--funnel-trials", type=int, default=4, help="0 to skip timing sample")
    p.add_argument("--max-nfev", type=int, default=14)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments/output/profile/live_monitor.jsonl",
    )
    args = p.parse_args()

    search_sum = ROOT / "experiments/output/choreography_search_n4/summary.json"
    path_a_sum = ROOT / "experiments/output/continuation_n4/summary.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    prev_trials: int | None = None
    prev_t = time.time()
    print(
        f"live monitor → {args.out} interval={args.interval_s}s "
        f"funnel_trials={args.funnel_trials}",
        flush=True,
    )
    while True:
        now = time.time()
        s = _read_json(search_sum)
        pa = _read_json(path_a_sum)
        trials = int(s.get("trials") or 0)
        tph = None
        if prev_trials is not None and now > prev_t:
            dt_h = (now - prev_t) / 3600.0
            if dt_h > 0:
                tph = (trials - prev_trials) / dt_h
        row: dict = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "search": {
                "trials": trials,
                "passed_gate": s.get("passed_gate"),
                "n_scout": s.get("n_scout"),
                "n_certified": s.get("n_certified"),
                "best_trial_no": s.get("best_trial_no"),
                "best_residual": s.get("best_residual"),
                "away_prob": s.get("away_prob"),
                "family_hit_rate": (s.get("anneal") or {}).get("family_hit_rate"),
                "status": s.get("status"),
                "trials_per_hour_window": tph,
            },
            "path_a": {
                "M_c_final": pa.get("M_c_final"),
                "steps": pa.get("steps"),
                "optics_soft": pa.get("optics_soft"),
                "res_tol": pa.get("res_tol"),
            },
        }
        if args.funnel_trials > 0:
            try:
                row["funnel_sample"] = _funnel_sample(4, args.funnel_trials, args.max_nfev)
            except Exception as exc:  # noqa: BLE001
                row["funnel_sample"] = {"error": str(exc)}

        line = json.dumps(row, ensure_ascii=False)
        with args.out.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

        prev_trials = trials
        prev_t = now
        time.sleep(max(5.0, float(args.interval_s)))


if __name__ == "__main__":
    main()
