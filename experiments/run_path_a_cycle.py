#!/usr/bin/env python3
"""Perpetual Path A: cycle accepted N=4 seeds until killed.

When one seed folds, move to the next unused pass (Floquet-stable first).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from fairy_orbit.design.seeds import OrbitSeed, save_seed
from fairy_orbit.observe.campaign_prefs import CERTIFY_MAX_RESIDUAL, _floquet_meta
from fairy_orbit.store.search_db import ChoreographySearchStore, DEFAULT_SEARCH_DB_NAME

ROOT = Path(__file__).resolve().parents[1]


def _load_candidates(db: Path) -> list[tuple[int, OrbitSeed, bool | None, float | None]]:
    rows: list[tuple[int, OrbitSeed, bool | None, float | None]] = []
    with ChoreographySearchStore(db) as store:
        passes = store.list_passes(4, limit=200, max_residual=CERTIFY_MAX_RESIDUAL)
    for rec in passes:
        if not rec.seed_json:
            continue
        seed = OrbitSeed.from_dict(rec.seed_json)
        st, mx = _floquet_meta(seed)
        rows.append((int(rec.trial_no), seed, st, mx))
    # Floquet-stable first, then lower residual order already from list_passes
    rows.sort(key=lambda r: (0 if r[2] is True else 1, r[0]))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed perpetual Path A")
    p.add_argument("--res-tol", type=float, default=1e-3)
    p.add_argument("--m-c-max", type=float, default=1.0)
    p.add_argument("--max-nfev", type=int, default=80)
    p.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "experiments/output/continuation_n4_cycle",
    )
    p.add_argument("--sleep-s", type=float, default=2.0)
    args = p.parse_args()

    py = sys.executable
    script = ROOT / "experiments/run_mass_continuation_campaign.py"
    db = ROOT / "experiments/output/choreography_search_n4" / DEFAULT_SEARCH_DB_NAME
    args.out_root.mkdir(parents=True, exist_ok=True)

    seen: set[int] = set()
    round_i = 0
    while True:
        cands = _load_candidates(db)
        unused = [c for c in cands if c[0] not in seen]
        if not unused:
            print("seed pool exhausted; clearing seen and reshuffling", flush=True)
            seen.clear()
            time.sleep(args.sleep_s)
            continue
        trial_no, seed, st, mx = unused[0]
        seen.add(trial_no)
        round_i += 1
        out = args.out_root / f"trial_{trial_no}"
        out.mkdir(parents=True, exist_ok=True)
        seed_path = out / "seed.json"
        save_seed(seed, seed_path)
        print(
            f"[cycle {round_i}] trial={trial_no} floquet_stable={st} "
            f"max_abs={mx} → {out}",
            flush=True,
        )
        cmd = [
            py,
            str(script),
            "--n",
            "4",
            "--seed",
            str(seed_path),
            "--wall-hours",
            "0",
            "--no-optics-soft",
            "--res-tol",
            str(args.res_tol),
            "--m-c-max",
            str(args.m_c_max),
            "--max-nfev",
            str(args.max_nfev),
            "--out",
            str(out),
            "--floquet-sweep",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT))
        summary: dict = {}
        sp = out / "summary.json"
        if sp.is_file():
            try:
                summary = json.loads(sp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {}
        print(
            f"[cycle {round_i}] exit={proc.returncode} "
            f"M_c_final={summary.get('M_c_final')} steps={summary.get('steps')}",
            flush=True,
        )
        time.sleep(args.sleep_s)


if __name__ == "__main__":
    main()
