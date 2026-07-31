#!/usr/bin/env python3
"""Floquet resweep along Path-A continuation checkpoints.

Reads ``state_Mc_*.json`` from a continuation output dir and writes
``max|λ|`` vs ``M_c`` for bifurcation hunting.

Default path: ``experiments/output/continuation_n4`` (run Path A first).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fairy_orbit.observe.campaign_prefs import FLOQUET_STABLE_ATOL
from fairy_orbit.observe.floquet_sweep import floquet_path_sweep

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Floquet resweep along continuation path")
    p.add_argument(
        "--cont-dir",
        type=Path,
        default=ROOT / "experiments/output/continuation_n4",
        help="directory with state_Mc_*.json from Path A",
    )
    p.add_argument("--stable-atol", type=float, default=FLOQUET_STABLE_ATOL)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: <cont-dir>/floquet_path_sweep.json",
    )
    args = p.parse_args()

    def on_row(path: Path, Mc: float, seed) -> None:
        print(f"floquet {path.name} M_c={Mc} period={seed.period:.4f} …", flush=True)

    try:
        payload = floquet_path_sweep(
            args.cont_dir,
            stable_atol=args.stable_atol,
            out=args.out,
            on_row=on_row,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc} — run Path A first "
            "(experiments/run_mass_continuation_campaign.py --seed …)"
        ) from exc
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    for row in payload["rows"]:
        print(
            f"  M_c={row['M_c']:.6g} stable={row['stable']} "
            f"max_abs={row['max_abs']:.4f} n_unstable={row['n_unstable']}",
            flush=True,
        )
    print(
        f"wrote {payload['out']} stable={payload['n_stable']}/{payload['n']} "
        f"crossings={len(payload['unit_circle_crossings'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
