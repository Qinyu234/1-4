#!/usr/bin/env python3
"""Long-running mass continuation (PROMPT Path A / B-style).

Requires an *accepted* choreography seed (not a maintained regular-ngon RE).

Policy (RESPONSE §7): Path A may start from Floquet-*unstable* equal-mass
seeds — after finishing, Floquet path resweep runs by default to hunt
``|λ|=1`` crossings along ``M_c``. Prefer that over harder endpoint shooting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fairy_orbit.design.seeds import load_seed
from fairy_orbit.observe.campaign_prefs import (
    ALLOW_UNSTABLE_PATH_A_SEED,
    FLOQUET_STABLE_ATOL,
    PATH_A_AUTO_FLOQUET_SWEEP,
    campaign_priority_blurb,
)
from fairy_orbit.observe.choreography_verify import accept_seed_choreography
from fairy_orbit.observe.continuation import run_path_a_continuation, run_path_b_mass_scan
from fairy_orbit.observe.floquet_sweep import floquet_path_sweep
from fairy_orbit.observe.stability import floquet_multipliers_fd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="Mass continuation campaign")
    p.add_argument("--n", type=int, required=True, choices=[4, 5])
    p.add_argument(
        "--seed",
        type=Path,
        required=True,
        help="accepted free choreography JSON (not regular n-gon)",
    )
    p.add_argument(
        "--wall-hours",
        type=float,
        default=0.0,
        help="<=0 means unlimited",
    )
    p.add_argument("--shift", type=int, default=1)
    p.add_argument("--max-nfev", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--floquet-sweep",
        action=argparse.BooleanOptionalAction,
        default=PATH_A_AUTO_FLOQUET_SWEEP,
        help="after Path A, Floquet-resweep state_Mc_*.json (default: on)",
    )
    p.add_argument(
        "--floquet-stable-atol",
        type=float,
        default=FLOQUET_STABLE_ATOL,
    )
    p.add_argument(
        "--require-floquet-stable-seed",
        action="store_true",
        help="refuse Path A if seed is Floquet-unstable (off by default: "
        "unstable seeds are allowed for crossing hunts)",
    )
    p.add_argument(
        "--log-rho",
        type=float,
        default=0.0,
        help="log10 density in [-1,1] for equal-density radii (default 0 → ρ=1)",
    )
    p.add_argument(
        "--optics-soft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Path A soft extras: gravity close-approach + perp optical deficit",
    )
    args = p.parse_args()

    if not (-1.0 <= float(args.log_rho) <= 1.0):
        raise SystemExit("--log-rho must be in [-1, 1]")

    print(campaign_priority_blurb(), flush=True)
    seed = load_seed(args.seed)
    acc = accept_seed_choreography(seed, shift=args.shift, atol_rel=1e-5)
    if not acc.ok:
        print(f"seed rejected: {acc.reason}", flush=True)
        raise SystemExit(2)

    # Diagnostic Floquet on the equal-mass start (does not block by default).
    try:
        fl0 = floquet_multipliers_fd(
            seed, shift=args.shift, stable_atol=args.floquet_stable_atol
        )
        print(
            f"seed Floquet: stable={fl0.stable} max_abs={fl0.max_abs:.4f} "
            f"n_unstable={fl0.n_unstable}",
            flush=True,
        )
        if not fl0.stable:
            if args.require_floquet_stable_seed or not ALLOW_UNSTABLE_PATH_A_SEED:
                print("refusing unstable seed (--require-floquet-stable-seed)", flush=True)
                raise SystemExit(3)
            print(
                "note: unstable equal-mass start is OK — Path A + Floquet "
                "resweep looks for |λ|=1 crossings along M_c",
                flush=True,
            )
    except SystemExit:
        raise
    except Exception as exc:
        print(f"seed Floquet diagnostic skipped: {exc}", flush=True)

    wall = None if args.wall_hours <= 0 else args.wall_hours

    if args.n == 4:
        out = args.out or (ROOT / "experiments" / "output" / "continuation_n4")
        print(
            f"Path A Mc↑ n=4 wall={'unlimited' if wall is None else f'{wall}h'} "
            f"seed={args.seed} → {out}",
            flush=True,
        )
        summary = run_path_a_continuation(
            seed,
            wall_hours=wall,
            shift=args.shift,
            max_nfev=args.max_nfev,
            out_dir=out,
            optics_soft=args.optics_soft,
            log_rho=args.log_rho,
        )
        if args.floquet_sweep:
            print("Floquet path resweep…", flush=True)
            try:
                sweep = floquet_path_sweep(
                    out,
                    stable_atol=args.floquet_stable_atol,
                    on_row=lambda path, Mc, s: print(
                        f"  floquet {path.name} M_c={Mc}", flush=True
                    ),
                )
                summary = dict(summary)
                summary["floquet_path_sweep"] = {
                    "out": sweep.get("out"),
                    "n": sweep["n"],
                    "n_stable": sweep["n_stable"],
                    "crossings": sweep["unit_circle_crossings"],
                }
                print(
                    f"floquet sweep: stable={sweep['n_stable']}/{sweep['n']} "
                    f"crossings={len(sweep['unit_circle_crossings'])} → {sweep['out']}",
                    flush=True,
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"floquet sweep skipped: {exc}", flush=True)
                summary = dict(summary)
                summary["floquet_path_sweep"] = {"skipped": str(exc)}
    else:
        out = args.out or (ROOT / "experiments" / "output" / "continuation_n5")
        print(
            f"Path B μ↓ n=5 wall={'unlimited' if wall is None else f'{wall}h'} "
            f"seed={args.seed} → {out}",
            flush=True,
        )
        summary = run_path_b_mass_scan(
            seed,
            wall_hours=wall,
            shift=args.shift,
            max_nfev=args.max_nfev,
            out_dir=out,
        )
    print("DONE", summary, flush=True)


if __name__ == "__main__":
    main()
