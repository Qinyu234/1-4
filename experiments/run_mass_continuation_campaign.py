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

from fairy_orbit.design.seeds import load_seed, save_seed
from fairy_orbit.observe.campaign_prefs import (
    ALLOW_UNSTABLE_PATH_A_SEED,
    FLOQUET_STABLE_ATOL,
    PATH_A_AUTO_FLOQUET_SWEEP,
    campaign_priority_blurb,
)
from fairy_orbit.observe.choreography_verify import accept_seed_choreography
from fairy_orbit.observe.continuation import (
    DEFAULT_PATH_A_HORIZON_PERIODS,
    DEFAULT_PATH_A_MAX_NFEV,
    attach_central_mass,
    correct_at_mass,
    run_path_a_continuation,
    run_path_b_mass_scan,
    symmetry_residual_vector,
)
from fairy_orbit.observe.floquet_sweep import floquet_path_sweep
from fairy_orbit.observe.stability import floquet_multipliers_fd

ROOT = Path(__file__).resolve().parents[1]


def _correct_only(
    seed,
    *,
    M_c: float,
    out: Path,
    shift: int,
    max_nfev: int,
    optics_soft: bool,
    log_rho: float,
    horizon_periods: float,
    floquet: bool,
    floquet_stable_atol: float,
) -> dict:
    import json

    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    sys0 = attach_central_mass(seed, float(M_c))
    f0 = symmetry_residual_vector(
        sys0,
        seed,
        seed.period,
        shift=shift,
        optics_soft=optics_soft,
        log_rho=log_rho,
        horizon_periods=horizon_periods,
    )
    n_before = float(np.linalg.norm(f0))
    polished, n_after, ls_ok = correct_at_mass(
        seed,
        float(M_c),
        shift=shift,
        max_nfev=max_nfev,
        optics_soft=optics_soft,
        log_rho=log_rho,
        horizon_periods=horizon_periods,
    )
    tag = (
        int(horizon_periods)
        if abs(horizon_periods - round(horizon_periods)) < 1e-12
        else horizon_periods
    )
    state_path = out / f"state_horizon{tag}.json"
    save_seed(polished, state_path)

    fl = None
    if floquet:
        try:
            fl = floquet_multipliers_fd(
                polished, shift=shift, stable_atol=floquet_stable_atol
            ).to_dict()
        except Exception as exc:  # noqa: BLE001
            fl = {"error": str(exc)}

    summary = {
        "path": "A_correct_only",
        "n": seed.n_bodies,
        "M_c": float(M_c),
        "horizon_periods": float(horizon_periods),
        "optics_soft": bool(optics_soft),
        "log_rho": float(log_rho),
        "residual_before": n_before,
        "residual_after": float(n_after),
        "ls_success": bool(ls_ok),
        "out": str(state_path),
        "floquet": fl,
    }
    (out / f"state_horizon{tag}.report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


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
    p.add_argument(
        "--max-nfev",
        type=int,
        default=DEFAULT_PATH_A_MAX_NFEV,
        help=f"LM residual eval budget per Mc step (default {DEFAULT_PATH_A_MAX_NFEV})",
    )
    p.add_argument(
        "--res-tol",
        type=float,
        default=1e-4,
        help="accept Mc step when ||F|| < res_tol (residual-dominated; "
        "ignores least_squares success flag)",
    )
    p.add_argument(
        "--m-c-max",
        type=float,
        default=1.0,
        help="Path A stop mass (default 1.0)",
    )
    p.add_argument(
        "--m-c",
        type=float,
        default=None,
        help="with --correct-only: fixed central mass to re-polish",
    )
    p.add_argument(
        "--correct-only",
        action="store_true",
        help="skip Mc continuation; LM-correct seed once at --m-c "
        "using --horizon-periods residual",
    )
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
    p.add_argument(
        "--horizon-periods",
        type=float,
        default=DEFAULT_PATH_A_HORIZON_PERIODS,
        help="Path A residual after this many orbital periods "
        f"(default {DEFAULT_PATH_A_HORIZON_PERIODS}; 0 = legacy τ=T/n). "
        "Pass once per run; for 3P and 4P invoke the script twice.",
    )
    args = p.parse_args()

    if not (-1.0 <= float(args.log_rho) <= 1.0):
        raise SystemExit("--log-rho must be in [-1, 1]")
    if args.correct_only and args.m_c is None:
        raise SystemExit("--correct-only requires --m-c")
    if args.correct_only and args.n != 4:
        raise SystemExit("--correct-only is Path A (n=4) only")

    print(campaign_priority_blurb(), flush=True)
    seed = load_seed(args.seed)
    if not args.correct_only:
        acc = accept_seed_choreography(seed, shift=args.shift, atol_rel=1e-5)
        if not acc.ok:
            print(f"seed rejected: {acc.reason}", flush=True)
            raise SystemExit(2)

    # Diagnostic Floquet on the equal-mass start (does not block by default).
    if not args.correct_only:
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
                    print(
                        "refusing unstable seed (--require-floquet-stable-seed)",
                        flush=True,
                    )
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

    if args.correct_only:
        out = args.out or (
            ROOT / "experiments" / "output" / "best_orbit_plots" / "path_a_best_Mc"
        )
        print(
            f"Path A correct-only M_c={args.m_c} horizon={args.horizon_periods}P "
            f"seed={args.seed} → {out}",
            flush=True,
        )
        summary = _correct_only(
            seed,
            M_c=float(args.m_c),
            out=out,
            shift=args.shift,
            max_nfev=args.max_nfev,
            optics_soft=args.optics_soft,
            log_rho=args.log_rho,
            horizon_periods=float(args.horizon_periods),
            floquet=bool(args.floquet_sweep),
            floquet_stable_atol=args.floquet_stable_atol,
        )
        print("DONE", summary, flush=True)
        return

    wall = None if args.wall_hours <= 0 else args.wall_hours

    if args.n == 4:
        out = args.out or (ROOT / "experiments" / "output" / "continuation_n4")
        print(
            f"Path A Mc↑ n=4 wall={'unlimited' if wall is None else f'{wall}h'} "
            f"horizon={args.horizon_periods}P seed={args.seed} → {out}",
            flush=True,
        )
        summary = run_path_a_continuation(
            seed,
            wall_hours=wall,
            M_c_max=args.m_c_max,
            shift=args.shift,
            max_nfev=args.max_nfev,
            res_tol=args.res_tol,
            out_dir=out,
            optics_soft=args.optics_soft,
            log_rho=args.log_rho,
            horizon_periods=args.horizon_periods,
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
            res_tol=args.res_tol,
            out_dir=out,
        )
    print("DONE", summary, flush=True)


if __name__ == "__main__":
    main()
