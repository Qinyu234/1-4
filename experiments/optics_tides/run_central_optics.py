#!/usr/bin/env python3
"""Central-observer optics: equal albedo; evenness = period variance of F_total.

* ``--evenness uneven``: Sun role rotates at brightness-swap points (higher Var).
* ``--evenness uniform``: Sun identity fixed (lower Var).
* All fairies share one ``--A``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairy_orbit.design.seeds import load_seed

from experiments.optics_tides.brightness_swap import run_brightness_swap_optics
from experiments.optics_tides.photometry import DEFAULT_ALBEDO, DEFAULT_L_SUN


def main() -> None:
    p = argparse.ArgumentParser(description="1 Sun + 3 Moons; period-variance evenness")
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--m-c", type=float, required=True)
    p.add_argument("--periods", type=float, default=2.0)
    p.add_argument("--n-outputs", type=int, default=480)
    p.add_argument("--r-frac", type=float, default=0.02)
    p.add_argument("--metric", choices=["angular", "perp"], default="angular")
    p.add_argument("--initial-sun", type=int, default=1)
    p.add_argument("--L-sun", type=float, default=DEFAULT_L_SUN)
    p.add_argument("--A", type=float, default=DEFAULT_ALBEDO, help="shared albedo for all fairies")
    p.add_argument(
        "--illumination",
        choices=["far", "near"],
        default="near",
        help="near=finite self-luminous Sun (role-rotate evenness); far=parallel rays",
    )
    p.add_argument(
        "--no-occultation",
        action="store_true",
        help="disable Planet/fairy line-of-sight occultation",
    )
    p.add_argument(
        "--phase-power",
        type=float,
        default=1.0,
        help="deprecated (ignored; flux from starry)",
    )
    p.add_argument(
        "--evenness",
        choices=["uneven", "uniform"],
        default="uneven",
        help="uneven=rotate Sun at swaps (high period Var); uniform=fixed Sun (low Var)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: optics_tides/output/brightness_swap_{evenness}.json",
    )
    args = p.parse_args()

    role_rotate = args.evenness == "uneven"
    out = args.out or (
        ROOT / "experiments/optics_tides/output" / f"brightness_swap_{args.evenness}.json"
    )

    seed = load_seed(args.seed)
    report = run_brightness_swap_optics(
        seed,
        float(args.m_c),
        periods=float(args.periods),
        n_outputs=int(args.n_outputs),
        r_frac=float(args.r_frac),
        metric=args.metric,  # type: ignore[arg-type]
        initial_sun=int(args.initial_sun),
        L_sun=float(args.L_sun),
        A=float(args.A),
        phase_power=float(args.phase_power),
        role_rotate=role_rotate,
        illumination=str(args.illumination),
        occultation=not bool(args.no_occultation),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pv = report["period_variance"]["F_total"]
    print(
        f"evenness={report['evenness']} A={args.A} role_rotate={role_rotate} "
        f"F_var={pv['F_var']:.4e} F_cv={pv['F_cv']:.3f} "
        f"F_range=[{pv['F_min']:.3e},{pv['F_max']:.3e}] → {out}",
        flush=True,
    )
    for ev in report["role_events"][:8]:
        print(
            f"  t={ev['swap_time']:.4f} pair={ev['pair']} "
            f"sun {ev['sun_before']}→{ev['sun_after']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
