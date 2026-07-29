#!/usr/bin/env python3
"""Plot PROMPT campaign best / final orbits (xy + 3d)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairy_orbit.design.seeds import load_seed
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate
from fairy_orbit.observe.continuation import attach_central_mass
from fairy_orbit.viz.orbits import plot_orbits_3d, plot_orbits_xy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "output"


def _integrate_seed(seed, *, periods: float = 1.0, M_c: float | None = None):
    if M_c is not None and M_c > 0:
        sys = attach_central_mass(seed, float(M_c))
    else:
        sys = seed.to_system()
    t_end = float(seed.period) * float(periods)
    return integrate(
        sys,
        t_end=t_end,
        n_outputs=max(80, int(200 * periods)),
        config=ReboundConfig(
            stop_on_escape=False,
            stop_on_collision=False,
            epsilon=0.0,
            dt=max(t_end / 400.0, 1e-3),
            min_dt=1e-5,
        ),
    )


def plot_one(
    json_path: Path,
    out_dir: Path,
    *,
    title: str,
    M_c: float | None = None,
    periods: float = 1.0,
) -> None:
    seed = load_seed(json_path)
    traj = _integrate_seed(seed, periods=periods, M_c=M_c)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = title
    plot_orbits_xy(traj, out_dir / "orbit_xy.png", title=tag)
    plot_orbits_3d(traj, out_dir / "orbit_3d.png", title=tag)
    meta = {
        "source": str(json_path),
        "id": seed.id,
        "n_bodies": seed.n_bodies,
        "period": seed.period,
        "M_c": M_c,
        "periods": periods,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot campaign best/final orbits")
    p.add_argument("--periods", type=float, default=1.0)
    p.add_argument(
        "--out",
        type=Path,
        default=OUT / "best_orbit_plots",
    )
    args = p.parse_args()

    jobs = [
        (
            OUT / "choreography_search_n4" / "best.json",
            args.out / "choreo_n4_best",
            "choreography N=4 best",
            None,
        ),
        (
            OUT / "choreography_search_n5" / "best.json",
            args.out / "choreo_n5_best",
            "choreography N=5 best",
            None,
        ),
        (
            OUT / "continuation_n4" / "final.json",
            args.out / "continuation_n4_Mc1",
            "continuation N=4 final (Mc=1)",
            1.0,
        ),
        (
            OUT / "continuation_n5" / "final.json",
            args.out / "continuation_n5_final",
            "continuation N=5 final (μ-scan)",
            None,
        ),
    ]
    for src, dest, title, mc in jobs:
        if not src.exists():
            print(f"skip missing {src}", flush=True)
            continue
        plot_one(src, dest, title=title, M_c=mc, periods=args.periods)


if __name__ == "__main__":
    main()
