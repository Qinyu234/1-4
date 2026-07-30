#!/usr/bin/env python3
"""Plot / animate diverse shape families from accepted SQLite passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairy_orbit.design.seeds import save_seed
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate
from fairy_orbit.observe.continuation import attach_central_mass
from fairy_orbit.observe.shape_families import (
    families_to_dict,
    select_diverse_families,
)
from fairy_orbit.store.search_db import DEFAULT_SEARCH_DB_NAME, ChoreographySearchStore
from fairy_orbit.viz.orbits import export_html_viewer, plot_orbits_3d, plot_orbits_xy

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
        n_outputs=max(120, int(240 * periods)),
        config=ReboundConfig(
            stop_on_escape=False,
            stop_on_collision=False,
            epsilon=0.0,
            dt=max(t_end / 500.0, 1e-3),
            min_dt=1e-5,
        ),
    )


def plot_one(
    seed,
    out_dir: Path,
    *,
    title: str,
    periods: float = 1.0,
    animate: bool = True,
    max_frames: int = 160,
    meta_extra: dict | None = None,
) -> None:
    traj = _integrate_seed(seed, periods=periods)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_seed(seed, out_dir / "seed.json")
    plot_orbits_xy(traj, out_dir / "orbit_xy.png", title=title)
    plot_orbits_3d(traj, out_dir / "orbit_3d.png", title=title)
    html_path = None
    if animate:
        html_path = export_html_viewer(
            traj,
            out_dir / "orbit_anim.html",
            title=title,
            max_frames=max_frames,
            trail=max(20, max_frames // 5),
        )
    meta = {
        "id": seed.id,
        "n_bodies": seed.n_bodies,
        "period": seed.period,
        "periods": periods,
        "anim": None if html_path is None else str(html_path),
        "notes": seed.notes,
    }
    if meta_extra:
        meta.update(meta_extra)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}", flush=True)


def plot_diverse_for_n(
    n: int,
    out_root: Path,
    *,
    n_families: int,
    min_sep: float,
    periods: float,
    animate: bool,
    max_frames: int,
    max_residual: float,
) -> None:
    db = OUT / f"choreography_search_n{n}" / DEFAULT_SEARCH_DB_NAME
    if not db.exists():
        print(f"skip missing db {db}", flush=True)
        return
    with ChoreographySearchStore(db) as store:
        picks = select_diverse_families(
            store, n, n_families=n_families, min_sep=min_sep, max_residual=max_residual
        )
        catalogue = families_to_dict(picks)
        catalogue["n"] = n
        fam_root = out_root / f"choreo_n{n}_families"
        fam_root.mkdir(parents=True, exist_ok=True)
        (fam_root / "families.json").write_text(
            json.dumps(catalogue, indent=2), encoding="utf-8"
        )
        print(
            f"n={n}: selected {len(picks)} families "
            f"(pool residual<={max_residual:g}; ok_gate={store.count_passed(n)})",
            flush=True,
        )
        for p in picks:
            title = (
                f"N={n} family{p.family_id} trial={p.record.trial_no} "
                f"res={p.residual:.2e} sep={p.min_dist_to_prev:.3f}"
            )
            dest = fam_root / f"family_{p.family_id:02d}_trial_{p.record.trial_no:05d}"
            plot_one(
                p.seed,
                dest,
                title=title,
                periods=periods,
                animate=animate,
                max_frames=max_frames,
                meta_extra={
                    "family_id": p.family_id,
                    "trial_no": p.record.trial_no,
                    "residual": p.residual,
                    "min_dist_to_prev": p.min_dist_to_prev,
                    "result_fp": p.record.result_fp,
                },
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot diverse shape families (not only residual-best)"
    )
    p.add_argument("--n", type=int, nargs="*", default=[4, 5], choices=[4, 5])
    p.add_argument("--n-families", type=int, default=6, help="target families per N")
    p.add_argument(
        "--min-sep",
        type=float,
        default=0.15,
        help="minimum shape-feature separation between families",
    )
    p.add_argument("--periods", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=160)
    p.add_argument(
        "--max-residual",
        type=float,
        default=1e-6,
        help="only use passes with polish residual <= this",
    )
    p.add_argument("--no-anim", action="store_true")
    p.add_argument("--out", type=Path, default=OUT / "best_orbit_plots")
    args = p.parse_args()

    for n in args.n:
        plot_diverse_for_n(
            n,
            args.out,
            n_families=args.n_families,
            min_sep=args.min_sep,
            periods=args.periods,
            animate=not args.no_anim,
            max_frames=args.max_frames,
            max_residual=args.max_residual,
        )


if __name__ == "__main__":
    main()
