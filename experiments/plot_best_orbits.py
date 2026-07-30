#!/usr/bin/env python3
"""Plot / animate accepted choreography best orbits (static PNG + time-slider HTML)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairy_orbit.design.seeds import OrbitSeed, load_seed, save_seed
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate
from fairy_orbit.observe.continuation import attach_central_mass
from fairy_orbit.store.search_db import ChoreographySearchStore, DEFAULT_SEARCH_DB_NAME
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


def _sync_best_from_db(n: int) -> Path | None:
    """Refresh best.json from SQLite if present."""
    d = OUT / f"choreography_search_n{n}"
    db = d / DEFAULT_SEARCH_DB_NAME
    if not db.exists():
        best = d / "best.json"
        return best if best.exists() else None
    with ChoreographySearchStore(db) as store:
        rec = store.best_accepted(n)
        if rec is None or rec.seed_json is None:
            return None
        seed = OrbitSeed.from_dict(rec.seed_json)
        path = d / "best.json"
        save_seed(seed, path)
        return path


def plot_one(
    json_path: Path,
    out_dir: Path,
    *,
    title: str,
    M_c: float | None = None,
    periods: float = 1.0,
    animate: bool = True,
    max_frames: int = 200,
) -> None:
    seed = load_seed(json_path)
    traj = _integrate_seed(seed, periods=periods, M_c=M_c)
    out_dir.mkdir(parents=True, exist_ok=True)
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
        "source": str(json_path),
        "id": seed.id,
        "n_bodies": seed.n_bodies,
        "period": seed.period,
        "M_c": M_c,
        "periods": periods,
        "anim": None if html_path is None else str(html_path),
        "residual_notes": seed.notes,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}" + (f" + {html_path.name}" if html_path else ""), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot/animate campaign best orbits")
    p.add_argument("--periods", type=float, default=1.0, help="integrate this many periods")
    p.add_argument("--max-frames", type=int, default=200, help="animation frame budget")
    p.add_argument("--no-anim", action="store_true", help="skip HTML time-slider")
    p.add_argument("--out", type=Path, default=OUT / "best_orbit_plots")
    args = p.parse_args()

    # Prefer SQLite bests
    n4 = _sync_best_from_db(4)
    n5 = _sync_best_from_db(5)

    jobs = [
        (n4, args.out / "choreo_n4_best", "choreography N=4 best (SQLite)", None),
        (n5, args.out / "choreo_n5_best", "choreography N=5 best (SQLite)", None),
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
        if src is None or not Path(src).exists():
            print(f"skip missing {src}", flush=True)
            continue
        plot_one(
            Path(src),
            dest,
            title=title,
            M_c=mc,
            periods=args.periods,
            animate=not args.no_anim,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
