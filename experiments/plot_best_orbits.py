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


def _integrate_seed(
    seed,
    *,
    periods: float = 1.0,
    M_c: float | None = None,
    precise: bool = False,
):
    if M_c is not None and M_c > 0:
        sys = attach_central_mass(seed, float(M_c))
    else:
        sys = seed.to_system()
    t_end = float(seed.period) * float(periods)
    if precise:
        # Free-N / coplanar: dense IAS15; keep min_dt tiny (energy, not Floquet).
        n_out = max(400, int(480 * periods))
        cfg = ReboundConfig(
            stop_on_escape=False,
            stop_on_collision=False,
            epsilon=0.0,
            dt=max(t_end / max(n_out * 4, 1), 1e-5),
            min_dt=1e-12,
        )
    else:
        n_out = max(120, int(240 * periods))
        cfg = ReboundConfig(
            stop_on_escape=False,
            stop_on_collision=False,
            epsilon=0.0,
            dt=max(t_end / 500.0, 1e-3),
            min_dt=1e-5,
        )
    return integrate(sys, t_end=t_end, n_outputs=n_out, config=cfg)


def _closure_error(seed, traj) -> dict[str, float]:
    """COM-frame max body drift at final time (absolute IC repeat)."""
    import numpy as np

    pos0 = np.asarray(seed.positions, dtype=float)
    n = int(seed.n_bodies)
    posT = np.asarray(traj.positions[-1, :n], dtype=float)
    if traj.positions.shape[1] > n:
        posT = np.asarray(traj.positions[-1, -n:], dtype=float)
    r0 = pos0 - pos0.mean(axis=0)
    rT = posT - posT.mean(axis=0)
    err = float(np.max(np.linalg.norm(rT - r0, axis=1)))
    scale = float(np.linalg.norm(r0)) + 1e-15
    return {"drift_abs": err, "drift_rel": err / scale, "frame": "inertial"}


def _map_residual(seed, traj) -> dict[str, float]:
    """Relative-map residual at traj end vs IC (Kabsch + identity perm^n)."""
    import numpy as np

    from fairy_orbit.observe.closure import closure_for_perm

    n = int(seed.n_bodies)
    r0 = np.asarray(seed.positions, dtype=float)
    v0 = np.asarray(seed.velocities, dtype=float)
    r = np.asarray(traj.positions[-1, :n], dtype=float)
    v = np.asarray(traj.velocities[-1, :n], dtype=float)
    cl = closure_for_perm(r, v, r0, v0, tuple(range(n)))
    scale = float(np.sum((r0 - r0.mean(axis=0)) ** 2)) + 1e-30
    return {
        "E_r": float(cl.E_r),
        "E_v": float(cl.E_v),
        "E_r_rel": float(cl.E_r / scale),
        "E_v_rel": float(cl.E_v / scale),
    }


def _corotating_trajectory(seed, traj):
    """Undo continuous rotation from §3.2 ``R`` so the shape sits still."""
    import numpy as np

    from fairy_orbit.engine.trajectory import Trajectory
    from fairy_orbit.observe.choreography_verify import verify_choreography_Tn

    gate = verify_choreography_Tn(
        seed.to_system(), float(seed.period), shift=1, atol_rel=1e-5
    )
    if not gate.ok:
        return traj
    axis = np.asarray(gate.axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    ang = float(gate.angle)
    tau = float(gate.tau)
    times = np.asarray(traj.times, dtype=float)
    pos = np.asarray(traj.positions, dtype=float).copy()
    n = pos.shape[1]
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    for fi, t in enumerate(times):
        phi = -ang * (t / max(tau, 1e-15))
        c, s = np.cos(phi), np.sin(phi)
        Rm = np.eye(3) + s * K + (1.0 - c) * (K @ K)
        for i in range(n):
            pos[fi, i] = Rm @ pos[fi, i]
    return Trajectory(
        times=times,
        positions=pos,
        velocities=np.asarray(traj.velocities, dtype=float),
        energies=np.asarray(traj.energies, dtype=float),
        angular_momenta=np.asarray(traj.angular_momenta, dtype=float),
        labels=list(traj.labels),
        G=float(traj.G),
        masses=None if traj.masses is None else np.asarray(traj.masses, dtype=float),
        status=getattr(traj, "status", "success"),
    )


def _tile_one_period(traj_1p, periods: float):
    """Repeat a closed 1-period traj ``periods`` times (stroboscopic multi-period).

    Floquet-unstable free-N orbits shed relative-map accuracy after a few
    periods under long IAS15; tiling the verified 1-period loop keeps the
    multi-period animation honest to the choreography shape.
    """
    import math

    import numpy as np

    from fairy_orbit.engine.trajectory import Trajectory

    p = float(periods)
    if p <= 0:
        raise ValueError("periods must be positive")
    n_full = int(math.floor(p + 1e-12))
    frac = p - n_full
    if n_full < 1:
        n_full = 0
        frac = p
    segments = n_full + (1 if frac > 1e-12 else 0)
    if segments < 1:
        segments = 1
        frac = min(p, 1.0)

    t0 = np.asarray(traj_1p.times, dtype=float)
    p0 = np.asarray(traj_1p.positions, dtype=float)
    v0 = np.asarray(traj_1p.velocities, dtype=float)
    e0 = np.asarray(traj_1p.energies, dtype=float)
    l0 = np.asarray(traj_1p.angular_momenta, dtype=float)
    T = float(t0[-1] - t0[0]) if len(t0) > 1 else float(t0[-1])

    times: list[np.ndarray] = []
    pos: list[np.ndarray] = []
    vel: list[np.ndarray] = []
    eng: list[np.ndarray] = []
    ang: list[np.ndarray] = []
    for k in range(segments):
        use_frac = 1.0 if k < n_full else float(frac)
        use_frac = max(min(use_frac, 1.0), 1e-9)
        if use_frac >= 1.0 - 1e-12:
            tk = t0 + k * T
            pk, vk, ek, lk = p0, v0, e0, l0
        else:
            t_cut = t0[0] + use_frac * T
            m = int(np.searchsorted(t0, t_cut, side="right"))
            m = max(2, min(m, len(t0)))
            sl = slice(0, m)
            tk = t0[sl] + k * T
            pk, vk, ek, lk = p0[sl], v0[sl], e0[sl], l0[sl]
        if k > 0:
            tk, pk, vk, ek, lk = tk[1:], pk[1:], vk[1:], ek[1:], lk[1:]
        times.append(tk)
        pos.append(pk)
        vel.append(vk)
        eng.append(ek)
        ang.append(lk)

    return Trajectory(
        times=np.concatenate(times),
        positions=np.concatenate(pos, axis=0),
        velocities=np.concatenate(vel, axis=0),
        energies=np.concatenate(eng),
        angular_momenta=np.concatenate(ang, axis=0),
        labels=list(traj_1p.labels),
        G=float(traj_1p.G),
        masses=None
        if traj_1p.masses is None
        else np.asarray(traj_1p.masses, dtype=float),
        status=getattr(traj_1p, "status", "success"),
    )


def _project_xy(traj):
    """Display-only: zero z so face-on coplanar view is exact."""
    import numpy as np

    from fairy_orbit.engine.trajectory import Trajectory

    pos = np.asarray(traj.positions, dtype=float).copy()
    vel = np.asarray(traj.velocities, dtype=float).copy()
    pos[:, :, 2] = 0.0
    vel[:, :, 2] = 0.0
    return Trajectory(
        times=np.asarray(traj.times, dtype=float),
        positions=pos,
        velocities=vel,
        energies=np.asarray(traj.energies, dtype=float),
        angular_momenta=np.asarray(traj.angular_momenta, dtype=float),
        labels=list(traj.labels),
        G=float(traj.G),
        masses=None if traj.masses is None else np.asarray(traj.masses, dtype=float),
        status=getattr(traj, "status", "success"),
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
    precise: bool = False,
    flatten_xy: bool = False,
    corotating: bool = False,
    tile_periods: bool = False,
) -> None:
    import numpy as np

    seed = load_seed(json_path)
    # Never zero IC z/vz for dynamics — that ejects the near-planar N=5 family.
    # Multi-period Floquet-unstable free-N: integrate 1P then tile in-frame.
    if tile_periods and M_c is None and float(periods) > 1.0 + 1e-12:
        traj_1p = _integrate_seed(seed, periods=1.0, M_c=None, precise=precise)
        map_1p = _map_residual(seed, traj_1p)
        if corotating:
            traj_1p = _corotating_trajectory(seed, traj_1p)
        traj = _tile_one_period(traj_1p, periods)
        integration = "tiled_1p"
    else:
        traj = _integrate_seed(seed, periods=periods, M_c=M_c, precise=precise)
        map_1p = _map_residual(seed, traj)
        if corotating and M_c is None:
            traj = _corotating_trajectory(seed, traj)
        integration = "direct"

    if flatten_xy:
        traj = _project_xy(traj)

    closure = _closure_error(seed, traj)
    if corotating and M_c is None:
        pos0 = np.asarray(seed.positions, dtype=float)
        pos0 = pos0 - pos0.mean(axis=0)
        if flatten_xy:
            pos0 = np.c_[pos0[:, :2], np.zeros(seed.n_bodies)]
        posT = np.asarray(traj.positions[-1, : seed.n_bodies], dtype=float)
        posT = posT - posT.mean(axis=0)
        err = float(np.max(np.linalg.norm(posT - pos0, axis=1)))
        scale = float(np.linalg.norm(pos0)) + 1e-15
        closure = {
            "drift_abs": err,
            "drift_rel": err / scale,
            "frame": "corotating",
            "integration": integration,
        }

    e = np.asarray(traj.energies, dtype=float)
    dE = float(abs(e[-1] - e[0]) / (abs(e[0]) + 1e-15)) if len(e) else None

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
            trail=max(30, max_frames // 4),
        )
    meta = {
        "source": str(json_path),
        "id": seed.id,
        "n_bodies": seed.n_bodies,
        "period": seed.period,
        "M_c": M_c,
        "periods": periods,
        "precise": precise,
        "flatten_xy": flatten_xy,
        "corotating": corotating,
        "integration": integration,
        "closure": closure,
        "map_residual_1p": map_1p,
        "energy_drift_rel": dE,
        "anim": None if html_path is None else str(html_path),
        "residual_notes": seed.notes,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"wrote {out_dir}"
        + (f" + {html_path.name}" if html_path else "")
        + f" closure_rel={closure['drift_rel']:.3e}"
        + f" map_Er_rel={map_1p['E_r_rel']:.3e}"
        + f" dE={dE:.3e}"
        + f" [{integration}]",
        flush=True,
    )


def _path_a_horizon_jobs(horizons: list[float]) -> list[tuple[Path, float, float]]:
    """Return (seed_path, M_c, horizon_periods) for refined showcases."""
    jobs: list[tuple[Path, float, float]] = []
    for hp in horizons:
        tag = int(hp) if abs(hp - round(hp)) < 1e-12 else hp
        refined = OUT / "best_orbit_plots" / "path_a_best_Mc" / f"state_horizon{tag}.json"
        refined_rep = (
            OUT / "best_orbit_plots" / "path_a_best_Mc" / f"state_horizon{tag}.report.json"
        )
        if not (refined.is_file() and refined_rep.is_file()):
            continue
        try:
            rep = json.loads(refined_rep.read_text(encoding="utf-8"))
            Mc = float(rep.get("M_c") or 0.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if Mc > 0:
            jobs.append((refined, Mc, float(hp)))
    return jobs


def _best_path_a_cycle() -> tuple[Path, float] | None:
    """Pick the highest-Mc *Floquet-stable* Path-A checkpoint across the cycle archive.

    Preferring raw ``M_c_final`` (often 1.0 under loose ``res_tol``) surfaces
    deep saddles that still have tiny symmetry residuals — misleading "best".
    """
    root = OUT / "continuation_n4_cycle"
    if not root.is_dir():
        return None
    best: tuple[Path, float] | None = None
    for d in root.iterdir():
        if not d.is_dir():
            continue
        sweep = d / "floquet_path_sweep.json"
        if sweep.is_file():
            try:
                payload = json.loads(sweep.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            rows = payload.get("rows") or []
            stable_rows = [
                r
                for r in rows
                if r.get("stable") and r.get("path") and Path(r["path"]).is_file()
            ]
            if stable_rows:
                row = max(stable_rows, key=lambda r: float(r["M_c"]))
                path = Path(row["path"])
                Mc = float(row["M_c"])
                if best is None or Mc > best[1]:
                    best = (path, Mc)
                continue
        # Fallback: tiny-Mc final only if no Floquet sweep
        final = d / "final.json"
        summary = d / "summary.json"
        if not final.is_file() or not summary.is_file():
            continue
        try:
            Mc = float(json.loads(summary.read_text(encoding="utf-8")).get("M_c_final", 0.0))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if Mc <= 0:
            continue
        if best is None or Mc > best[1]:
            # only use as weak fallback when nothing Floquet-stable found yet
            if best is None:
                best = (final, Mc)
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Plot/animate campaign best orbits")
    p.add_argument("--periods", type=float, default=2.0, help="integrate this many periods")
    p.add_argument("--max-frames", type=int, default=240, help="animation frame budget")
    p.add_argument("--no-anim", action="store_true", help="skip HTML time-slider")
    p.add_argument("--out", type=Path, default=OUT / "best_orbit_plots")
    p.add_argument(
        "--only",
        choices=["n4", "n5", "path_a", "all"],
        default="all",
        help="which showcase to regenerate",
    )
    p.add_argument(
        "--precise",
        action="store_true",
        help="denser multi-period integrate (lower drift)",
    )
    p.add_argument(
        "--flatten-xy",
        action="store_true",
        help="display-only: project traj to z=0 (does NOT rewrite IC dynamics)",
    )
    p.add_argument(
        "--horizon-periods",
        type=str,
        default="3,4",
        help="comma-separated Path A refined horizons to plot "
        "(looks for state_horizon{k}.json; empty = Floquet-stable cycle pick)",
    )
    p.add_argument(
        "--seed",
        type=Path,
        default=None,
        help="optional extra seed JSON to plot as custom/",
    )
    args = p.parse_args()

    def _parse_horizons(raw: str) -> list[float]:
        raw = (raw or "").strip()
        if not raw:
            return []
        return [float(x.strip()) for x in raw.split(",") if x.strip()]

    horizons = _parse_horizons(args.horizon_periods)

    n4 = _sync_best_from_db(4) if args.only in {"n4", "all"} else None
    n5 = _sync_best_from_db(5) if args.only in {"n5", "all"} else None
    path_a_horizons = (
        _path_a_horizon_jobs(horizons) if args.only in {"path_a", "all"} else []
    )
    path_a = None
    if args.only in {"path_a", "all"} and not path_a_horizons:
        path_a = _best_path_a_cycle()

    jobs: list[tuple[Path | None, Path, str, float | None, bool]] = []
    if n4 is not None:
        jobs.append(
            (n4, args.out / "choreo_n4_best", "choreography N=4 best (SQLite)", None, False)
        )
    if n5 is not None:
        jobs.append(
            (
                n5,
                args.out / "choreo_n5_best",
                "choreography N=5 coplanar best (free, no Mc)",
                None,
                True,
            )
        )
    for src, Mc, hp in path_a_horizons:
        jobs.append(
            (
                src,
                args.out / f"path_a_best_{int(hp)}P",
                f"Path A best {int(hp)}P obj (M_c={Mc:g})",
                Mc if Mc > 0 else None,
                False,
            )
        )
    if path_a is not None:
        src, Mc = path_a
        jobs.append(
            (
                src,
                args.out / "path_a_best_Mc",
                f"Path A Floquet-stable best (M_c={Mc:g})",
                Mc if Mc > 0 else None,
                False,
            )
        )
    if args.seed is not None:
        jobs.append((args.seed, args.out / "custom", f"custom {args.seed.name}", None, False))

    for src, dest, title, mc, is_n5 in jobs:
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
            precise=bool(args.precise or is_n5),
            flatten_xy=bool(args.flatten_xy or is_n5),
            corotating=bool(is_n5),
            tile_periods=bool(is_n5),
        )


if __name__ == "__main__":
    main()
