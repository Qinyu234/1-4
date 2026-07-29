#!/usr/bin/env python3
"""
Progressive-refinement (m,e) heatmap with choreography gate.

Grid starts coarse, then each level inserts midpoints in log(m) and linear(e)
(≈2× resolution per axis). Only unevaluated (m,e) pairs are beam-searched.

Example:
  python experiments/run_me_heatmap.py --levels 3 --nm0 6 --ne0 5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.observe.rep_error import RepSigmas, load_required_sigmas
from fairy_orbit.observe.search import (
    LOSS_FAIL,
    BeamConfig,
    SearchBounds,
    grid_beam_search,
)

ROOT = Path(__file__).resolve().parents[1]
SIGMAS = ROOT / "experiments" / "output" / "rep_error" / "sigmas.json"
OUT_DEFAULT = ROOT / "experiments" / "output" / "campaign_10h" / "heatmaps"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _me_key(m: float, e: float) -> tuple[float, float]:
    return (round(float(m), 14), round(float(e), 12))


def _refine_log_axis(xs: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return xs
    mids = np.sqrt(xs[:-1] * xs[1:])
    return np.sort(np.unique(np.concatenate([xs, mids])))


def _refine_lin_axis(xs: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return xs
    mids = 0.5 * (xs[:-1] + xs[1:])
    return np.sort(np.unique(np.concatenate([xs, mids])))


def progressive_axes(
    m_lo: float,
    m_hi: float,
    e_lo: float,
    e_hi: float,
    nm0: int,
    ne0: int,
    levels: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Return [(ms, es, level_index), ...] for each refinement stage."""
    ms = np.logspace(math.log10(m_lo), math.log10(m_hi), nm0)
    es = np.linspace(e_lo, e_hi, ne0)
    stages = [(ms.copy(), es.copy(), 0)]
    for lv in range(1, levels):
        ms = _refine_log_axis(ms)
        es = _refine_lin_axis(es)
        stages.append((ms.copy(), es.copy(), lv))
    return stages


def eval_me_cell(
    m: float,
    e: float,
    *,
    bounds: SearchBounds,
    sigmas: RepSigmas,
    cfg: BeamConfig,
) -> dict:
    e1_hi = min(bounds.e1[1], max(0.0, (0.95 - float(e)) / 3.0))
    e1_lo = max(bounds.e1[0], -float(e) / 3.0 + 1e-6)
    if e1_lo > e1_hi:
        e1_lo, e1_hi = 0.0, 0.0
    local_bounds = SearchBounds(
        a1=bounds.a1,
        e1=(e1_lo, e1_hi),
        M1=bounds.M1,
        vx=bounds.vx,
        vy=bounds.vy,
        vz=bounds.vz,
    )
    res = grid_beam_search(float(m), float(e), bounds=local_bounds, config=cfg, sigmas=sigmas)
    best = res.best
    if best is not None and math.isfinite(best.loss) and best.loss < LOSS_FAIL * 0.5:
        return {
            "m": float(m),
            "e": float(e),
            "loss": float(best.loss),
            "status": best.status,
            "params": best.params.as_dict(),
            "E_r_final": best.summary.get("E_r_final"),
            "E_v_final": best.summary.get("E_v_final"),
            "n_evals": res.n_evals,
            "wall_s": res.wall_s,
        }
    return {
        "m": float(m),
        "e": float(e),
        "loss": float("nan"),
        "status": None if best is None else best.status,
        "params": None,
        "E_r_final": None,
        "E_v_final": None,
        "n_evals": res.n_evals,
        "wall_s": res.wall_s,
    }


def plot_scatter_field(
    rows: list[dict],
    out_png: Path,
    *,
    field: str,
    title: str,
    m_lo: float,
    m_hi: float,
    e_lo: float,
    e_hi: float,
    plot_nx: int = 80,
    plot_ne: int = 60,
) -> None:
    finite = [r for r in rows if isinstance(r.get(field), (int, float)) and math.isfinite(r[field])]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if not finite:
        ax.text(0.5, 0.5, f"no finite {field}", ha="center", transform=ax.transAxes)
    else:
        xs = np.array([r["m"] for r in finite])
        ys = np.array([r["e"] for r in finite])
        zs = np.array([r[field] for r in finite])
        # display grid (log-m × linear-e)
        gm = np.logspace(math.log10(m_lo), math.log10(m_hi), plot_nx)
        ge = np.linspace(e_lo, e_hi, plot_ne)
        Gm, Ge = np.meshgrid(gm, ge)
        try:
            from scipy.interpolate import griddata

            Zi = griddata(
                (np.log10(xs), ys),
                zs,
                (np.log10(Gm), Ge),
                method="linear",
            )
        except Exception:
            Zi = np.full(Gm.shape, np.nan)
        vmin, vmax = float(np.nanmin(zs)), float(np.nanpercentile(zs, 95))
        im = ax.imshow(
            Zi,
            origin="lower",
            aspect="auto",
            extent=[math.log10(m_lo), math.log10(m_hi), e_lo, e_hi],
            cmap="magma_r",
            vmin=vmin,
            vmax=max(vmax, vmin + 1e-12),
        )
        fig.colorbar(im, ax=ax, fraction=0.046, label=field)
        ax.scatter(np.log10(xs), ys, c="w", s=8, alpha=0.5, linewidths=0)
        j = int(np.nanargmin(zs))
        ax.plot(np.log10(xs[j]), ys[j], "c+", ms=14, mew=2)
    ax.set_xlabel(r"$\log_{10} m$")
    ax.set_ylabel("e")
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Progressive (m,e) heatmap")
    p.add_argument("--levels", type=int, default=3, help="refinement levels (≈2× axes each level)")
    p.add_argument("--nm0", type=int, default=6)
    p.add_argument("--ne0", type=int, default=5)
    p.add_argument("--m-lo", type=float, default=1e-6)
    p.add_argument("--m-hi", type=float, default=1e-2)
    p.add_argument("--e-lo", type=float, default=0.0)
    p.add_argument("--e-hi", type=float, default=0.12)
    p.add_argument("--cell-evals", type=int, default=120)
    p.add_argument("--n-outputs", type=int, default=100)
    p.add_argument("--n-periods", type=float, default=2.0)
    p.add_argument("--wall-hours", type=float, default=0.0, help="0 = no wall")
    p.add_argument("--resume", action="store_true", help="load me_grid.json and skip done cells")
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "STATUS.md"
    grid_path = out / "me_grid.json"
    t0 = time.perf_counter()
    deadline = t0 + args.wall_hours * 3600.0 if args.wall_hours > 0 else None

    sigmas = load_required_sigmas(SIGMAS)
    bounds = SearchBounds(
        a1=(0.08, 0.50),
        e1=(-0.03, 0.06),
        M1=(0.3, 6.5),
        vx=(-0.08, 0.08),
        vy=(-0.08, 0.08),
        vz=(-0.08, 0.08),
    )
    cfg = BeamConfig(
        beam_width=3,
        coarse_points=2,
        refine_points=(5,),
        n_periods=args.n_periods,
        n_outputs=args.n_outputs,
        max_evals=args.cell_evals,
        bisect_iters=4,
        grad_steps=4,
    )

    stages = progressive_axes(
        args.m_lo, args.m_hi, args.e_lo, args.e_hi, args.nm0, args.ne0, args.levels
    )
    done: dict[tuple[float, float], dict] = {}
    level_rows: dict[int, list[dict]] = {}
    if args.resume and grid_path.exists():
        prev = json.loads(grid_path.read_text(encoding="utf-8"))
        for r in prev.get("rows", []):
            done[_me_key(r["m"], r["e"])] = r
        for k, v in (prev.get("level_counts") or {}).items():
            level_rows[int(k)] = []
        _log(f"resume: loaded {len(done)} cells from {grid_path}")

    def _checkpoint(lv: int) -> None:
        rows = list(done.values())
        payload = {
            "levels": args.levels,
            "nm0": args.nm0,
            "ne0": args.ne0,
            "m_range": [args.m_lo, args.m_hi],
            "e_range": [args.e_lo, args.e_hi],
            "n_evaluated": len(rows),
            "n_finite": sum(
                1 for r in rows if isinstance(r["loss"], float) and math.isfinite(r["loss"])
            ),
            "rows": rows,
            "level_counts": {str(k): len(v) for k, v in level_rows.items()},
            "current_level": lv,
        }
        grid_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        plot_scatter_field(
            rows,
            out / "me_loss.png",
            field="loss",
            title=f"PEO loss (m,e) progressive L{lv}",
            m_lo=args.m_lo,
            m_hi=args.m_hi,
            e_lo=args.e_lo,
            e_hi=args.e_hi,
        )

    for ms, es, lv in stages:
        if deadline is not None and time.perf_counter() > deadline - 30:
            _log(f"wall stop before level {lv}")
            break
        new_pts = [
            (float(m), float(e))
            for m in ms
            for e in es
            if _me_key(m, e) not in done
        ]
        _log(f"level {lv}: grid {len(ms)}×{len(es)} new_pts={len(new_pts)} done={len(done)}")
        status_path.write_text(
            f"# (m,e) heatmap progressive\n\n"
            f"- level {lv}/{args.levels - 1}\n"
            f"- grid {len(ms)}×{len(es)}, new={len(new_pts)}, total={len(done)}\n",
            encoding="utf-8",
        )
        level_rows.setdefault(lv, [])
        for i, (m, e) in enumerate(new_pts):
            if deadline is not None and time.perf_counter() > deadline - 20:
                _log("wall stop mid-level")
                break
            row = eval_me_cell(m, e, bounds=bounds, sigmas=sigmas, cfg=cfg)
            done[_me_key(m, e)] = row
            level_rows[lv].append(row)
            if (i + 1) % 5 == 0 or i + 1 == len(new_pts):
                _log(
                    f"  [{i+1}/{len(new_pts)}] m={m:.2e} e={e:.3f} "
                    f"status={row['status']} loss={row['loss']}"
                )
            if (i + 1) % max(1, args.checkpoint_every) == 0 or i + 1 == len(new_pts):
                _checkpoint(lv)
                status_path.write_text(
                    f"# (m,e) heatmap progressive\n\n"
                    f"- level {lv}/{args.levels - 1}\n"
                    f"- progress {i+1}/{len(new_pts)} new, total={len(done)}\n",
                    encoding="utf-8",
                )
        _checkpoint(lv)

    rows = list(done.values())
    finite = [r for r in rows if isinstance(r["loss"], float) and math.isfinite(r["loss"])]
    best = min(finite, key=lambda r: r["loss"]) if finite else None
    lines = [
        "# Progressive (m,e) heatmap",
        "",
        f"levels={args.levels} start={args.nm0}×{args.ne0} "
        f"final≈{len(stages[-1][0])}×{len(stages[-1][1])}",
        f"evaluated={len(rows)} finite={len(finite)}",
        f"best={None if best is None else best['loss']} at m={best['m']:.2e} e={best['e']:.3f}"
        if best
        else "best=—",
        "",
        "Plots: `me_loss.png` (interpolated), white dots = sample points.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    _log(f"done → {out} finite={len(finite)} best={None if best is None else best['loss']}")


if __name__ == "__main__":
    main()
