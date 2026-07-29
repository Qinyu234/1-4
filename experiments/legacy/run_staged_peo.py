#!/usr/bin/env python3
"""
Staged search for ABCD→BCDA reachability (then detailed σ-score).

Requires Stage-A sigmas.json first (base error). Then:
  一阶 linear coarse (soft residual) →
  二阶 expanding local grid →
  三阶 stain flood + optional a2→e2→M2→v1 (if no survivors yet) →
  细致 score on gate survivors

Example:
  python experiments/run_staged_peo.py --out experiments/output/staged_peo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.observe.rep_error import load_required_sigmas
from fairy_orbit.observe.search import SearchBounds
from fairy_orbit.observe.staged_search import StagedConfig, run_staged_search

ROOT = Path(__file__).resolve().parents[1]
SIGMAS = ROOT / "experiments" / "output" / "rep_error" / "sigmas.json"
OUT_DEFAULT = ROOT / "experiments" / "output" / "staged_peo"


def _log(msg: str) -> None:
    print(msg, flush=True)


def plot_soft(stage1, stage2, survivors, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for samples, label, color in (
        (stage1, "stage1", "C0"),
        (stage2, "stage2", "C1"),
    ):
        if not samples:
            continue
        soft = np.array([s.soft_choreo for s in samples])
        axes[0].semilogy(np.maximum(soft, 1e-12), ".", ms=3, alpha=0.7, label=label, color=color)
    axes[0].set_xlabel("sample index")
    axes[0].set_ylabel("soft_choreo")
    axes[0].set_title("reachability residual")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for samples, label, marker in (
        (stage1, "s1", "."),
        (stage2, "s2", "+"),
        (survivors, "surv", "*"),
    ):
        if not samples:
            continue
        ms = np.log10([s.m for s in samples])
        es = [s.e for s in samples]
        axes[1].scatter(ms, es, s=40 if label == "surv" else 18, marker=marker, label=label, alpha=0.8)
    axes[1].set_xlabel(r"$\log_{10} m$")
    axes[1].set_ylabel("e")
    axes[1].set_title("(m,e) coverage")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "staged_soft.png", dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Staged ABCD→BCDA reachability search")
    p.add_argument("--sigmas", type=Path, default=SIGMAS)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--n-m", type=int, default=5)
    p.add_argument("--n-e", type=int, default=4)
    p.add_argument("--stage1-points", type=int, default=3)
    p.add_argument("--stage1-top-k", type=int, default=8)
    p.add_argument("--stage2-points", type=int, default=3)
    p.add_argument("--stage2-rounds", type=int, default=3)
    p.add_argument("--stage3-points", type=int, default=3)
    p.add_argument("--stain-frac", type=float, default=0.25)
    p.add_argument("--n-periods", type=float, default=2.0)
    p.add_argument("--n-outputs-coarse", type=int, default=80)
    p.add_argument("--n-outputs-fine", type=int, default=160)
    p.add_argument("--log-m-lo", type=float, default=-6.0)
    p.add_argument("--log-m-hi", type=float, default=-2.0)
    p.add_argument("--e-lo", type=float, default=0.0)
    p.add_argument("--e-hi", type=float, default=0.20)
    p.add_argument("--force-stage3", action="store_true", help="run stain even if earlier survivors")
    p.add_argument("--no-unlock", action="store_true", help="skip a2→e2→M2→v1 unlock")
    p.add_argument("--smoke", action="store_true", help="tiny grids for a quick run")
    p.add_argument(
        "--wide",
        action="store_true",
        help="very wide (m,e) + free bounds near HARD_BOUNDS",
    )
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    t_load = time.perf_counter()
    sigmas = load_required_sigmas(args.sigmas)
    _log(f"sigmas source={sigmas.source} n={sigmas.n_samples} ({time.perf_counter()-t_load:.2f}s)")

    if args.wide:
        # Override CLI defaults unless user explicitly changed them via non-defaults —
        # here we always apply the wide box when --wide is set.
        args.log_m_lo, args.log_m_hi = -7.0, -1.5
        args.e_lo, args.e_hi = 0.0, 0.40
        wide_bounds = SearchBounds(
            a1=(0.05, 1.50),
            e1=(-0.20, 0.30),
            M1=(0.2, 10.0),
            vx=(-0.40, 0.40),
            vy=(-0.40, 0.40),
            vz=(-0.40, 0.40),
            a2=(-0.12, 0.12),
            e2=(-0.06, 0.06),
            M2=(-1.8, 1.8),
            v1x=(-0.15, 0.15),
            v1y=(-0.15, 0.15),
            v1z=(-0.15, 0.15),
        )
        _log(
            f"WIDE bounds log_m=({args.log_m_lo},{args.log_m_hi}) "
            f"e=({args.e_lo},{args.e_hi}) a1={wide_bounds.a1} "
            f"kick=±{wide_bounds.vx[1]} M1={wide_bounds.M1}"
        )
    else:
        wide_bounds = None

    if args.smoke:
        cfg = StagedConfig(
            log_m=(args.log_m_lo, args.log_m_hi),
            e_range=(args.e_lo, args.e_hi),
            n_m=2,
            n_e=2,
            stage1_axes=("a1", "M1"),
            stage1_points=2,
            stage1_top_k=3,
            stage2_axes=("a1", "M1"),
            stage2_points=2,
            stage2_expand_rounds=1,
            stage3_points=2,
            stain_frac=0.5,
            stain_max_seeds=2,
            flood_chebyshev=0,
            n_periods=args.n_periods,
            n_outputs_coarse=40,
            n_outputs_fine=60,
            unlock_high_order=False,
            free_bounds=wide_bounds
            or SearchBounds(
                a1=(0.10, 0.30),
                e1=(-0.02, 0.04),
                M1=(0.5, 4.0),
                vx=(-0.05, 0.05),
                vy=(-0.05, 0.05),
                vz=(-0.05, 0.05),
            ),
        )
    else:
        cfg = StagedConfig(
            log_m=(args.log_m_lo, args.log_m_hi),
            e_range=(args.e_lo, args.e_hi),
            n_m=args.n_m,
            n_e=args.n_e,
            stage1_points=args.stage1_points,
            stage1_top_k=args.stage1_top_k,
            stage2_points=args.stage2_points,
            stage2_expand_rounds=args.stage2_rounds,
            stage3_points=args.stage3_points,
            stain_frac=args.stain_frac,
            n_periods=args.n_periods,
            n_outputs_coarse=args.n_outputs_coarse,
            n_outputs_fine=args.n_outputs_fine,
            unlock_high_order=not args.no_unlock,
            free_bounds=wide_bounds or SearchBounds(),
        )
        if args.wide:
            # Slightly denser seed hunt over the wide box.
            cfg = StagedConfig(
                log_m=cfg.log_m,
                e_range=cfg.e_range,
                n_m=max(args.n_m, 7),
                n_e=max(args.n_e, 5),
                stage1_axes=("a1", "e1", "M1"),
                stage1_points=max(args.stage1_points, 3),
                stage1_top_k=max(args.stage1_top_k, 12),
                stage2_axes=("a1", "e1", "M1", "vx", "vy", "vz"),
                stage2_points=max(args.stage2_points, 3),
                stage2_expand_rounds=max(args.stage2_rounds, 4),
                stage2_expand_grow=0.45,
                stage2_max_product=120,
                stage3_points=max(args.stage3_points, 3),
                stain_frac=args.stain_frac,
                stain_max_seeds=16,
                flood_chebyshev=1,
                unlock_high_order=not args.no_unlock,
                n_periods=args.n_periods,
                n_outputs_coarse=args.n_outputs_coarse,
                n_outputs_fine=args.n_outputs_fine,
                free_bounds=wide_bounds or SearchBounds(),
            )

    _log(
        f"staged n_m={cfg.n_m} n_e={cfg.n_e} s1_pts={cfg.stage1_points} "
        f"s2_rounds={cfg.stage2_expand_rounds} unlock={cfg.unlock_high_order} "
        f"a1={cfg.free_bounds.a1}"
    )
    (out / "STATUS.md").write_text(
        f"# Staged PEO (running)\n\n- started wide={args.wide}\n"
        f"- log_m={cfg.log_m} e={cfg.e_range}\n"
        f"- a1={cfg.free_bounds.a1} M1={cfg.free_bounds.M1} kick={cfg.free_bounds.vx}\n",
        encoding="utf-8",
    )
    result = run_staged_search(
        sigmas=sigmas,
        config=cfg,
        skip_stage3_if_survivors=not args.force_stage3,
    )

    payload = result.as_dict()
    payload["stage1_best_soft"] = result.stage1[0].soft_choreo if result.stage1 else None
    payload["stage2_best_soft"] = result.stage2[0].soft_choreo if result.stage2 else None
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_soft(result.stage1, result.stage2, result.survivors, out)

    lines = [
        "# Staged ABCD→BCDA search",
        "",
        f"- sigmas: `{sigmas.source}` (n={sigmas.n_samples}) — Stage-A base error first",
        f"- evals={result.n_evals}, wall={result.wall_s/60:.2f} min",
        f"- stage1={len(result.stage1)}, stage2={len(result.stage2)}, "
        f"stained={len(result.stained)}, stage3={len(result.stage3)}",
        f"- survivors (gate pass + detailed score)={len(result.survivors)}",
        "",
    ]
    if result.survivors:
        best = result.survivors[0]
        lines.append(
            f"- best score={best.score:.6g} m={best.m:.3g} e={best.e:.4g} "
            f"shift={best.summary.get('choreography_shift_k')}"
        )
        lines.append(f"- best free={json.dumps(best.free.as_dict())}")
    else:
        best_soft = min(
            (s.soft_choreo for s in result.stage1 + result.stage2 + result.stage3),
            default=None,
        )
        lines.append(f"- no gate survivors; best soft_choreo={best_soft}")
    lines += ["", "Plot: `staged_soft.png`.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "STATUS.md").write_text(
        f"# Staged PEO\n\nsurvivors={len(result.survivors)} evals={result.n_evals}\n",
        encoding="utf-8",
    )
    _log(
        f"done survivors={len(result.survivors)} evals={result.n_evals} "
        f"wall={result.wall_s:.1f}s → {out / 'REPORT.md'}"
    )


if __name__ == "__main__":
    main()
