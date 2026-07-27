#!/usr/bin/env python3
"""
Re-integrate top campaign candidates and write orbit / closure plots.

Reads summary.json (long_campaign or beam_search), plots unique top seeds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.design.manifold import elements_for_index
from fairy_orbit.observe.peo import evaluate_peo
from fairy_orbit.observe.rep_error import RepSigmas
from fairy_orbit.observe.search import FreeParams, SeedAnchors, to_manifold
from fairy_orbit.viz.orbits import plot_orbits_3d, plot_orbits_xy

ROOT = Path(__file__).resolve().parents[1]


def _unique_bests(results: list[dict], top_n: int) -> list[dict]:
    ranked = sorted([r for r in results if r.get("best")], key=lambda r: r["best"]["loss"])
    seen: set[tuple[float, float]] = set()
    out = []
    for r in ranked:
        key = (float(r["seed"]["m"]), float(r["seed"]["e"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= top_n:
            break
    return out


def plot_closure(series, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in ("E_r", "E_v", "E_a", "E_e"):
        if k in series.channels:
            ax.semilogy(series.times, np.maximum(series.channels[k], 1e-30), label=k)
    ax.set_xlabel("t")
    ax.set_ylabel("error")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_elements_ladder(params, out: Path, title: str) -> None:
    idx = np.arange(4)
    a = [elements_for_index(params, i).a for i in idx]
    e = [elements_for_index(params, i).e for i in idx]
    M = [elements_for_index(params, i).M for i in idx]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].plot(idx, a, "o-")
    axes[0].set_title("a_i")
    axes[1].plot(idx, e, "o-")
    axes[1].set_title("e_i")
    axes[2].plot(idx, M, "o-")
    axes[2].set_title("M_i")
    for ax in axes:
        ax.set_xlabel("i (nearest=0)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot orbit figures for campaign bests")
    p.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "experiments" / "output" / "long_campaign" / "summary.json",
    )
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--n-periods", type=float, default=2.0)
    p.add_argument("--n-outputs", type=int, default=200)
    p.add_argument(
        "--sigmas",
        type=Path,
        default=ROOT / "experiments" / "output" / "rep_error" / "sigmas.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "long_campaign" / "orbits",
    )
    args = p.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    results = data.get("results", data if isinstance(data, list) else [])
    picks = _unique_bests(results, args.top)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sigmas = RepSigmas.from_json(args.sigmas) if args.sigmas.exists() else RepSigmas()
    t_end = args.n_periods * 2.0 * math.pi

    index_rows = [
        "# Campaign orbit gallery",
        "",
        f"t_end = {args.n_periods} × 2π = {t_end:.4f}",
        "",
        "| # | m | e | loss | Er | Ev | a-span | plots |",
        "|---|---|---|------|----|----|--------|-------|",
    ]

    for i, r in enumerate(picks):
        seed = SeedAnchors(m=float(r["seed"]["m"]), e=float(r["seed"]["e"]))
        free = FreeParams(**r["best"]["params"])
        params = to_manifold(seed, free)
        tag = f"{i:02d}_m{seed.m:.0e}_e{seed.e:.2f}_L{r['best']['loss']:.3g}".replace("+", "")
        sub = out / tag
        sub.mkdir(exist_ok=True)

        result = evaluate_peo(
            params, t_end=t_end, n_outputs=args.n_outputs, sigmas=sigmas
        )
        (sub / "summary.json").write_text(
            json.dumps(
                {
                    "seed": r["seed"],
                    "free": free.as_dict(),
                    "theta": list(params.as_theta()),
                    "campaign_loss": r["best"]["loss"],
                    "re_eval": result.summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        plot_elements_ladder(params, sub / "elements_ladder.png", title=tag)
        if result.traj is not None:
            plot_orbits_xy(
                result.traj,
                sub / "orbit_xy.png",
                title=f"{tag}  loss={result.summary.get('score')}",
            )
            plot_orbits_3d(result.traj, sub / "orbit_3d.png", title=tag)
        if result.closure_rep is not None:
            plot_closure(result.closure_rep, sub / "closure.png", title=tag)

        s = result.summary
        index_rows.append(
            f"| {i} | {seed.m:.0e} | {seed.e:.3f} | {r['best']['loss']:.4g} | "
            f"{s.get('E_r_final', float('nan')):.3g} | {s.get('E_v_final', float('nan')):.3g} | "
            f"{s.get('a_span', float('nan')):.3g} | `{tag}/` |"
        )
        print(f"wrote {sub}", flush=True)

    (out / "INDEX.md").write_text("\n".join(index_rows) + "\n", encoding="utf-8")
    print(f"done → {out / 'INDEX.md'}", flush=True)


if __name__ == "__main__":
    main()
