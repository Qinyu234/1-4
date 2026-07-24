#!/usr/bin/env python3
"""
Post-scan report: rank SQL store by interest, recompute AMD on top runs, write plots.

Follows docs/DIRECTION.md — REBOUND truth first; AMD as secular exchange diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.core import SystemConfig
from fairy_orbit.design import LadderParams, build_orbital_ladder
from fairy_orbit.observe import diagnose, extract_amd_series, extract_element_series
from fairy_orbit.store import DEFAULT_DB, OrbitStore
from fairy_orbit.viz.orbits import plot_orbit_gallery, plot_orbits_xy
from fairy_orbit.viz.report import save_ladder_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Report hierarchical resonant-chain scan")
    parser.add_argument("--db", type=Path, default=ROOT / DEFAULT_DB)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--min-interest", type=float, default=5.0)
    parser.add_argument("--require-swap", action="store_true", default=True)
    parser.add_argument("--no-require-swap", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "dynamics" / "report",
    )
    parser.add_argument("--t-end", type=float, default=800.0)
    parser.add_argument("--n-outputs", type=int, default=500)
    args = parser.parse_args()
    require_swap = False if args.no_require_swap else True

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    with OrbitStore(args.db) as store:
        n = store.count()
        classes = store.list_classes()
        hits = store.query(
            min_interest=args.min_interest,
            a_order_changed=True if require_swap else None,
            order_by="interest",
            limit=args.top * 3,
        )
        # Prefer success + finite a_rms not huge escape-like
        filtered = [
            r
            for r in hits
            if r.status == "success"
            and r.a_delta_rms is not None
            and r.a_delta_rms < 5.0
        ][: args.top]

    lines = [
        "# Hierarchical Resonant Chain — scan report",
        "",
        f"DB: `{args.db}`  |  runs in DB: **{n}**  |  classes: **{len(classes)}**",
        "",
        "Judgment source: [`docs/DIRECTION.md`](../../../docs/DIRECTION.md).",
        "",
        "## Top classes (by max interest)",
        "",
        "| param_class | n | maxI | mean a_rms | swaps |",
        "|-------------|---|------|------------|-------|",
    ]
    for c in classes[:20]:
        lines.append(
            f"| `{c['param_class']}` | {c['n']} | {c['max_interest']} | "
            f"{c['mean_a_rms']} | {c['n_swap']} |"
        )

    lines += [
        "",
        "## Top changing chains (re-integrated + AMD)",
        "",
        "| id | e | μ | interest | a_Δrms | enc | swap | AMD_total_ptp |",
        "|----|---|---|----------|--------|-----|------|---------------|",
    ]

    diagnoses = []
    details = []
    for rec in filtered:
        cfg = SystemConfig(mass_ratio=rec.mass_ratio)
        params = LadderParams(
            eccentricity=rec.eccentricity,
            a_inner=rec.a_inner,
            period_ratios=tuple(rec.period_ratios),  # type: ignore[arg-type]
            tetrahedral=rec.tetrahedral,
        )
        system = build_orbital_ladder(cfg, params)
        d = diagnose(
            system,
            cfg,
            t_end=args.t_end,
            n_outputs=args.n_outputs,
            ladder=params,
            run_megno=False,
        )
        label = (
            f"id{rec.id} e={rec.eccentricity:.2f} μ={rec.mass_ratio:.0e} "
            f"I={d.summary.get('interest', 0):.1f}"
        )
        diagnoses.append((label, d))
        case_dir = out / f"run_{rec.id:04d}"
        case_dir.mkdir(exist_ok=True)
        plot_orbits_xy(d.trajectory, case_dir / "xy.png", title=label, encounters=d.encounters)
        save_ladder_report(d, case_dir / "report")

        # AMD panel
        el = d.elements
        masses = np.full(el.a.shape[1], cfg.fairy_mass)
        amd = extract_amd_series(el, masses, cfg.mu)
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for j, lab in enumerate(el.labels):
            axes[0].plot(el.times, el.a[:, j], label=lab)
            axes[1].plot(amd.times, amd.amd[:, j], label=lab)
        axes[1].plot(amd.times, amd.amd_total, "k--", lw=1.2, label="total")
        axes[0].set_ylabel("a")
        axes[0].set_title(f"{label} — semi-major axes")
        axes[0].legend(fontsize=7)
        axes[1].set_ylabel("AMD")
        axes[1].set_xlabel("t")
        axes[1].set_title("Angular Momentum Deficit exchange")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(case_dir / "a_amd.png", dpi=140)
        plt.close(fig)

        amd_ptp = float(d.summary.get("amd_total_ptp", np.ptp(amd.amd_total)))
        lines.append(
            f"| {rec.id} | {rec.eccentricity:.2f} | {rec.mass_ratio:.0e} | "
            f"{d.summary.get('interest'):.2f} | {d.summary.get('a_delta_rms'):.3f} | "
            f"{d.summary.get('n_encounters')} | {d.summary.get('a_order_changed')} | "
            f"{amd_ptp:.4g} |"
        )
        details.append(
            {
                "run_id": rec.id,
                "param_class": rec.param_class,
                "summary": d.summary,
            }
        )

    if diagnoses:
        plot_orbit_gallery(diagnoses, out / "orbit_gallery.png", cols=4)

    lines += [
        "",
        "## Next (from DIRECTION)",
        "",
        "1. Inspect top `a_amd.png` — look for anti-correlated AMD / a between rungs.",
        "2. Defer free \(W_{ij}K_{ij}\) secular toys; optional later: Laplace adjacent-coeff check + AMD stability.",
        "3. Query more with: `python scripts/run_campaign.py query --min-interest 10 --swap`.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "top_details.json").write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
    print(f"n_db={n} reported={len(filtered)} out={out}")
    print(f"report={out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
