#!/usr/bin/env python3
"""
Re-integrate top candidates at higher REBOUND precision (smaller IAS15 timestep).

Verification, not search: does the interesting behaviour (a-order swap, encounters,
migration) survive when the adaptive timestep is tightened? If interest / swap / a(t)
persist and energy drift drops, the candidate is physical, not a numerical artifact.

See docs/DIRECTION.md (REBOUND = truth).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.core import SystemConfig
from fairy_orbit.design import LadderParams, build_orbital_ladder
from fairy_orbit.engine import ReboundConfig
from fairy_orbit.observe import diagnose, extract_amd_series
from fairy_orbit.store import DEFAULT_DB, OrbitStore
from fairy_orbit.viz.orbits import plot_orbits_xy

ROOT = Path(__file__).resolve().parents[1]


def _select_records(store: OrbitStore, args) -> list:
    if args.ids:
        return [r for r in (store.get(i) for i in args.ids) if r is not None]
    hits = store.query(
        min_interest=args.min_interest,
        a_order_changed=True if args.swap else None,
        order_by="interest",
        limit=args.top * 3,
    )
    good = [
        r
        for r in hits
        if r.status == "success" and r.a_delta_rms is not None and r.a_delta_rms < 5.0
    ]
    return good[: args.top]


def _run(rec, epsilon: float | None, t_end: float, n_outputs: int):
    cfg = SystemConfig(mass_ratio=rec.mass_ratio)
    params = LadderParams(
        eccentricity=rec.eccentricity,
        a_inner=rec.a_inner,
        period_ratios=tuple(rec.period_ratios),  # type: ignore[arg-type]
        tetrahedral=rec.tetrahedral,
    )
    system = build_orbital_ladder(cfg, params)
    rc = ReboundConfig(epsilon=epsilon)
    d = diagnose(
        system,
        cfg,
        t_end=t_end,
        n_outputs=n_outputs,
        ladder=params,
        run_megno=False,
        rebound_config=rc,
    )
    return cfg, params, d


def main() -> None:
    p = argparse.ArgumentParser(description="High-precision REBOUND recompute of top candidates")
    p.add_argument("--db", type=Path, default=ROOT / DEFAULT_DB)
    p.add_argument("--ids", type=int, nargs="*", help="explicit run ids (else pick top)")
    p.add_argument("--top", type=int, default=6)
    p.add_argument("--min-interest", type=float, default=15.0)
    p.add_argument("--swap", action="store_true", default=True)
    p.add_argument("--epsilon", type=float, default=1e-11, help="IAS15 accuracy (smaller=finer dt)")
    p.add_argument("--baseline-epsilon", type=float, default=1e-9, help="reference IAS15 accuracy")
    p.add_argument("--t-end", type=float, default=800.0)
    p.add_argument("--n-outputs", type=int, default=1000)
    p.add_argument(
        "--out", type=Path, default=ROOT / "experiments" / "output" / "dynamics" / "refine"
    )
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    with OrbitStore(args.db) as store:
        records = _select_records(store, args)

    if not records:
        print("no candidates selected")
        return

    lines = [
        "# High-precision recompute of top candidates",
        "",
        f"Baseline IAS15 ε = {args.baseline_epsilon:g}  →  refined ε = {args.epsilon:g}  "
        f"(t_end={args.t_end:g}, n_outputs={args.n_outputs}).",
        "",
        "Physical iff swap/encounters/interest persist and energy drift drops.",
        "",
        "| id | e | μ | I(base) | I(fine) | swap b→f | enc b→f | a_rms b→f | Edrift b→f | verdict |",
        "|----|---|---|---------|---------|----------|---------|-----------|------------|---------|",
    ]
    details = []

    for rec in records:
        _, _, d_base = _run(rec, args.baseline_epsilon, args.t_end, args.n_outputs)
        cfg, _, d_fine = _run(rec, args.epsilon, args.t_end, args.n_outputs)
        sb, sf = d_base.summary, d_fine.summary

        swap_ok = bool(sb.get("a_order_changed")) == bool(sf.get("a_order_changed"))
        drift_ok = (sf.get("energy_drift") or 1.0) <= (sb.get("energy_drift") or 1.0) * 2.0
        status_ok = sf.get("status") == "success"
        verdict = "physical" if (swap_ok and status_ok and drift_ok) else "review"

        label = f"id{rec.id} e={rec.eccentricity:.2f} μ={rec.mass_ratio:.0e} ε={args.epsilon:g}"
        case = out / f"run_{rec.id:04d}"
        case.mkdir(exist_ok=True)
        plot_orbits_xy(d_fine.trajectory, case / "xy_fine.png", title=label, encounters=d_fine.encounters)

        el = d_fine.elements
        masses = np.full(el.a.shape[1], cfg.fairy_mass)
        amd = extract_amd_series(el, masses, cfg.mu)
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for j, lab in enumerate(el.labels):
            ax[0].plot(el.times, el.a[:, j], label=lab)
            ax[1].plot(amd.times, amd.amd[:, j], label=lab)
        ax[1].plot(amd.times, amd.amd_total, "k--", lw=1.2, label="total")
        ax[0].set_ylabel("a")
        ax[0].set_title(f"{label} — semi-major axes (fine)")
        ax[0].legend(fontsize=7)
        ax[1].set_ylabel("AMD")
        ax[1].set_xlabel("t")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(case / "a_amd_fine.png", dpi=140)
        plt.close(fig)

        lines.append(
            f"| {rec.id} | {rec.eccentricity:.2f} | {rec.mass_ratio:.0e} | "
            f"{sb.get('interest'):.2f} | {sf.get('interest'):.2f} | "
            f"{sb.get('a_order_changed')}→{sf.get('a_order_changed')} | "
            f"{sb.get('n_encounters')}→{sf.get('n_encounters')} | "
            f"{sb.get('a_delta_rms'):.3f}→{sf.get('a_delta_rms'):.3f} | "
            f"{sb.get('energy_drift'):.1e}→{sf.get('energy_drift'):.1e} | {verdict} |"
        )
        details.append(
            {
                "run_id": rec.id,
                "param_class": rec.param_class,
                "verdict": verdict,
                "baseline": sb,
                "fine": sf,
            }
        )
        print(f"id={rec.id} verdict={verdict}")

    (out / "REFINE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "refine_details.json").write_text(
        json.dumps(details, indent=2, default=str), encoding="utf-8"
    )
    print(f"refined={len(records)} out={out}")
    print(f"report={out / 'REFINE.md'}")


if __name__ == "__main__":
    main()
