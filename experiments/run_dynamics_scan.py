#!/usr/bin/env python3
"""
Dynamics scan: find ladders with visible a-migration / encounters / role change.

Not a soak-for-MEGNO≈2 stability farm. Prefer PROMPT §3 secular exchange signals.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from fairy_orbit.core import SystemConfig
from fairy_orbit.design import LadderParams, build_orbital_ladder
from fairy_orbit.design.ladder import DEFAULT_PERIOD_RATIOS
from fairy_orbit.observe import diagnose
from fairy_orbit.store import DEFAULT_DB, OrbitStore
from fairy_orbit.viz import save_ladder_report
from fairy_orbit.viz.orbits import plot_orbit_gallery, plot_orbits_3d, plot_orbits_xy

ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def grid() -> list[tuple[float, float, tuple[float, float, float]]]:
    """(e, μ, period_ratios) — coupled regime + mild period-ratio detunes."""
    eccentricities = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]
    mass_ratios = [2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3]
    scales = [0.94, 0.97, 1.00, 1.03, 1.06]
    cases: list[tuple[float, float, tuple[float, float, float]]] = []
    for e in eccentricities:
        for mu in mass_ratios:
            for s in scales:
                ratios = tuple(float(r * s) for r in DEFAULT_PERIOD_RATIOS)
                cases.append((e, mu, ratios))  # type: ignore[arg-type]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan for dynamically interesting ladders")
    parser.add_argument("--t-end", type=float, default=800.0)
    parser.add_argument("--n-outputs", type=int, default=500)
    parser.add_argument("--top", type=int, default=6, help="how many interesting cases to plot")
    parser.add_argument("--megno", action="store_true", help="also compute MEGNO (slower)")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "dynamics",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT / DEFAULT_DB,
        help="SQLite orbit store path",
    )
    parser.add_argument("--no-store-traj", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.t_end = 80.0
        args.n_outputs = 100
        args.top = 3

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "runs.jsonl"
    if jsonl.exists():
        jsonl.unlink()

    cases = grid()
    if args.smoke:
        cases = cases[:8]

    rows: list[dict] = []
    log_path = out / "scan.log"
    store = OrbitStore(args.db)

    def log(msg: str) -> None:
        line = f"[{_now()}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"Dynamics scan start n_cases={len(cases)} t_end={args.t_end} db={args.db}")
    try:
        for e, mu, ratios in cases:
            cfg = SystemConfig(mass_ratio=mu)
            params = LadderParams(eccentricity=e, period_ratios=ratios, tetrahedral=True)
            system = build_orbital_ladder(cfg, params)
            t0 = time.perf_counter()
            d = diagnose(
                system,
                cfg,
                t_end=args.t_end,
                n_outputs=args.n_outputs,
                ladder=params,
                run_megno=args.megno,
            )
            elapsed = time.perf_counter() - t0
            run_id = store.save(
                config=cfg,
                params=params,
                diagnosis=d,
                source="dynamics",
                t_end=args.t_end,
                n_outputs=args.n_outputs,
                store_trajectory=not args.no_store_traj,
                elapsed_s=elapsed,
            )
            row = {
                "ts": _now(),
                "run_id": run_id,
                "eccentricity": e,
                "mass_ratio": mu,
                "period_ratios": list(ratios),
                "elapsed_s": elapsed,
                **d.summary,
            }
            with jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
            log(
                f"id={run_id} e={e:.2f} μ={mu:.0e} status={row['status']} "
                f"interest={row.get('interest'):.2f} a_rms={row.get('a_delta_rms'):.4f} "
                f"enc={row.get('n_encounters')} swap={row.get('a_order_changed')} "
                f"elapsed={elapsed:.1f}s"
            )
    finally:
        store.close()

    ranked = sorted(
        [r for r in rows if r.get("status") == "success"],
        key=lambda r: r.get("interest", -1e9),
        reverse=True,
    )
    lines = [
        "# Dynamics ranking (higher interest = more secular change)",
        "",
        f"Generated: {_now()}",
        "",
        "| rank | e | μ | interest | a_Δrms | a_ptp | enc | swap | MEGNO |",
        "|------|---|---|----------|--------|-------|-----|------|-------|",
    ]
    for i, r in enumerate(ranked[:30], start=1):
        megno = r.get("megno")
        megno_s = f"{megno:.2f}" if isinstance(megno, (int, float)) and megno is not None else "—"
        lines.append(
            f"| {i} | {r['eccentricity']:.2f} | {r['mass_ratio']:.0e} | "
            f"{r.get('interest', float('nan')):.2f} | {r.get('a_delta_rms', 0):.4f} | "
            f"{r.get('a_ptp_mean', 0):.4f} | {r.get('n_encounters', 0)} | "
            f"{r.get('a_order_changed')} | {megno_s} |"
        )
    (out / "RANKING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps({"n": len(rows), "top": ranked[:15]}, indent=2, default=str),
        encoding="utf-8",
    )

    # Re-integrate top cases for orbit gallery (longer if not smoke)
    diagnoses = []
    viz_dir = out / "orbits"
    viz_dir.mkdir(exist_ok=True)
    for r in ranked[: args.top]:
        cfg = SystemConfig(mass_ratio=r["mass_ratio"])
        params = LadderParams(
            eccentricity=r["eccentricity"],
            period_ratios=tuple(r["period_ratios"]),  # type: ignore[arg-type]
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
        label = f"e={r['eccentricity']:.2f}, μ={r['mass_ratio']:.0e}, I={r.get('interest', 0):.1f}"
        diagnoses.append((label, d))
        safe = f"e{r['eccentricity']:.2f}_mu{r['mass_ratio']:.0e}".replace("+", "")
        case_dir = viz_dir / safe
        case_dir.mkdir(exist_ok=True)
        plot_orbits_xy(d.trajectory, case_dir / "xy.png", title=label, encounters=d.encounters)
        plot_orbits_3d(d.trajectory, case_dir / "xyz.png", title=label)
        save_ladder_report(d, case_dir / "report")

    if diagnoses:
        plot_orbit_gallery(diagnoses, out / "orbit_gallery.png", cols=3)
    log(f"Done. top_interest={ranked[0].get('interest') if ranked else None}")
    log(f"See {out / 'RANKING.md'} and {out / 'orbit_gallery.png'}")


if __name__ == "__main__":
    main()
