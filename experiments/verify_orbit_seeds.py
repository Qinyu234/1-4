#!/usr/bin/env python3
"""Verify orbit seeds: accept gate (§3.2 + reject regular n-gon), then diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fairy_orbit.design.seeds import (
    CATALOGUE_PATH,
    SEEDS_DIR,
    load_catalogue,
    load_seed,
    regenerate_canonical_seeds,
    update_catalogue_verification,
    verify_free_shape_congruence,
    write_catalogue,
)
from fairy_orbit.engine.rebound_engine import ReboundConfig, integrate
from fairy_orbit.observe.choreography_verify import accept_seed_choreography

ROOT = Path(__file__).resolve().parents[1]


def frobenius(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    return float(np.sqrt(np.sum(X * X)))


def verify_seed_rebound(seed, *, n_outputs: int = 200, eps_tol: float = 1e-3) -> dict:
    sys0 = seed.to_system()
    r0 = np.stack([b.position.copy() for b in sys0.bodies])
    v0 = np.stack([b.velocity.copy() for b in sys0.bodies])
    traj = integrate(
        sys0,
        t_end=float(seed.period),
        n_outputs=n_outputs,
        config=ReboundConfig(
            stop_on_escape=False,
            stop_on_collision=False,
            epsilon=1e-9,
            min_dt=1e-8,
        ),
    )
    rT = traj.positions[-1]
    vT = traj.velocities[-1]
    denom_r = max(frobenius(r0), 1e-300)
    denom_v = max(frobenius(v0), 1e-300)
    eps_r = frobenius(rT - r0) / denom_r
    eps_v = frobenius(vT - v0) / denom_v
    require = seed.family.startswith("free_")
    ok = (eps_r < eps_tol and eps_v < eps_tol) if require else True
    return {
        "eps_r": eps_r,
        "eps_v": eps_v,
        "t_end": float(traj.times[-1]),
        "status": traj.status,
        "ok": ok,
        "require_close": require,
        "traj": traj,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Verify seeds with accept gate")
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--eps-tol", type=float, default=1e-3)
    p.add_argument("--atol-rel", type=float, default=1e-6)
    p.add_argument("--shift", type=int, default=1)
    p.add_argument("--n-outputs", type=int, default=200)
    p.add_argument("--skip-rebound", action="store_true")
    p.add_argument("--update-catalogue", action="store_true", default=True)
    p.add_argument("--no-update-catalogue", action="store_false", dest="update_catalogue")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "seed_verify" / "report.json",
    )
    args = p.parse_args()

    if args.regenerate or not CATALOGUE_PATH.exists():
        regenerate_canonical_seeds()
        print(f"regenerated seeds → {SEEDS_DIR}", flush=True)

    cat = load_catalogue()
    rows = []
    all_ok = True
    for entry in cat["seeds"]:
        seed = load_seed(SEEDS_DIR / entry["path"])
        gate = None
        shape = None
        reb = None
        traj = None
        acc = None

        if seed.family.startswith("free_"):
            acc = accept_seed_choreography(
                seed,
                shift=args.shift,
                atol_rel=args.atol_rel,
                n_outputs=min(64, args.n_outputs),
            )
            gate = acc.choreography
            gate_ok = bool(acc.ok)
            print(
                f"[{'OK' if gate_ok else 'FAIL'}-accept] {seed.id}: "
                f"reason={acc.reason} Er_rel={gate.E_r_rel:.3e} "
                f"maintains_regular_ngon={acc.maintains_regular_ngon}",
                flush=True,
            )
            if not gate_ok:
                all_ok = False
        else:
            gate_ok = True
            print(f"[SKIP] {seed.id}: hier baseline", flush=True)

        if not args.skip_rebound:
            reb = verify_seed_rebound(
                seed, n_outputs=args.n_outputs, eps_tol=args.eps_tol
            )
            traj = reb.pop("traj", None)
            print(
                f"[{'OK' if reb['ok'] else 'FAIL'}-period] {seed.id}: "
                f"eps_r={reb['eps_r']:.3e} eps_v={reb['eps_v']:.3e}",
                flush=True,
            )

        if seed.family.startswith("free_") and traj is not None:
            shape = verify_free_shape_congruence(seed, traj=traj, atol=args.atol_rel)

        row = {
            "id": seed.id,
            "family": seed.family,
            "period": seed.period,
            "choreography_Tn": None if gate is None else gate.to_dict(),
            "choreography_ok": gate_ok,
            "shape_diag": shape,
            "rebound": reb,
            "ok": gate_ok,
        }
        if acc is not None:
            row["accept_reason"] = acc.reason
            row["maintains_regular_ngon"] = acc.maintains_regular_ngon
        rows.append(row)

        if args.update_catalogue and gate is not None and acc is not None:
            update_catalogue_verification(
                cat,
                seed.id,
                gate=gate,
                orbit_class=(
                    "literature_choreography"
                    if acc.ok
                    else (
                        "rejected_maintained_regular_ngon"
                        if acc.maintains_regular_ngon
                        else "unverified"
                    )
                ),
            )
            for e in cat["seeds"]:
                if e.get("id") == seed.id:
                    e["accept_reason"] = acc.reason
                    e["maintains_regular_ngon"] = acc.maintains_regular_ngon
                    if acc.maintains_regular_ngon:
                        e["verified_claim"] = "rejected_maintained_regular_ngon"
                        e["choreography_ok"] = False
                    break

    if args.update_catalogue:
        write_catalogue(cat["seeds"], path=CATALOGUE_PATH)
        print(f"catalogue updated → {CATALOGUE_PATH}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows, "all_ok": all_ok}, indent=2), encoding="utf-8")
    print(f"report → {args.out}", flush=True)
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
