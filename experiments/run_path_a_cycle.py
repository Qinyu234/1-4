#!/usr/bin/env python3
"""Path A over accepted N=4 seeds: perpetual cycle, or multi-period stability scan.

Scan mode (--scan-top-k): take the best free-N seeds, reuse existing Path A
checkpoints when present, LM-correct with --horizon-periods residual, and
rank (trial, M_c) that stay Floquet-stable with small long-horizon ||F||.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from fairy_orbit.design.seeds import OrbitSeed, load_seed, save_seed
from fairy_orbit.observe.campaign_prefs import (
    CERTIFY_MAX_RESIDUAL,
    FLOQUET_STABLE_ATOL,
    _floquet_meta,
)
from fairy_orbit.observe.continuation import (
    DEFAULT_PATH_A_HORIZON_PERIODS,
    DEFAULT_PATH_A_MAX_NFEV,
    attach_central_mass,
    correct_at_mass,
    symmetry_residual_vector,
)
from fairy_orbit.observe.stability import floquet_multipliers_fd
from fairy_orbit.store.search_db import ChoreographySearchStore, DEFAULT_SEARCH_DB_NAME

ROOT = Path(__file__).resolve().parents[1]


def _load_candidates(
    db: Path, *, limit: int = 200
) -> list[tuple[int, OrbitSeed, bool | None, float | None, float | None]]:
    """(trial_no, seed, floquet_stable, max_abs, residual) — Floquet-stable, low residual first."""
    rows: list[tuple[int, OrbitSeed, bool | None, float | None, float | None]] = []
    with ChoreographySearchStore(db) as store:
        passes = store.list_passes(4, limit=limit, max_residual=CERTIFY_MAX_RESIDUAL)
    for rec in passes:
        if not rec.seed_json:
            continue
        seed = OrbitSeed.from_dict(rec.seed_json)
        st, mx = _floquet_meta(seed)
        rows.append((int(rec.trial_no), seed, st, mx, rec.residual))
    rows.sort(
        key=lambda r: (
            0 if r[2] is True else 1,
            float(r[4]) if r[4] is not None else 1e300,
        )
    )
    return rows


def _horizon_residual(seed: OrbitSeed, M_c: float, *, horizon_periods: float) -> float:
    sys = attach_central_mass(seed, float(M_c))
    f = symmetry_residual_vector(
        sys,
        seed,
        seed.period,
        optics_soft=False,
        horizon_periods=float(horizon_periods),
    )
    return float(np.linalg.norm(f))


def _eval_checkpoint(
    seed: OrbitSeed,
    M_c: float,
    *,
    horizon_periods: float,
    polish: bool,
    max_nfev: int,
    stable_atol: float,
    skip_floquet: bool = False,
    archive_stable: bool | None = None,
) -> dict:
    res0 = _horizon_residual(seed, M_c, horizon_periods=horizon_periods)
    out_seed = seed
    res1 = res0
    ls_ok = None
    if polish:
        out_seed, res1, ls_ok = correct_at_mass(
            seed,
            float(M_c),
            max_nfev=max_nfev,
            optics_soft=False,
            horizon_periods=float(horizon_periods),
        )
    if skip_floquet and not polish:
        fl = {
            "stable": archive_stable,
            "max_abs": None,
            "map_residual": None,
        }
    else:
        try:
            fl = floquet_multipliers_fd(
                out_seed, shift=1, stable_atol=stable_atol
            ).to_dict()
        except Exception as exc:  # noqa: BLE001
            fl = {
                "error": str(exc),
                "stable": False,
                "max_abs": None,
                "map_residual": None,
            }
    return {
        "M_c": float(M_c),
        "horizon_periods": float(horizon_periods),
        "residual_before": res0,
        "residual_after": float(res1),
        "ls_success": ls_ok,
        "floquet_stable": fl.get("stable"),
        "floquet_max_abs": fl.get("max_abs"),
        "map_residual": fl.get("map_residual"),
        "seed": out_seed,
    }


def _mc_grid(m_c_max: float, n: int = 8) -> list[float]:
    hi = max(min(float(m_c_max), 0.1), 1e-3)
    return [float(x) for x in np.geomspace(1e-3, hi, num=max(n, 2))]


def _archive_max_stable_mc(out_root: Path, trial_no: int) -> float:
    sweep = out_root / f"trial_{trial_no}" / "floquet_path_sweep.json"
    if not sweep.is_file():
        return -1.0
    try:
        payload = json.loads(sweep.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return -1.0
    best = -1.0
    for r in payload.get("rows") or []:
        if r.get("stable"):
            best = max(best, float(r["M_c"]))
    return best


def run_scan(args: argparse.Namespace) -> Path:
    db = ROOT / "experiments/output/choreography_search_n4" / DEFAULT_SEARCH_DB_NAME
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "experiments/output/profile"
    report_dir.mkdir(parents=True, exist_ok=True)

    cands = _load_candidates(db, limit=max(500, args.scan_top_k * 20))
    # Rank: Path A archive with high Floquet-stable Mc first, then residual.
    ranked = []
    for row in cands:
        trial_no = row[0]
        max_mc = _archive_max_stable_mc(out_root, trial_no)
        ranked.append((max_mc, row))
    ranked.sort(
        key=lambda t: (
            0 if t[0] >= 0 else 1,
            -t[0],
            float(t[1][4]) if t[1][4] is not None else 1e300,
        )
    )
    top = [r for _, r in ranked[: int(args.scan_top_k)]]

    print(
        f"scan top-{len(top)} seeds (max Floquet-stable Mc first) "
        f"horizon={args.horizon_periods}P polish={args.polish} res_tol={args.res_tol}",
        flush=True,
    )

    rows: list[dict] = []
    per_trial = max(1, int(args.per_trial))
    for trial_no, seed0, st0, mx0, res0 in top:
        trial_dir = out_root / f"trial_{trial_no}"
        max_mc = _archive_max_stable_mc(out_root, trial_no)
        print(
            f"[scan] trial={trial_no} free_floq={st0} max|λ|={mx0} "
            f"res={res0} archive_max_stable_Mc={max_mc}",
            flush=True,
        )
        checkpoints: list[tuple[Path | None, float, OrbitSeed, bool | None]] = []
        sweep = trial_dir / "floquet_path_sweep.json"
        if sweep.is_file():
            try:
                payload = json.loads(sweep.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            stable_rows = []
            for r in payload.get("rows") or []:
                if args.stable_checkpoints_only and not r.get("stable"):
                    continue
                path = Path(r["path"]) if r.get("path") else None
                Mc = float(r["M_c"])
                if Mc > float(args.m_c_max) + 1e-15:
                    continue
                if path is not None and path.is_file():
                    stable_rows.append((path, Mc, load_seed(path), bool(r.get("stable"))))
            stable_rows.sort(key=lambda x: -x[1])
            checkpoints = stable_rows[:per_trial]
            if not checkpoints:
                print(f"  no usable checkpoints under {trial_dir}", flush=True)
        if not checkpoints:
            if not args.allow_mc_grid:
                print("  skip (no archive; pass --allow-mc-grid to sample Mc)", flush=True)
                continue
            for Mc in _mc_grid(float(args.m_c_max), n=int(args.mc_grid)):
                checkpoints.append((None, Mc, seed0, None))

        for path, Mc, seed, arch_stable in checkpoints:
            res_pre = _horizon_residual(
                seed, Mc, horizon_periods=float(args.horizon_periods)
            )
            do_polish = bool(args.polish) and (
                res_pre < float(args.res_tol) * float(args.polish_gate)
            )
            if bool(args.polish) and not do_polish:
                print(
                    f"  skip-polish M_c={Mc:.5g} ||F||_pre={res_pre:.3e} "
                    f"arch_floq={arch_stable}",
                    flush=True,
                )
                # Still record pre-screen using archive Floquet when available.
                row = {
                    "trial_no": trial_no,
                    "source_checkpoint": None if path is None else str(path),
                    "free_residual": res0,
                    "free_floquet_stable": st0,
                    "polished": False,
                    "M_c": float(Mc),
                    "horizon_periods": float(args.horizon_periods),
                    "residual_before": res_pre,
                    "residual_after": res_pre,
                    "ls_success": None,
                    "floquet_stable": arch_stable,
                    "floquet_max_abs": None,
                    "map_residual": None,
                    "ok_multi": bool(
                        arch_stable and res_pre < float(args.res_tol)
                    ),
                }
                rows.append(row)
                if row["ok_multi"]:
                    print(
                        f"  HIT(pre) M_c={Mc:.5g} ||F||={res_pre:.3e} arch_floq=True",
                        flush=True,
                    )
                continue

            ev = _eval_checkpoint(
                seed,
                Mc,
                horizon_periods=float(args.horizon_periods),
                polish=do_polish,
                max_nfev=int(args.max_nfev),
                stable_atol=float(args.floquet_stable_atol),
                skip_floquet=float(res_pre) >= float(args.res_tol) and not do_polish,
                archive_stable=arch_stable,
            )
            polished = ev.pop("seed")
            row = {
                "trial_no": trial_no,
                "source_checkpoint": None if path is None else str(path),
                "free_residual": res0,
                "free_floquet_stable": st0,
                "polished": do_polish,
                **ev,
                "ok_multi": bool(
                    ev.get("floquet_stable")
                    and float(ev["residual_after"]) < float(args.res_tol)
                ),
            }
            rows.append(row)
            tag = (
                int(args.horizon_periods)
                if abs(args.horizon_periods - round(args.horizon_periods)) < 1e-12
                else args.horizon_periods
            )
            if row["ok_multi"]:
                dest = trial_dir / f"state_horizon{tag}_Mc_{Mc:.6e}.json"
                trial_dir.mkdir(parents=True, exist_ok=True)
                save_seed(polished, dest)
                row["saved"] = str(dest)
                print(
                    f"  HIT M_c={Mc:.5g} ||F||={ev['residual_after']:.3e} "
                    f"max|λ|={ev['floquet_max_abs']}",
                    flush=True,
                )
            else:
                print(
                    f"  miss M_c={Mc:.5g} ||F||={ev['residual_after']:.3e} "
                    f"floq={ev['floquet_stable']} max|λ|={ev['floquet_max_abs']}",
                    flush=True,
                )

    hits = [r for r in rows if r.get("ok_multi")]
    hits.sort(key=lambda r: (-float(r["M_c"]), float(r["residual_after"])))
    rows.sort(
        key=lambda r: (
            0 if r.get("ok_multi") else 1,
            0 if r.get("floquet_stable") else 1,
            float(r["residual_after"]),
            -float(r["M_c"]),
        )
    )
    tag = (
        int(args.horizon_periods)
        if abs(args.horizon_periods - round(args.horizon_periods)) < 1e-12
        else args.horizon_periods
    )
    out_path = report_dir / f"multipperiod_stable_scan_{tag}P.json"
    payload = {
        "horizon_periods": float(args.horizon_periods),
        "res_tol": float(args.res_tol),
        "scan_top_k": int(args.scan_top_k),
        "polish": bool(args.polish),
        "n_eval": len(rows),
        "n_hits": len(hits),
        "hits": hits,
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"DONE hits={len(hits)}/{len(rows)} → {out_path}",
        flush=True,
    )
    if hits:
        h0 = hits[0]
        print(
            f"best hit: trial={h0['trial_no']} M_c={h0['M_c']:.6g} "
            f"||F||={h0['residual_after']:.3e} max|λ|={h0['floquet_max_abs']}",
            flush=True,
        )
    return out_path


def run_perpetual(args: argparse.Namespace) -> None:
    py = sys.executable
    script = ROOT / "experiments/run_mass_continuation_campaign.py"
    db = ROOT / "experiments/output/choreography_search_n4" / DEFAULT_SEARCH_DB_NAME
    args.out_root.mkdir(parents=True, exist_ok=True)

    seen: set[int] = set()
    round_i = 0
    while True:
        cands = _load_candidates(db)
        unused = [c for c in cands if c[0] not in seen]
        if not unused:
            print("seed pool exhausted; clearing seen and reshuffling", flush=True)
            seen.clear()
            time.sleep(args.sleep_s)
            continue
        trial_no, seed, st, mx, _res = unused[0]
        seen.add(trial_no)
        round_i += 1
        out = args.out_root / f"trial_{trial_no}"
        out.mkdir(parents=True, exist_ok=True)
        seed_path = out / "seed.json"
        save_seed(seed, seed_path)
        print(
            f"[cycle {round_i}] trial={trial_no} floquet_stable={st} "
            f"max_abs={mx} → {out}",
            flush=True,
        )
        cmd = [
            py,
            str(script),
            "--n",
            "4",
            "--seed",
            str(seed_path),
            "--wall-hours",
            "0",
            "--no-optics-soft",
            "--res-tol",
            str(args.res_tol),
            "--m-c-max",
            str(args.m_c_max),
            "--max-nfev",
            str(args.max_nfev),
            "--horizon-periods",
            str(args.horizon_periods),
            "--out",
            str(out),
            "--floquet-sweep",
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT))
        summary: dict = {}
        sp = out / "summary.json"
        if sp.is_file():
            try:
                summary = json.loads(sp.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                summary = {}
        print(
            f"[cycle {round_i}] exit={proc.returncode} "
            f"M_c_final={summary.get('M_c_final')} steps={summary.get('steps')}",
            flush=True,
        )
        time.sleep(args.sleep_s)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Path A cycle / multi-period stability scan on top seeds"
    )
    p.add_argument("--res-tol", type=float, default=1e-3)
    p.add_argument("--m-c-max", type=float, default=1.0)
    p.add_argument("--max-nfev", type=int, default=DEFAULT_PATH_A_MAX_NFEV)
    p.add_argument(
        "--horizon-periods",
        type=float,
        default=DEFAULT_PATH_A_HORIZON_PERIODS,
        help=f"residual after N orbital periods (default {DEFAULT_PATH_A_HORIZON_PERIODS})",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "experiments/output/continuation_n4_cycle",
    )
    p.add_argument("--sleep-s", type=float, default=2.0)
    p.add_argument(
        "--scan-top-k",
        type=int,
        default=0,
        help="if >0: finite scan of top-k seeds for multi-period+Floquet-stable Mc "
        "(no perpetual loop)",
    )
    p.add_argument(
        "--polish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="LM-correct each checkpoint under --horizon-periods (scan mode)",
    )
    p.add_argument(
        "--stable-checkpoints-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="in scan, only reuse Floquet-stable Path A checkpoints when present",
    )
    p.add_argument(
        "--mc-grid",
        type=int,
        default=8,
        help="geomspace Mc samples when a trial has no Path A archive",
    )
    p.add_argument(
        "--allow-mc-grid",
        action="store_true",
        help="if a top seed has no Path A archive, sample Mc from free-N (slow)",
    )
    p.add_argument(
        "--polish-gate",
        type=float,
        default=20.0,
        help="only LM-polish if pre-residual < res_tol * polish_gate",
    )
    p.add_argument(
        "--per-trial",
        type=int,
        default=3,
        help="max Path A checkpoints to test per seed (highest Mc first)",
    )
    p.add_argument(
        "--floquet-stable-atol",
        type=float,
        default=FLOQUET_STABLE_ATOL,
    )
    args = p.parse_args()

    if int(args.scan_top_k) > 0:
        run_scan(args)
    else:
        run_perpetual(args)


if __name__ == "__main__":
    main()
