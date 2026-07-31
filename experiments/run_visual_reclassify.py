#!/usr/bin/env python3
"""Visual reclassify of *post-continuation* orbits only (PROMPT §4.1).

Free-N search archives are skipped. Sources:

* ``experiments/output/continuation_n4/state_Mc_*.json`` (+ final.json)
* ``experiments/output/continuation_n5/state_mu_*.json`` (+ final.json)

If no checkpoints exist, exits with a clear message (run Path A/B first).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from fairy_orbit.design.seeds import load_seed
from fairy_orbit.observe.optical_encounter import DEFAULT_LOG_RHO
from fairy_orbit.observe.visual_classify import classify_continued_orbit

ROOT = Path(__file__).resolve().parents[1]
_MC_RE = re.compile(r"state_Mc_([0-9.eE+-]+)\.json$")
_MU_RE = re.compile(r"state_mu_([0-9.eE+-]+)\.json$")


def _collect_n4(cont: Path) -> list[tuple[Path, float]]:
    rows: list[tuple[Path, float]] = []
    for path in sorted(cont.glob("state_Mc_*.json")):
        m = _MC_RE.search(path.name)
        if m:
            rows.append((path, float(m.group(1))))
    final = cont / "final.json"
    summary = cont / "summary.json"
    if final.exists() and rows:
        Mc = rows[-1][1]
        if summary.exists():
            try:
                Mc = float(json.loads(summary.read_text(encoding="utf-8")).get("M_c_final", Mc))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        # avoid duplicate if final == last state
        if not any(p.resolve() == final.resolve() for p, _ in rows):
            rows.append((final, Mc))
    return rows


def _collect_n5(cont: Path) -> list[tuple[Path, float]]:
    """Path B: peripheral mu; treat central role as body0 mass=1, optics still on C."""
    rows: list[tuple[Path, float]] = []
    for path in sorted(cont.glob("state_mu_*.json")):
        m = _MU_RE.search(path.name)
        if m:
            # store mu in the M_c field slot as documentation; classify uses M_c=1
            rows.append((path, float(m.group(1))))
    final = cont / "final.json"
    if final.exists() and rows:
        if not any(p.resolve() == final.resolve() for p, _ in rows):
            rows.append((final, rows[-1][1]))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Visual reclassify post-continuation orbits (not free-N)"
    )
    p.add_argument("--n", type=int, nargs="+", default=[4, 5], choices=[4, 5])
    p.add_argument("--log-rho", type=float, default=DEFAULT_LOG_RHO)
    p.add_argument("--n-outputs", type=int, default=40)
    p.add_argument("--encounter-scale", type=float, default=0.35)
    p.add_argument(
        "--cont-dir",
        type=Path,
        default=None,
        help="override continuation dir (default experiments/output/continuation_n{N})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "experiments/output/profile",
    )
    args = p.parse_args()
    if not (-1.0 <= float(args.log_rho) <= 1.0):
        raise SystemExit("--log-rho must be in [-1, 1]")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    any_work = False
    for n in args.n:
        cont = args.cont_dir or (ROOT / f"experiments/output/continuation_n{n}")
        if args.cont_dir is not None and len(args.n) > 1:
            # single override dir only makes sense for one n
            cont = args.cont_dir
        if n == 4:
            jobs = _collect_n4(cont) if cont.is_dir() else []
        else:
            jobs = _collect_n5(cont) if cont.is_dir() else []

        if not jobs:
            print(
                f"n={n}: no continuation checkpoints under {cont} "
                f"(need state_Mc_*.json / state_mu_*.json). Skipping free-N.",
                flush=True,
            )
            continue

        any_work = True
        reports = []
        print(f"n={n} classifying {len(jobs)} continuation checkpoints …", flush=True)
        for path, param in jobs:
            seed = load_seed(path)
            if n == 4:
                Mc = float(param)
            else:
                # Path B: body 0 is the heavy "central" role at mass 1
                Mc = 1.0
            rep = classify_continued_orbit(
                seed,
                Mc,
                log_rho=args.log_rho,
                n_outputs=args.n_outputs,
                encounter_scale=args.encounter_scale,
                source_path=str(path),
            )
            if n == 5:
                d = rep.to_dict()
                d["mu"] = float(param)
                reports.append(d)
            else:
                reports.append(rep.to_dict())
            print(
                f"  {path.name} M_c/param={param:.4g} → {rep.klass} "
                f"enc={rep.n_encounters} swap={rep.n_light_swap}",
                flush=True,
            )

        counts = Counter(r["klass"] for r in reports)
        by_class: dict[str, list] = {}
        for r in reports:
            by_class.setdefault(r["klass"], []).append(r)

        payload = {
            "scope": "post_continuation_only",
            "n": n,
            "log_rho": args.log_rho,
            "encounter_scale": args.encounter_scale,
            "n_checkpoints": len(reports),
            "counts": dict(counts),
            "by_class": by_class,
            "rows": reports,
        }
        out = args.out_dir / f"visual_reclassify_continuation_n{n}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"n={n} counts={dict(counts)} → {out}", flush=True)

    if not any_work:
        raise SystemExit(
            "No post-continuation orbits found. Run Path A/B first, e.g.\n"
            "  python experiments/run_mass_continuation_campaign.py --n 4 "
            "--seed … --wall-hours 0.25"
        )


if __name__ == "__main__":
    main()
