#!/usr/bin/env python3
"""Launch PROMPT mainline jobs (self-expanding; optional wall clock limit).

Thread budget 7:2:6:1 = choreo_n4 : choreo_n5 : path_a : branch2.
Default activation: **n4 + Path A** (n5 dark). Opt-in: ``--with-n5``.
Path A via ``--seed-n4`` / ``--path-a-from-db``.

Finite ``--wall-hours`` is split across *launched* slots by those weights.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from fairy_orbit.observe.campaign_prefs import (
    BRANCH2_IN_DEFAULT_CAMPAIGN,
    CHOREO_N5_IN_DEFAULT_CAMPAIGN,
    FLOQUET_GATE_DEFAULT,
    PATH_A_IN_DEFAULT_CAMPAIGN,
    campaign_priority_blurb,
    pick_path_a_seed,
    wall_hours_for_slots,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(
        description="PROMPT campaign launcher (threads 7:2:6:1; n5 off by default)"
    )
    p.add_argument(
        "--wall-hours",
        type=float,
        default=0.0,
        help="total wall hours split by 7:2:6:1 among launched slots; <=0 unlimited",
    )
    p.add_argument(
        "--seed-n4",
        type=Path,
        default=None,
        help="accepted free-4 choreography JSON for Path A continuation",
    )
    p.add_argument(
        "--path-a-from-db",
        action="store_true",
        help="pick Path-A seed from N=4 SQLite (prefer Floquet-stable; else best)",
    )
    p.add_argument(
        "--seed-n5",
        type=Path,
        default=None,
        help="accepted free-5 choreography JSON for Path B (implies --with-n5 path-B)",
    )
    p.add_argument(
        "--with-n5",
        action="store_true",
        help="opt-in choreography search N=5 (weight 2; off by default)",
    )
    p.add_argument(
        "--no-path-a",
        action="store_true",
        help="do not launch Path A even if seed / --path-a-from-db is set",
    )
    p.add_argument(
        "--no-floquet-gate",
        action="store_true",
        help="disable Floquet certify on search jobs (not recommended)",
    )
    p.add_argument("--fresh", action="store_true", help="wipe prior search/continuation dirs first")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--wait", action="store_true", help="block until all jobs finish")
    args = p.parse_args()

    print(campaign_priority_blurb(), flush=True)
    assert BRANCH2_IN_DEFAULT_CAMPAIGN is False
    assert CHOREO_N5_IN_DEFAULT_CAMPAIGN is False

    py = sys.executable
    out_root = ROOT / "experiments" / "output"
    log_dir = out_root / "prompt_campaign_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    seed_n4 = args.seed_n4
    if args.path_a_from_db and seed_n4 is None:
        picked_path = out_root / "continuation_n4" / "path_a_seed.json"
        pick = pick_path_a_seed(4, out_seed_path=picked_path)
        if pick is None:
            print("WARNING: --path-a-from-db but no accepted N=4 passes", flush=True)
        else:
            seed_n4 = picked_path
            print(
                f"Path-A seed trial={pick.trial_no} residual={pick.residual} "
                f"floquet_stable={pick.floquet_stable} — {pick.note}",
                flush=True,
            )

    want_path_a = (
        PATH_A_IN_DEFAULT_CAMPAIGN
        and not args.no_path_a
        and seed_n4 is not None
    )
    want_n5 = bool(args.with_n5) or CHOREO_N5_IN_DEFAULT_CAMPAIGN

    active_slots: list[str] = ["choreo_n4"]
    if want_n5:
        active_slots.append("choreo_n5")
    if want_path_a:
        active_slots.append("path_a")

    walls = wall_hours_for_slots(args.wall_hours, active_slots)
    print(f"active slots={active_slots} wall_split={walls}", flush=True)

    if args.fresh:
        wipe = ["choreography_search_n4", "continuation_n4"]
        if want_n5 or args.seed_n5 is not None:
            wipe.extend(["choreography_search_n5", "continuation_n5"])
        for name in wipe:
            d = out_root / name
            if d.is_dir():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()

    floquet_args: list[str] = []
    if args.no_floquet_gate or not FLOQUET_GATE_DEFAULT:
        floquet_args = ["--no-floquet-gate"]

    def _wall_arg(slot: str) -> list[str]:
        w = walls.get(slot, 0.0)
        return ["--wall-hours", str(w)]

    jobs: list[tuple[str, list[str]]] = [
        (
            "choreo_n4",
            [
                py,
                str(ROOT / "experiments" / "run_choreography_search.py"),
                "--n",
                "4",
                *_wall_arg("choreo_n4"),
                *floquet_args,
            ],
        ),
    ]
    if args.fresh:
        jobs[0][1].append("--fresh")

    if want_n5:
        n5_cmd = [
            py,
            str(ROOT / "experiments" / "run_choreography_search.py"),
            "--n",
            "5",
            *_wall_arg("choreo_n5"),
            *floquet_args,
        ]
        if args.fresh:
            n5_cmd.append("--fresh")
        jobs.append(("choreo_n5", n5_cmd))

    if want_path_a:
        assert seed_n4 is not None
        jobs.append(
            (
                "cont_n4",
                [
                    py,
                    str(ROOT / "experiments" / "run_mass_continuation_campaign.py"),
                    "--n",
                    "4",
                    "--seed",
                    str(seed_n4),
                    *_wall_arg("path_a"),
                ],
            )
        )

    if args.seed_n5 is not None:
        # Path B is outside the 7:2:6:1 metaphor; reuse n5 wall share if present else full.
        pb_wall = walls.get("choreo_n5", args.wall_hours)
        jobs.append(
            (
                "cont_n5",
                [
                    py,
                    str(ROOT / "experiments" / "run_mass_continuation_campaign.py"),
                    "--n",
                    "5",
                    "--seed",
                    str(args.seed_n5),
                    "--wall-hours",
                    str(pb_wall),
                ],
            )
        )

    procs: list[tuple[str, subprocess.Popen]] = []
    for name, cmd in jobs:
        log = log_dir / f"{name}.log"
        print(f"START {name}: {' '.join(cmd)} → {log}", flush=True)
        if args.dry_run:
            continue
        f = open(log, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((name, proc))
        print(f"  pid={proc.pid} {name}", flush=True)

    if args.dry_run:
        return

    wall_label = "unlimited" if args.wall_hours <= 0 else f"{args.wall_hours}h total"
    print(f"launched {len(procs)} jobs; wall={wall_label}; logs={log_dir}", flush=True)
    if args.wait:
        for name, proc in procs:
            code = proc.wait()
            print(f"DONE {name} exit={code}", flush=True)
    else:
        print(
            "Parent exiting; children keep running. "
            "Monitor trials.jsonl / summary.json under experiments/output/.",
            flush=True,
        )


if __name__ == "__main__":
    main()
