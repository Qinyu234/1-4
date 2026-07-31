#!/usr/bin/env python3
"""Automatic PROMPT pipeline: search (SQL resume) + periodic shape-family plots.

Launches N=4/N=5 choreography searches (unlimited by default) and, on a timer,
re-selects diverse shape families from SQLite and regenerates PNG/HTML.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from fairy_orbit.observe.campaign_prefs import (
    CERTIFY_ATOL_REL,
    CERTIFY_MAX_RESIDUAL,
    CHOREO_N5_IN_DEFAULT_CAMPAIGN,
    FLOQUET_GATE_DEFAULT,
    campaign_priority_blurb,
)

ROOT = Path(__file__).resolve().parents[1]


def _py() -> str:
    return sys.executable


def _running_search_pids() -> list[tuple[int, str]]:
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
            timeout=90,
            cwd=str(ROOT),
        ).strip()
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return []
    if not raw:
        return []
    data = json.loads(raw)
    if isinstance(data, dict):
        data = [data]
    out: list[tuple[int, str]] = []
    for d in data:
        cmd = d.get("CommandLine") or ""
        if "run_choreography_search.py" not in cmd:
            continue
        if "Astrophysics" not in cmd:
            continue
        out.append((int(d["ProcessId"]), cmd))
    return out


def ensure_searches(
    *,
    wall_hours: float,
    fresh: bool,
    atol_rel: float = CERTIFY_ATOL_REL,
    max_residual: float = CERTIFY_MAX_RESIDUAL,
    floquet_gate: bool = FLOQUET_GATE_DEFAULT,
    ns: tuple[int, ...] = (4,),
) -> list[subprocess.Popen]:
    """Start missing search workers for ``ns`` (default N=4 only)."""
    running = _running_search_pids()
    have = set()
    for _pid, cmd in running:
        if "--n" in cmd:
            parts = cmd.replace("=", " ").split()
            for i, p in enumerate(parts):
                if p == "--n" and i + 1 < len(parts):
                    try:
                        have.add(int(parts[i + 1]))
                    except ValueError:
                        pass

    log_dir = ROOT / "experiments" / "output" / "prompt_campaign_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    spawned: list[subprocess.Popen] = []
    for n in ns:
        if n in have and not fresh:
            print(f"search n={n} already running — skip launch", flush=True)
            continue
        cmd = [
            _py(),
            str(ROOT / "experiments" / "run_choreography_search.py"),
            "--n",
            str(n),
            "--wall-hours",
            str(wall_hours),
            "--atol-rel",
            str(atol_rel),
            "--max-residual",
            str(max_residual),
        ]
        if not floquet_gate:
            cmd.append("--no-floquet-gate")
        if fresh:
            cmd.append("--fresh")
        log = log_dir / f"choreo_n{n}.log"
        f = open(log, "a", encoding="utf-8")
        f.write(f"\n--- pipeline launch {time.strftime('%Y-%m-%dT%H:%M:%S')} ---\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        spawned.append(proc)
        print(f"START choreo_n{n} pid={proc.pid} → {log}", flush=True)
    return spawned


def run_family_plots(
    *,
    n_families: int,
    min_sep: float,
    periods: float,
    max_frames: int,
    max_residual: float,
) -> None:
    cmd = [
        _py(),
        str(ROOT / "experiments" / "plot_shape_families.py"),
        "--n",
        "4",
        "5",
        "--n-families",
        str(n_families),
        "--min-sep",
        str(min_sep),
        "--periods",
        str(periods),
        "--max-frames",
        str(max_frames),
        "--max-residual",
        str(max_residual),
    ]
    print("+", " ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=str(ROOT))
    if code != 0:
        print(f"plot_shape_families exit={code}", flush=True)


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Auto pipeline: search + periodic family plots")
    p.add_argument(
        "--wall-hours",
        type=float,
        default=0.0,
        help="search wall hours; <=0 unlimited",
    )
    p.add_argument(
        "--plot-every-min",
        type=float,
        default=30.0,
        help="regenerate shape-family plots every N minutes",
    )
    p.add_argument("--n-families", type=int, default=6)
    p.add_argument("--min-sep", type=float, default=0.15)
    p.add_argument("--periods", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=140)
    p.add_argument("--atol-rel", type=float, default=CERTIFY_ATOL_REL)
    p.add_argument("--max-residual", type=float, default=CERTIFY_MAX_RESIDUAL)
    p.add_argument("--fresh", action="store_true", help="clear search DB before launch")
    p.add_argument(
        "--no-floquet-gate",
        action="store_true",
        help="disable Floquet certify on launched searches (not recommended)",
    )
    p.add_argument(
        "--with-n5",
        action="store_true",
        help="also launch N=5 search (off by default; threads 7:2:6:1)",
    )
    p.add_argument(
        "--plot-once",
        action="store_true",
        help="run family plots once at start, then only search (no periodic replot)",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="only ensure searches; skip plotting",
    )
    args = p.parse_args()

    print(campaign_priority_blurb(), flush=True)
    status_path = ROOT / "experiments" / "output" / "pipeline_status.json"
    floquet_gate = FLOQUET_GATE_DEFAULT and not args.no_floquet_gate
    ns: tuple[int, ...] = (4, 5) if (args.with_n5 or CHOREO_N5_IN_DEFAULT_CAMPAIGN) else (4,)
    spawned = ensure_searches(
        wall_hours=args.wall_hours,
        fresh=args.fresh,
        atol_rel=args.atol_rel,
        max_residual=args.max_residual,
        floquet_gate=floquet_gate,
        ns=ns,
    )

    if not args.no_plot:
        print("initial shape-family plot…", flush=True)
        run_family_plots(
            n_families=args.n_families,
            min_sep=args.min_sep,
            periods=args.periods,
            max_frames=args.max_frames,
            max_residual=args.max_residual,
        )

    write_status(
        status_path,
        {
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_hours": args.wall_hours,
            "plot_every_min": args.plot_every_min,
            "spawned_pids": [p.pid for p in spawned],
            "status": "running",
        },
    )

    if args.no_plot or args.plot_once:
        print(
            "pipeline armed (searches running); "
            + ("no periodic replot" if args.plot_once or args.no_plot else ""),
            flush=True,
        )
        # keep parent alive only if we spawned and want to wait — detach style:
        print("Parent exiting; search children keep running.", flush=True)
        return

    interval = max(60.0, float(args.plot_every_min) * 60.0)
    tick = 0
    print(f"periodic replot every {args.plot_every_min} min (interval={interval:.0f}s)", flush=True)
    try:
        while True:
            time.sleep(interval)
            tick += 1
            print(f"--- plot tick {tick} ---", flush=True)
            # relaunch searches if they died
            ensure_searches(
                wall_hours=args.wall_hours,
                fresh=False,
                atol_rel=args.atol_rel,
                max_residual=args.max_residual,
                floquet_gate=floquet_gate,
                ns=ns,
            )
            run_family_plots(
                n_families=args.n_families,
                min_sep=args.min_sep,
                periods=args.periods,
                max_frames=args.max_frames,
                max_residual=args.max_residual,
            )
            write_status(
                status_path,
                {
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "tick": tick,
                    "status": "running",
                    "running_searches": [
                        {"pid": pid, "cmd": cmd[:120]} for pid, cmd in _running_search_pids()
                    ],
                },
            )
    except KeyboardInterrupt:
        write_status(status_path, {"status": "stopped", "tick": tick})
        print("pipeline stopped", flush=True)


if __name__ == "__main__":
    main()
