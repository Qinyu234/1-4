#!/usr/bin/env python3
"""
Performance analysis for the orbital-ladder campaign / diagnose hot path.

1. Summarize wall-time from campaign runs.jsonl by phase
2. cProfile a representative diagnose() call
3. Write PERF.md + timing plots under experiments/output/perf/
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.core import SystemConfig
from fairy_orbit.design import LadderParams, build_orbital_ladder
from fairy_orbit.observe import diagnose

ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_rows(jsonl: Path) -> list[dict]:
    rows: list[dict] = []
    if not jsonl.exists():
        return rows
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def summarize_campaign(rows: list[dict]) -> dict:
    by_phase: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if "elapsed_s" in r and r.get("elapsed_s") is not None:
            by_phase[str(r.get("phase", "?"))].append(float(r["elapsed_s"]))

    phase_stats = {}
    for phase, xs in sorted(by_phase.items()):
        arr = np.asarray(xs, dtype=float)
        phase_stats[phase] = {
            "n": int(arr.size),
            "total_s": float(arr.sum()),
            "mean_s": float(arr.mean()),
            "median_s": float(np.median(arr)),
            "p95_s": float(np.percentile(arr, 95)),
            "max_s": float(arr.max()),
        }

    totals = [float(r["elapsed_s"]) for r in rows if r.get("elapsed_s") is not None]
    return {
        "n_rows": len(rows),
        "n_timed": len(totals),
        "wall_sum_s": float(sum(totals)) if totals else 0.0,
        "wall_sum_h": float(sum(totals) / 3600.0) if totals else 0.0,
        "phases": phase_stats,
    }


def plot_timing(rows: list[dict], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    timed = [r for r in rows if r.get("elapsed_s") is not None]
    if not timed:
        return paths

    phases = sorted({str(r.get("phase", "?")) for r in timed})
    data = [[float(r["elapsed_s"]) for r in timed if str(r.get("phase")) == p] for p in phases]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].boxplot(data, tick_labels=phases, showfliers=False)
    axes[0].set_ylabel("elapsed_s")
    axes[0].set_title("Per-case runtime by phase")
    axes[0].tick_params(axis="x", rotation=20)

    # cumulative time share
    totals = [sum(xs) for xs in data]
    axes[1].bar(phases, totals, color="#2c5f6e")
    axes[1].set_ylabel("sum elapsed_s")
    axes[1].set_title("Total compute time by phase")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    p = out_dir / "timing_by_phase.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths["timing_by_phase"] = str(p)

    # elapsed vs t_end if present
    xs, ys, cs = [], [], []
    for r in timed:
        te = r.get("t_end_requested") or r.get("t_end")
        if te is None:
            continue
        xs.append(float(te))
        ys.append(float(r["elapsed_s"]))
        cs.append(str(r.get("phase", "?")))
    if xs:
        fig2, ax = plt.subplots(figsize=(7, 4.5))
        phase_colors = {
            "scout": "#1b9e77",
            "map": "#d95f02",
            "deep": "#7570b3",
            "archive": "#e7298a",
            "soak": "#66a61e",
        }
        for phase in sorted(set(cs)):
            mask = [c == phase for c in cs]
            ax.scatter(
                [x for x, m in zip(xs, mask) if m],
                [y for y, m in zip(ys, mask) if m],
                s=18,
                alpha=0.7,
                label=phase,
                c=phase_colors.get(phase, "#333333"),
            )
        ax.set_xlabel("t_end requested")
        ax.set_ylabel("elapsed_s")
        ax.set_title("Runtime vs integration length")
        ax.legend(fontsize=8)
        fig2.tight_layout()
        p2 = out_dir / "timing_vs_tend.png"
        fig2.savefig(p2, dpi=140)
        plt.close(fig2)
        paths["timing_vs_tend"] = str(p2)

    return paths


def profile_diagnose(out_dir: Path, *, t_end: float = 500.0, with_megno: bool = True) -> dict:
    """cProfile a single diagnose hot path; return top cumulative stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SystemConfig(mass_ratio=1e-4)
    params = LadderParams(eccentricity=0.12)
    system = build_orbital_ladder(cfg, params)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    diagnosis = diagnose(
        system,
        cfg,
        t_end=t_end,
        n_outputs=600,
        ladder=params,
        run_megno=with_megno,
    )
    pr.disable()
    wall = time.perf_counter() - t0

    prof_path = out_dir / "diagnose.prof"
    pr.dump_stats(str(prof_path))

    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    stats.print_stats(40)
    text = buf.getvalue()
    (out_dir / "diagnose_cprofile.txt").write_text(text, encoding="utf-8")

    # Parse top callers roughly from pstats via get_stats_profile if available
    top = []
    for func, (cc, nc, tt, ct, callers) in sorted(
        stats.stats.items(), key=lambda kv: kv[1][3], reverse=True
    )[:25]:
        filename, line, name = func
        top.append(
            {
                "file": str(filename),
                "line": line,
                "func": name,
                "cumtime": ct,
                "tottime": tt,
                "ncalls": nc,
            }
        )

    return {
        "wall_s": wall,
        "status": diagnosis.summary.get("status"),
        "megno": diagnosis.summary.get("megno"),
        "prof": str(prof_path),
        "text": str(out_dir / "diagnose_cprofile.txt"),
        "top": top,
        "t_end": t_end,
        "with_megno": with_megno,
    }


def write_perf_md(out_dir: Path, campaign: dict, profile: dict, plot_paths: dict) -> Path:
    lines = [
        "# Performance report",
        "",
        f"Generated: {_now()}",
        "",
        "## Campaign wall time (from runs.jsonl)",
        "",
        f"- rows: **{campaign['n_rows']}** (timed: {campaign['n_timed']})",
        f"- sum of case elapsed: **{campaign['wall_sum_h']:.3f} h** ({campaign['wall_sum_s']:.1f} s)",
        "",
        "| phase | n | total_s | mean_s | median_s | p95_s | max_s |",
        "|-------|---|---------|--------|----------|-------|-------|",
    ]
    for phase, st in campaign.get("phases", {}).items():
        lines.append(
            f"| {phase} | {st['n']} | {st['total_s']:.1f} | {st['mean_s']:.2f} | "
            f"{st['median_s']:.2f} | {st['p95_s']:.2f} | {st['max_s']:.2f} |"
        )

    lines += [
        "",
        "## Hot-path cProfile (`diagnose`)",
        "",
        f"- wall: **{profile['wall_s']:.2f} s** (t_end={profile['t_end']}, megno={profile['with_megno']})",
        f"- status={profile.get('status')} megno={profile.get('megno')}",
        f"- raw: `{profile['prof']}`, `{profile['text']}`",
        "",
        "| cumtime | tottime | ncalls | function |",
        "|---------|---------|--------|----------|",
    ]
    for row in profile.get("top", [])[:15]:
        short = Path(row["file"]).name
        lines.append(
            f"| {row['cumtime']:.3f} | {row['tottime']:.3f} | {row['ncalls']} | "
            f"`{short}:{row['line']}` `{row['func']}` |"
        )

    if plot_paths:
        lines += ["", "## Plots", ""]
        for k, p in plot_paths.items():
            lines.append(f"- {k}: `{p}`")

    lines += [
        "",
        "## Notes",
        "",
        "- Campaign `elapsed_s` is wall time per case (includes MEGNO pass when enabled).",
        "- Bottlenecks in cProfile usually sit in REBOUND `integrate`, then elements/`from_state`.",
        "- If soak dominates total time, prefer longer `t_end` on fewer seeds over many short MEGNO runs.",
        "",
    ]
    path = out_dir / "PERF.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Campaign + diagnose performance analysis")
    parser.add_argument(
        "--campaign",
        type=Path,
        default=ROOT / "experiments" / "output" / "dynamics",
        help="directory containing runs.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "output" / "perf",
    )
    parser.add_argument("--t-end", type=float, default=500.0)
    parser.add_argument("--no-megno", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.campaign / "runs.jsonl")
    campaign = summarize_campaign(rows)
    plot_paths = plot_timing(rows, out)

    if args.skip_profile:
        profile = {
            "wall_s": float("nan"),
            "status": None,
            "megno": None,
            "prof": "",
            "text": "",
            "top": [],
            "t_end": args.t_end,
            "with_megno": not args.no_megno,
        }
    else:
        profile = profile_diagnose(out, t_end=args.t_end, with_megno=not args.no_megno)

    report = write_perf_md(out, campaign, profile, plot_paths)
    summary = {"campaign": campaign, "profile": {k: v for k, v in profile.items() if k != "top"}, "plots": plot_paths}
    (out / "perf_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"rows={campaign['n_rows']} wall_sum_h={campaign['wall_sum_h']:.3f}")
    print(f"report={report}")
    for k, p in plot_paths.items():
        print(f"{k}={p}")
    if not args.skip_profile:
        print(f"diagnose_wall_s={profile['wall_s']:.2f}")
        if profile["top"]:
            top = profile["top"][0]
            print(f"top_cum={Path(top['file']).name}:{top['line']} {top['func']} {top['cumtime']:.3f}s")


if __name__ == "__main__":
    main()
