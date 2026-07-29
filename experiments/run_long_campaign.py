#!/usr/bin/env python3
"""
Long campaign: beam search until a PEO solution, a loss plateau, or wall.

Stop conditions (first hit wins):
  1. solution: loss ≤ target_loss, or E_r_final≤target_Er and E_v_final≤target_Ev
  2. plateau: global best not improved by plateau_rel for plateau_rounds consecutive rounds
  3. wall: optional --wall-min safety budget (0 = disabled)

Example:
  python experiments/run_long_campaign.py --skip-rep-error --wall-min 0 \\
      --target-loss 1.0 --plateau-rounds 12
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from fairy_orbit.observe.rep_error import RepSigmas, load_required_sigmas
from fairy_orbit.observe.search import (
    BeamConfig,
    SearchBounds,
    grid_beam_search,
    result_to_dict,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "output" / "long_campaign"
SIGMAS = ROOT / "experiments" / "output" / "rep_error" / "sigmas.json"
PY = sys.executable

DEFAULT_SEEDS = [
    (1e-6, 0.0),
    (1e-6, 0.03),
    (1e-6, 0.05),
    (1e-4, 0.0),
    (1e-4, 0.03),
    (1e-4, 0.05),
    (1e-3, 0.0),
    (1e-3, 0.03),
    (1e-3, 0.05),
    (1e-2, 0.0),
    (1e-2, 0.03),
    (1e-2, 0.05),
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_rep_error(wall_left: float | None) -> None:
    if wall_left is not None and wall_left < 120.0:
        _log(f"skip rep_error (only {wall_left:.0f}s left)")
        return
    cmd = [
        PY,
        str(ROOT / "experiments" / "run_rep_error_scan.py"),
        "--m",
        "1e-6,1e-4,1e-3,1e-2",
        "--beta",
        "0.9,1.0,1.15",
        "--e",
        "0.0,0.05,0.3,0.6",
        "--rho",
        "1.0",
        "--t-end",
        "6.0",
        "--n-outputs",
        "160",
        "--out",
        str(ROOT / "experiments" / "output" / "rep_error"),
    ]
    _log("=== Stage A: rep_error_scan ===")
    _log(" ".join(cmd))
    subprocess.run(cmd, check=False)


def _is_solution(
    best: dict,
    *,
    target_loss: float,
    target_er: float,
    target_ev: float,
) -> bool:
    loss = float(best["loss"])
    if loss <= target_loss:
        return True
    summary = best.get("summary") or {}
    er = summary.get("E_r_final")
    ev = summary.get("E_v_final")
    if er is None or ev is None:
        return False
    return float(er) <= target_er and float(ev) <= target_ev


def _write_report(
    out: Path,
    *,
    rounds: int,
    stop_reason: str,
    sigmas: RepSigmas,
    ranked: list[dict],
    args: argparse.Namespace,
    global_best: float | None,
    n_expands: int = 0,
    final_bounds: dict | None = None,
) -> None:
    lines = [
        "# Long campaign REPORT",
        "",
        f"stop_reason={stop_reason}",
        f"rounds={rounds}, global_best={global_best}, n_expands={n_expands}",
        f"target_loss={args.target_loss}, target_Er={args.target_Er}, target_Ev={args.target_Ev}",
        f"plateau_rounds={args.plateau_rounds}, plateau_rel={args.plateau_rel}",
        f"wall_min={args.wall_min}, sigmas={sigmas.source}",
        f"final_bounds={json.dumps(final_bounds) if final_bounds else ''}",
        "",
        "| rank | m | e | loss | E_r | E_v | n_evals | free |",
        "|------|---|---|------|-----|-----|---------|------|",
    ]
    for i, r in enumerate(ranked[:20], 1):
        b = r["best"]
        s = b.get("summary") or {}
        er = s.get("E_r_final", "")
        ev = s.get("E_v_final", "")
        er_s = f"{er:.4g}" if isinstance(er, (int, float)) else ""
        ev_s = f"{ev:.4g}" if isinstance(ev, (int, float)) else ""
        lines.append(
            f"| {i} | {r['seed']['m']:.0e} | {r['seed']['e']:.3f} | "
            f"{b['loss']:.6g} | {er_s} | {ev_s} | {r['n_evals']} | "
            f"`{json.dumps(b['params'])}` |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="PEO campaign until solution / plateau / wall")
    p.add_argument(
        "--wall-min",
        type=float,
        default=0.0,
        help="safety wall in minutes; 0 disables wall (stop only on solution/plateau)",
    )
    p.add_argument("--target-loss", type=float, default=1.0)
    p.add_argument("--target-Er", type=float, default=0.05)
    p.add_argument("--target-Ev", type=float, default=0.05)
    p.add_argument(
        "--plateau-rounds",
        type=int,
        default=12,
        help="stop after this many consecutive rounds without relative improvement",
    )
    p.add_argument(
        "--plateau-rel",
        type=float,
        default=0.01,
        help="relative improvement needed to reset plateau counter (e.g. 0.01 = 1%)",
    )
    p.add_argument("--n-outputs", type=int, default=120)
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--coarse", type=int, default=3)
    p.add_argument("--max-evals-per-seed", type=int, default=5000)
    p.add_argument("--skip-rep-error", action="store_true")
    p.add_argument("--wide-bounds", action="store_true", help="widen a1 and kick ranges")
    p.add_argument(
        "--edge-frac",
        type=float,
        default=0.05,
        help="treat param as on-edge if within this fraction of the local span",
    )
    p.add_argument(
        "--expand-grow",
        type=float,
        default=0.5,
        help="when on edge, grow that side by grow*span",
    )
    p.add_argument(
        "--max-expands",
        type=int,
        default=20,
        help="cap adaptive bound expansions (then allow plateau)",
    )
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    deadline = (
        time.perf_counter() + args.wall_min * 60.0 if args.wall_min > 0 else None
    )
    t0 = time.perf_counter()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "seeds").mkdir(exist_ok=True)
    status_path = out / "STATUS.md"

    def write_status(line: str) -> None:
        elapsed = time.perf_counter() - t0
        if deadline is None:
            left_s = "∞"
        else:
            left_s = f"{max(0.0, deadline - time.perf_counter()) / 60:.1f} min"
        text = (
            f"# Long campaign status\n\n"
            f"- mode: until solution / plateau"
            f"{'' if deadline is None else f' / wall={args.wall_min:.1f} min'}\n"
            f"- elapsed ≈ {elapsed/60:.1f} min, left ≈ {left_s}\n"
            f"- target_loss={args.target_loss}, plateau_rounds={args.plateau_rounds}\n"
            f"- last: {line}\n"
        )
        status_path.write_text(text, encoding="utf-8")
        _log(line)

    write_status("started")

    if not args.skip_rep_error:
        left = None if deadline is None else deadline - time.perf_counter()
        run_rep_error(left)
        write_status("rep_error_scan finished")

    sigmas = load_required_sigmas(SIGMAS)
    _log(f"sigmas source={sigmas.source} n={sigmas.n_samples}")

    cfg = BeamConfig(
        beam_width=args.beam,
        coarse_points=args.coarse,
        refine_points=(5, 7, 9),
        n_periods=2.0,
        t_end=None,
        n_outputs=args.n_outputs,
        max_evals=args.max_evals_per_seed,
        bisect_iters=8,
        grad_steps=10,
    )
    if args.wide_bounds:
        bounds = SearchBounds(
            a1=(0.08, 0.50),
            e1=(-0.03, 0.06),
            M1=(0.3, 6.5),
            vx=(-0.08, 0.08),
            vy=(-0.08, 0.08),
            vz=(-0.08, 0.08),
        )
        _log(f"wide bounds a1={bounds.a1} kicks=±0.08")
    else:
        bounds = SearchBounds()

    results: list[dict] = []
    seed_i = 0
    rounds = 0
    global_best: float | None = None
    stagnant = 0
    n_expands = 0
    stop_reason = "unknown"
    expand_log: list[dict] = []

    while True:
        if deadline is not None:
            left = deadline - time.perf_counter()
            if left < 30.0:
                stop_reason = "wall"
                write_status("stopping (wall)")
                break
        else:
            left = None

        m, e = DEFAULT_SEEDS[seed_i % len(DEFAULT_SEEDS)]
        seed_i += 1

        if left is None:
            local_cfg = cfg
            left_label = "∞"
        else:
            est_cap = max(64, int(left / 0.12))
            local_cfg = BeamConfig(
                beam_width=cfg.beam_width,
                coarse_points=cfg.coarse_points,
                refine_points=cfg.refine_points if left > 600 else (5,),
                n_periods=cfg.n_periods,
                t_end=None,
                n_outputs=cfg.n_outputs,
                max_evals=min(cfg.max_evals, est_cap),
                bisect_iters=cfg.bisect_iters if left > 300 else 4,
                grad_steps=cfg.grad_steps if left > 300 else 4,
                epsilon=cfg.epsilon,
                min_dt=cfg.min_dt,
            )
            left_label = f"{left/60:.1f}min"

        write_status(
            f"beam seed m={m:.0e} e={e:.3f} max_evals={local_cfg.max_evals} "
            f"left={left_label} stagnant={stagnant}/{args.plateau_rounds} "
            f"expands={n_expands} bounds_a1={bounds.a1}"
        )
        res = grid_beam_search(m, e, bounds=bounds, config=local_cfg, sigmas=sigmas)
        payload = result_to_dict(res)
        payload["bounds_used"] = {k: list(getattr(bounds, k)) for k in ("a1", "e1", "M1", "vx", "vy", "vz")}
        results.append(payload)
        tag = f"r{rounds:03d}_m{m:.0e}_e{e:.2f}".replace("+", "")
        (out / "seeds" / f"{tag}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        best = payload.get("best")
        loss = None if best is None else float(best["loss"])
        _log(
            f"  done n_evals={res.n_evals} wall={res.wall_s:.1f}s "
            f"best={loss} global={global_best} stagnant={stagnant}"
        )
        rounds += 1

        if best is not None:
            # Expand bounds when the best sits on an edge (before plateau logic).
            edges = bounds.near_edges(best["params"], frac=args.edge_frac)
            if edges and n_expands < args.max_expands:
                new_bounds, changed = bounds.expand_edges(edges, grow=args.expand_grow)
                if changed:
                    bounds = new_bounds
                    n_expands += 1
                    stagnant = 0
                    expand_log.append(
                        {
                            "round": rounds,
                            "edges": edges,
                            "changed": {k: list(v) for k, v in changed.items()},
                            "loss": loss,
                        }
                    )
                    _log(f"  EXPAND #{n_expands} edges={edges} -> {changed}")
                    write_status(f"expanded bounds #{n_expands}: {changed}")

            if global_best is None or loss < global_best * (1.0 - args.plateau_rel):
                if global_best is None or loss < global_best:
                    global_best = loss
                stagnant = 0
            else:
                if loss < global_best:
                    global_best = loss
                # Only count plateau if not currently edge-limited (or expand exhausted).
                still_edged = bool(bounds.near_edges(best["params"], frac=args.edge_frac))
                if still_edged and n_expands < args.max_expands:
                    stagnant = 0
                else:
                    stagnant += 1

            if _is_solution(
                best,
                target_loss=args.target_loss,
                target_er=args.target_Er,
                target_ev=args.target_Ev,
            ):
                stop_reason = "solved"
                write_status(f"SOLVED loss={loss}")
                ranked = sorted(
                    [r for r in results if r.get("best")],
                    key=lambda r: r["best"]["loss"],
                )
                (out / "summary.json").write_text(
                    json.dumps(
                        {
                            "stop_reason": stop_reason,
                            "sigmas_source": sigmas.source,
                            "n_rounds": rounds,
                            "global_best": global_best,
                            "n_expands": n_expands,
                            "expand_log": expand_log,
                            "final_bounds": {
                                k: list(getattr(bounds, k))
                                for k in ("a1", "e1", "M1", "vx", "vy", "vz")
                            },
                            "results": results,
                            "top": ranked[:10],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                break

            if stagnant >= args.plateau_rounds:
                stop_reason = "plateau"
                write_status(
                    f"PLATEAU global_best={global_best} "
                    f"no {args.plateau_rel:.1%} gain for {stagnant} rounds "
                    f"(expands={n_expands})"
                )
                ranked = sorted(
                    [r for r in results if r.get("best")],
                    key=lambda r: r["best"]["loss"],
                )
                (out / "summary.json").write_text(
                    json.dumps(
                        {
                            "stop_reason": stop_reason,
                            "sigmas_source": sigmas.source,
                            "n_rounds": rounds,
                            "global_best": global_best,
                            "n_expands": n_expands,
                            "expand_log": expand_log,
                            "final_bounds": {
                                k: list(getattr(bounds, k))
                                for k in ("a1", "e1", "M1", "vx", "vy", "vz")
                            },
                            "results": results,
                            "top": ranked[:10],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                break
        else:
            stagnant += 1
            if stagnant >= args.plateau_rounds:
                stop_reason = "plateau"
                write_status("PLATEAU (no successful bests)")
                break

        ranked = sorted(
            [r for r in results if r.get("best")],
            key=lambda r: r["best"]["loss"],
        )
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "stop_reason": "running",
                    "sigmas_source": sigmas.source,
                    "n_rounds": rounds,
                    "global_best": global_best,
                    "stagnant": stagnant,
                    "n_expands": n_expands,
                    "expand_log": expand_log,
                    "bounds": {
                        k: list(getattr(bounds, k))
                        for k in ("a1", "e1", "M1", "vx", "vy", "vz")
                    },
                    "results": results,
                    "top": ranked[:10],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    ranked = sorted(
        [r for r in results if r.get("best")],
        key=lambda r: r["best"]["loss"],
    )
    _write_report(
        out,
        rounds=rounds,
        stop_reason=stop_reason,
        sigmas=sigmas,
        ranked=ranked,
        args=args,
        global_best=global_best,
        n_expands=n_expands,
        final_bounds={
            k: list(getattr(bounds, k)) for k in ("a1", "e1", "M1", "vx", "vy", "vz")
        },
    )
    write_status(f"finished stop={stop_reason} rounds={rounds} → REPORT.md")
    _log(f"done stop={stop_reason} → {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
