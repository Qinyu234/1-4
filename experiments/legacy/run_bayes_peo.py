#!/usr/bin/env python3
"""
Bayesian (TPE) search for ABCD→BCDA/CDAB/DABC PEO with staged escalation.

Stagnation (no success for --stagnate trials) → expand bounds, then unlock
high-order knobs one stage at a time: a2 → e2 → M2 → (v1x,v1y,v1z).

  q_i = q0 + i q1 + i² q2
  δv_i = R_i · (v + i v₁)

Example:
  python experiments/run_bayes_peo.py --n-trials 200 --stagnate 40 --out experiments/output/bayes_peo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.observe.bayes import BayesSpace, UNLOCK_STAGES, run_bayes_search
from fairy_orbit.observe.rep_error import load_required_sigmas

ROOT = Path(__file__).resolve().parents[1]
SIGMAS = ROOT / "experiments" / "output" / "rep_error" / "sigmas.json"
OUT_DEFAULT = ROOT / "experiments" / "output" / "bayes_peo"


def _log(msg: str) -> None:
    print(msg, flush=True)


def plot_history(history, events, out: Path) -> None:
    losses = np.array([h.loss for h in history], dtype=float)
    soft = np.array([h.soft_choreo for h in history], dtype=float)
    stages = np.array([h.stage for h in history], dtype=int)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].semilogy(np.maximum(losses, 1e-12), ".-", ms=3)
    for ev in events:
        axes[0].axvline(ev.trial_index, color="C3" if ev.action == "unlock" else "C2", alpha=0.5, ls="--")
    axes[0].set_xlabel("trial")
    axes[0].set_ylabel("BO loss")
    axes[0].set_title("objective (+ escalate)")
    axes[0].grid(True, alpha=0.3)
    ok = np.isfinite(soft)
    axes[1].plot(np.where(ok)[0], soft[ok], ".-", ms=3, color="C1")
    axes[1].set_xlabel("trial")
    axes[1].set_ylabel("soft_choreo")
    axes[1].set_title("soft radial residual")
    axes[1].grid(True, alpha=0.3)
    axes[2].step(np.arange(len(stages)), stages, where="post")
    axes[2].set_xlabel("trial")
    axes[2].set_ylabel("unlock stage")
    axes[2].set_title("a2→e2→M2→v1")
    axes[2].set_yticks(range(len(UNLOCK_STAGES)))
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "history.png", dpi=140)
    plt.close(fig)


def plot_me_scatter(history, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ms = np.array([h.m for h in history])
    es = np.array([h.e for h in history])
    losses = np.array([h.loss for h in history])
    success = np.array([h.status == "success" for h in history])
    sc = ax.scatter(
        np.log10(ms[~success]),
        es[~success],
        c=np.log10(np.maximum(losses[~success], 1e-12)),
        cmap="magma",
        s=28,
        alpha=0.85,
        label="fail",
    )
    if success.any():
        ax.scatter(
            np.log10(ms[success]),
            es[success],
            c="cyan",
            s=80,
            marker="*",
            edgecolors="k",
            label="success",
            zorder=5,
        )
    fig.colorbar(sc, ax=ax, fraction=0.046, label=r"$\log_{10}$ BO loss")
    ax.set_xlabel(r"$\log_{10} m$")
    ax.set_ylabel("e")
    ax.set_title("Bayesian trials")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "me_scatter.png", dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Bayesian PEO search (full / staged high-order)")
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--n-periods", type=float, default=2.0)
    p.add_argument("--n-outputs", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stagnate", type=int, default=40, help="trials without success → escalate")
    p.add_argument("--expand-grow", type=float, default=0.35)
    p.add_argument("--max-expands", type=int, default=8)
    p.add_argument("--log-m-lo", type=float, default=-6.0)
    p.add_argument("--log-m-hi", type=float, default=-2.0)
    p.add_argument("--e-lo", type=float, default=0.0)
    p.add_argument("--e-hi", type=float, default=0.20)
    p.add_argument("--a1-hi", type=float, default=1.0)
    p.add_argument("--kick", type=float, default=0.25, help="|v| kick half-width")
    p.add_argument(
        "--full",
        action="store_true",
        help="suggest all knobs (a2,e2,M2,v1*) from trial 0 — no staged unlock",
    )
    p.add_argument(
        "--wide",
        action="store_true",
        help="very wide search box for all free params",
    )
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sigmas = load_required_sigmas(SIGMAS)

    if args.wide:
        args.log_m_lo, args.log_m_hi = -7.0, -1.5
        args.e_lo, args.e_hi = 0.0, 0.40
        args.a1_hi = 1.5
        args.kick = 0.40
        space = BayesSpace(
            log_m=(args.log_m_lo, args.log_m_hi),
            e=(args.e_lo, args.e_hi),
            a1=(0.05, args.a1_hi),
            e1=(-0.20, 0.30),
            M1=(0.2, 10.0),
            vx=(-args.kick, args.kick),
            vy=(-args.kick, args.kick),
            vz=(-args.kick, args.kick),
            a2=(-0.12, 0.12),
            e2=(-0.06, 0.06),
            M2=(-1.8, 1.8),
            v1x=(-0.15, 0.15),
            v1y=(-0.15, 0.15),
            v1z=(-0.15, 0.15),
        )
    else:
        space = BayesSpace(
            log_m=(args.log_m_lo, args.log_m_hi),
            e=(args.e_lo, args.e_hi),
            a1=(0.05, args.a1_hi),
            vx=(-args.kick, args.kick),
            vy=(-args.kick, args.kick),
            vz=(-args.kick, args.kick),
        )

    # Fresh DB for full/wide runs (avoid clobbering old staged studies)
    db_name = "study_full.db" if args.full else "study.db"
    storage = f"sqlite:///{(out / db_name).as_posix()}"

    def on_event(ev) -> None:
        _log(f"  ESCALATE @{ev.trial_index} {ev.action} {ev.detail}")

    unlocked0 = list(UNLOCK_STAGES[-1]) if args.full else list(UNLOCK_STAGES[0])
    _log(
        f"bayes n_trials={args.n_trials} full={args.full} wide={args.wide} "
        f"stagnate={args.stagnate} unlocked0={unlocked0} "
        f"space log_m={space.log_m} e={space.e} a1={space.a1} kick=±{args.kick} "
        f"sigmas={sigmas.source}"
    )
    (out / "STATUS.md").write_text(
        f"# Bayes PEO (running)\n\n- full={args.full} wide={args.wide} n_trials={args.n_trials}\n"
        f"- unlocked={unlocked0}\n",
        encoding="utf-8",
    )
    t0 = time.perf_counter()
    study, history, events = run_bayes_search(
        n_trials=args.n_trials,
        space=space,
        n_periods=args.n_periods,
        n_outputs=args.n_outputs,
        sigmas=sigmas,
        seed=args.seed,
        storage=storage,
        study_name="peo_bayes_full" if args.full else "peo_bayes",
        stagnate_trials=args.stagnate,
        expand_grow=args.expand_grow,
        max_expands=args.max_expands,
        unlock_all=args.full,
        on_event=on_event,
    )
    wall = time.perf_counter() - t0

    best = study.best_trial
    n_success = sum(1 for h in history if h.status == "success")
    n_choreo = sum(1 for h in history if h.status == "choreography")
    rows = [
        {
            "loss": h.loss,
            "status": h.status,
            "m": h.m,
            "e": h.e,
            "free": h.free.as_dict(),
            "soft_choreo": h.soft_choreo,
            "score": h.summary.get("score"),
            "choreography_shift_k": h.summary.get("choreography_shift_k"),
            "reason": h.summary.get("reason"),
            "stage": h.stage,
            "unlocked": list(h.unlocked),
        }
        for h in history
    ]
    payload = {
        "n_trials": len(history),
        "wall_s": wall,
        "n_success": n_success,
        "n_choreography": n_choreo,
        "full": args.full,
        "wide": args.wide,
        "best_value": float(study.best_value),
        "best_params": dict(best.params),
        "best_attrs": dict(best.user_attrs),
        "stagnate_trials": args.stagnate,
        "events": [
            {"trial_index": e.trial_index, "action": e.action, "detail": e.detail}
            for e in events
        ],
        "unlock_stages": [list(s) for s in UNLOCK_STAGES],
        "space": {
            "log_m": list(space.log_m),
            "e": list(space.e),
            "a1": list(space.a1),
            "kick": args.kick,
            "a2": list(space.a2),
            "M2": list(space.M2),
        },
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_history(history, events, out)
    plot_me_scatter(history, out)

    lines = [
        "# Bayesian PEO search",
        "",
        f"full={args.full}, wide={args.wide}",
        f"trials={len(history)}, wall={wall/60:.1f} min, success={n_success}, choreography={n_choreo}",
        f"best_value={study.best_value:.6g} status={best.user_attrs.get('status')}",
        f"best_params={json.dumps(best.params)}",
        f"escalations={len(events)}",
        "",
        "Soft residual guides TPE when hard radial gate rejects.",
        "Plots: `history.png`, `me_scatter.png`.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "STATUS.md").write_text(
        f"# Bayes PEO\n\n- finished trials={len(history)} success={n_success}\n"
        f"- best={study.best_value:.6g} escalations={len(events)}\n",
        encoding="utf-8",
    )
    _log(
        f"done success={n_success}/{len(history)} best={study.best_value:.6g} "
        f"status={best.user_attrs.get('status')} escalations={len(events)} → {out / 'REPORT.md'}"
    )


if __name__ == "__main__":
    main()
