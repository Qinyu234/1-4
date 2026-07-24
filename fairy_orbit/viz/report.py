"""Plot and JSON report for the orbital-ladder experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.observe.diagnose import Diagnosis


def save_ladder_report(diagnosis: Diagnosis, out_dir: str | Path) -> dict[str, str]:
    """Write plots + summary JSON; return map of artifact paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    el = diagnosis.elements
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for j, label in enumerate(el.labels):
        axes[0].plot(el.times, el.a[:, j], label=label)
        axes[1].plot(el.times, el.e[:, j], label=label)
    axes[0].set_ylabel("a")
    axes[0].set_title("Semi-major axis evolution")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_ylabel("e")
    axes[1].set_title("Eccentricity evolution")

    for k, plabel in enumerate(diagnosis.resonance.pair_labels):
        axes[2].plot(
            diagnosis.resonance.times,
            diagnosis.resonance.angles[:, k],
            label=plabel,
        )
    axes[2].set_ylabel("φ")
    axes[2].set_xlabel("t")
    axes[2].set_title("Resonance angles")
    axes[2].legend(loc="best", fontsize=8)
    fig.tight_layout()
    elem_path = out / "elements.png"
    fig.savefig(elem_path, dpi=140)
    plt.close(fig)
    paths["elements"] = str(elem_path)

    # Encounter scatter in x–y
    fig2, ax = plt.subplots(figsize=(7, 7))
    traj = diagnosis.trajectory
    for i, lab in enumerate(traj.labels):
        if lab == "central":
            continue
        ax.plot(traj.positions[:, i, 0], traj.positions[:, i, 1], lw=0.6, alpha=0.7, label=lab)
    if diagnosis.encounters:
        xs = [e.position_mid[0] for e in diagnosis.encounters]
        ys = [e.position_mid[1] for e in diagnosis.encounters]
        ax.scatter(xs, ys, c="k", s=18, zorder=5, label="encounters")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectories and encounters")
    ax.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    enc_path = out / "encounters.png"
    fig2.savefig(enc_path, dpi=140)
    plt.close(fig2)
    paths["encounters"] = str(enc_path)

    summary = dict(diagnosis.summary)
    summary["encounters"] = [
        {
            "time": e.time,
            "pair": [e.label_i, e.label_j],
            "distance": e.distance,
            "position_mid": e.position_mid.tolist(),
        }
        for e in diagnosis.encounters
    ]
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths["summary"] = str(summary_path)
    return paths
