"""Plot helpers for orbit experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fairy_orbit.search.grid import Candidate
from fairy_orbit.simulation.runner import Trajectory


def plot_trajectories(
    traj: Trajectory,
    path: str | Path,
    fairy_indices: tuple[int, ...] = (1, 2, 3, 4),
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    # Planet
    ax.plot(
        traj.positions[:, 0, 0],
        traj.positions[:, 0, 1],
        traj.positions[:, 0, 2],
        color="gray",
        lw=0.8,
        label=traj.labels[0] if traj.labels else "Planet",
    )
    colors = ["C0", "C1", "C2", "C3"]
    for k, idx in enumerate(fairy_indices):
        ax.plot(
            traj.positions[:, idx, 0],
            traj.positions[:, idx, 1],
            traj.positions[:, idx, 2],
            color=colors[k % len(colors)],
            lw=1.2,
            label=traj.labels[idx] if idx < len(traj.labels) else f"fairy_{idx}",
        )
        ax.scatter(
            traj.positions[0, idx, 0],
            traj.positions[0, idx, 1],
            traj.positions[0, idx, 2],
            color=colors[k % len(colors)],
            s=30,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_energy_error(traj: Trajectory, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    e0 = traj.energies[0]
    rel = (traj.energies - e0) / max(abs(e0), 1e-30)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(traj.times, rel)
    ax.set_xlabel("t")
    ax.set_ylabel("(E(t)-E(0))/|E(0)|")
    ax.set_title("Energy drift")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_score_heatmap(
    candidates: list[Candidate],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rads = sorted({c.v_rad for c in candidates})
    tans = sorted({c.v_tan for c in candidates})
    grid = np.full((len(tans), len(rads)), np.nan)
    lookup = {(c.v_rad, c.v_tan): c.score for c in candidates}
    for i, vt in enumerate(tans):
        for j, vr in enumerate(rads):
            grid[i, j] = lookup.get((vr, vt), np.nan)
    fig, ax = plt.subplots(figsize=(6, 5))
    r0, r1 = rads[0], rads[-1]
    t0, t1 = tans[0], tans[-1]
    if r0 == r1:
        r0, r1 = r0 - 0.5, r1 + 0.5
    if t0 == t1:
        t0, t1 = t0 - 0.5, t1 + 0.5
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[r0, r1, t0, t1],
    )
    fig.colorbar(im, ax=ax, label="score")
    ax.set_xlabel("v_rad")
    ax.set_ylabel("v_tan")
    ax.set_title("Grid scan scores")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
