"""Orbit trajectory visualization (static + HTML)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from fairy_orbit.engine.trajectory import Trajectory
from fairy_orbit.observe.diagnose import Diagnosis


# Distinct colors for T1..T4
FAIRY_COLORS = ["#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#d35400"]


def plot_orbits_xy(
    trajectory: Trajectory,
    out_path: str | Path,
    *,
    title: str = "Orbital ladder — x–y",
    encounters: list | None = None,
    stride: int = 1,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter([0], [0], c="k", s=40, zorder=6, label="central")
    color_i = 0
    for i, lab in enumerate(trajectory.labels):
        if lab == "central":
            continue
        c = FAIRY_COLORS[color_i % len(FAIRY_COLORS)]
        color_i += 1
        x = trajectory.positions[::stride, i, 0]
        y = trajectory.positions[::stride, i, 1]
        ax.plot(x, y, color=c, lw=0.7, alpha=0.85, label=lab)
        ax.scatter(x[0], y[0], color=c, s=28, zorder=5)
        ax.scatter(x[-1], y[-1], color=c, s=28, marker="x", zorder=5)

    if encounters:
        xs = [e.position_mid[0] for e in encounters]
        ys = [e.position_mid[1] for e in encounters]
        ax.scatter(xs, ys, c="k", s=16, alpha=0.7, zorder=7, label="encounters")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_orbits_3d(
    trajectory: Trajectory,
    out_path: str | Path,
    *,
    title: str = "Orbital ladder — 3D",
    stride: int = 2,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([0], [0], [0], c="k", s=40, label="central")
    color_i = 0
    for i, lab in enumerate(trajectory.labels):
        if lab == "central":
            continue
        c = FAIRY_COLORS[color_i % len(FAIRY_COLORS)]
        color_i += 1
        p = trajectory.positions[::stride, i]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=c, lw=0.6, alpha=0.85, label=lab)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    # Equal-ish aspect
    pts = trajectory.positions.reshape(-1, 3)
    span = float(np.max(np.abs(pts))) * 1.05
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(-span, span)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_orbit_gallery(
    diagnoses: list[tuple[str, Diagnosis]],
    out_path: str | Path,
    *,
    cols: int = 3,
) -> Path:
    """Small-multiple x–y orbits for comparing initial parameters."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(diagnoses)
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows), squeeze=False)

    for idx, (label, d) in enumerate(diagnoses):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        traj = d.trajectory
        ax.scatter([0], [0], c="k", s=18, zorder=5)
        ci = 0
        for i, lab in enumerate(traj.labels):
            if lab == "central":
                continue
            color = FAIRY_COLORS[ci % len(FAIRY_COLORS)]
            ci += 1
            ax.plot(
                traj.positions[:, i, 0],
                traj.positions[:, i, 1],
                color=color,
                lw=0.5,
                alpha=0.8,
            )
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.suptitle("Orbit gallery — varying initial parameters", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def export_html_viewer(
    trajectory: Trajectory,
    out_path: str | Path,
    *,
    title: str = "Ladder orbit viewer",
    max_frames: int = 250,
    trail: int = 40,
) -> Path:
    """Lightweight Plotly HTML animation for a Trajectory."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    T = len(trajectory)
    step = max(1, T // max_frames)
    times = trajectory.times[::step]
    pos = trajectory.positions[::step]
    labels = trajectory.labels

    frames = []
    for fi in range(len(times)):
        data = []
        i0 = max(0, fi - trail)
        for i, lab in enumerate(labels):
            color = "#111111" if lab == "central" else FAIRY_COLORS[(i - 1) % len(FAIRY_COLORS)]
            trail_pts = pos[i0 : fi + 1, i]
            data.append(
                {
                    "type": "scatter3d",
                    "mode": "lines+markers",
                    "x": trail_pts[:, 0].tolist(),
                    "y": trail_pts[:, 1].tolist(),
                    "z": trail_pts[:, 2].tolist(),
                    "name": lab,
                    "line": {"width": 3, "color": color},
                    "marker": {"size": 3 if lab != "central" else 5, "color": color},
                }
            )
        frames.append({"name": f"f{fi}", "data": data})

    init = frames[0]["data"] if frames else []
    span = float(np.max(np.abs(pos))) * 1.1
    layout = {
        "title": title,
        "scene": {
            "aspectmode": "cube",
            "xaxis": {"range": [-span, span]},
            "yaxis": {"range": [-span, span]},
            "zaxis": {"range": [-span, span]},
        },
        "updatemenus": [
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 40, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        "sliders": [
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "t = ",
                    "suffix": "",
                    "xanchor": "right",
                    "font": {"size": 14},
                },
                "pad": {"t": 50, "b": 10},
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [f"f{i}"],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": f"{times[i]:.3g}",
                    }
                    for i in range(len(times))
                ],
                "x": 0.08,
                "len": 0.84,
            }
        ],
    }

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>html,body{{margin:0;height:100%;font-family:Georgia,serif;background:#f7f4ef;}}
  #v{{width:100vw;height:100vh;}}</style>
</head>
<body>
  <div id="v"></div>
  <script>
    const data = {json.dumps(init)};
    const frames = {json.dumps(frames)};
    const layout = {json.dumps(layout)};
    Plotly.newPlot('v', data, layout).then(() => Plotly.addFrames('v', frames));
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path
