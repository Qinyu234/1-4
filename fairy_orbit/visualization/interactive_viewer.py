"""Simple interactive orbit viewer generator for local exploration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairy_orbit.icgen.generator import generate_system, orbital_period
from fairy_orbit.simulation.runner import Trajectory, run



def export_interactive_viewer(traj: Trajectory, path: str | Path) -> Path:
    """Export a simple HTML viewer that animates the trajectory in the browser."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    coords = traj.positions
    frames: list[dict[str, Any]] = []
    for idx, frame in enumerate(coords):
        frame_data = {
            "x": [float(frame[0, 0]), float(frame[1, 0]), float(frame[2, 0]), float(frame[3, 0]), float(frame[4, 0])],
            "y": [float(frame[0, 1]), float(frame[1, 1]), float(frame[2, 1]), float(frame[3, 1]), float(frame[4, 1])],
            "z": [float(frame[0, 2]), float(frame[1, 2]), float(frame[2, 2]), float(frame[3, 2]), float(frame[4, 2])],
            "mode": "markers+lines",
            "type": "scatter3d",
            "name": f"frame_{idx}",
        }
        frames.append(frame_data)

    payload = {
        "frames": [
            {
                "data": [
                    {
                        "x": [float(frame_data["x"][i]) for i in range(5)],
                        "y": [float(frame_data["y"][i]) for i in range(5)],
                        "z": [float(frame_data["z"][i]) for i in range(5)],
                        "mode": "markers+lines",
                        "type": "scatter3d",
                        "name": f"frame_{idx}",
                    }
                ],
                "name": f"frame_{idx}",
            }
            for idx, frame_data in enumerate(frames)
        ],
        "layout": {
            "title": "Interactive Orbit Viewer",
            "scene": {"aspectmode": "cube"},
        },
    }

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>Interactive Orbit Viewer</title>
    <script src=\"https://cdn.plot.ly/plotly-3.7.0.min.js\"></script>
    <style>body{{margin:0;padding:0;font-family:sans-serif;}} #viewer{{width:100vw;height:100vh;}}</style>
  </head>
  <body>
    <div id=\"viewer\"></div>
    <script>
      const frames = {json.dumps(payload['frames'])};
      const layout = {json.dumps(payload['layout'])};
      const trace = {{
        type: 'scatter3d',
        mode: 'markers+lines',
        x: [],
        y: [],
        z: [],
        line: {{color: '#4f46e5', width: 2}},
        marker: {{size: 4, color: '#ef4444'}}
      }};
      const data = [trace];
      const viewer = document.getElementById('viewer');
      Plotly.newPlot(viewer, data, layout);
      let frameIndex = 0;
      function updateFrame() {{
        const frame = frames[frameIndex];
        const currentTrace = frame.data[0];
        Plotly.restyle(viewer, {{x: [currentTrace.x], y: [currentTrace.y], z: [currentTrace.z]}}, [0]);
        frameIndex = (frameIndex + 1) % frames.length;
      }}
      setInterval(updateFrame, 80);
    </script>
  </body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a simple interactive orbit viewer")
    parser.add_argument("--out", type=str, default="experiments/output/orbit_viewer.html")
    parser.add_argument("--radius", type=float, default=20.0)
    parser.add_argument("--v-rad", type=float, default=0.1)
    parser.add_argument("--v-tan", type=float, default=0.95)
    parser.add_argument("--solver-type", choices=["own", "rebound"], default="own")
    args = parser.parse_args()

    from fairy_orbit.icgen.tetrahedron import escape_speed

    vesc = escape_speed(1.0, 1.0, args.radius)
    system = generate_system(args.v_rad * vesc, args.v_tan * vesc, radius=args.radius)
    period = orbital_period(1.0, 1.0, args.radius)
    traj = run(system, dt=period / 100, t_end=period, record_every=1, solver_type=args.solver_type)
    out_path = export_interactive_viewer(traj, args.out)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
